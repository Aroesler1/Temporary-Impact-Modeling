#!/usr/bin/env python3
"""Conditional impact accuracy on held-out metaorders, all fifteen sessions.

The headline number of the propagator section: given an order's size and the
seconds it executed over, how close was the predicted impact to the realised
one, on orders the model never saw.

Eight models, every parameter fitted on the first 70% of the session and applied
unchanged to the last 30%. The two time-of-day profiles are estimated from OTHER
sessions of the same symbol, never from the scored one, and never across
symbols.

Usage:
    python scripts/run_conditional_impact.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import panel  # noqa: E402
from conditional_impact import MODEL_ORDER, evaluate_session, halfhour_ratios  # noqa: E402

ARTIFACTS = (
    "summary.csv",
    "calibration_by_decile.csv",
    "calibration_pooled.csv",
    "model_comparison.csv",
    "historical_vs_corrected.csv",
    "input_manifest.csv",
    "methodology.csv",
)

# Fitted floats are mathematical results, not a portable byte serialization.
# These bounds are far below the precision of published claims. Counts,
# labels, input hashes, missingness and row order remain exact.
FLOAT_RTOL = 1e-8
FLOAT_ATOL = 1e-10


def verify_artifacts(expected_dir: Path, rebuilt_dir: Path) -> None:
    for name in ARTIFACTS:
        stored = pd.read_csv(expected_dir / name)
        rebuilt = pd.read_csv(rebuilt_dir / name)
        pd.testing.assert_index_equal(stored.columns, rebuilt.columns)
        if stored.shape != rebuilt.shape:
            raise ValueError(f"{name}: artifact shape differs")
        for column in stored:
            left, right = stored[column], rebuilt[column]
            floating = (pd.api.types.is_float_dtype(left.dtype)
                        and pd.api.types.is_float_dtype(right.dtype))
            pd.testing.assert_series_equal(
                left, right, check_exact=not floating,
                rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
                obj=f"{name}/{column}",
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_manifest() -> pd.DataFrame:
    """Hash every committed aggregate input used by the reproduction."""
    paths = [
        panel.DATA / "session_meta.csv",
        panel.DATA.parent
        / "reports"
        / "conditional_impact"
        / "model_comparison.csv",
    ]
    for session in panel.session_keys():
        paths.extend(
            [
                panel.DATA / f"{session}_1s.csv",
                panel.DATA / f"{session}_metaorders.csv",
            ]
        )
    return pd.DataFrame(
        [
            {
                "path": path.relative_to(panel.DATA.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ]
    )


def build_profiles() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Half-hour volatility shape for each session, from other sessions only.

    `loso` takes the median ratio across the symbol's OTHER four sessions. It
    uses nothing from the scored session, but some donors are later days, so it
    is cross-validated rather than strictly causal.

    `prior` takes the median across that symbol's sessions BEFORE the scored
    one. Strictly causal, and empty for the earliest session of each symbol,
    which is then simply not scored on that model.
    """
    meta = panel.meta()
    ratios = {row.session: halfhour_ratios(panel.bars(row.session))
              for row in meta.itertuples()}
    loso, prior = {}, {}
    for row in meta.itertuples():
        siblings = meta[(meta.symbol == row.symbol) & (meta.session != row.session)]
        donors = [ratios[s] for s in siblings.session if not ratios[s].empty]
        loso[row.session] = (pd.concat(donors, axis=1).median(axis=1)
                             if donors else pd.Series(dtype=float))
        earlier = siblings[siblings.date < row.date]
        donors = [ratios[s] for s in earlier.session if not ratios[s].empty]
        prior[row.session] = (pd.concat(donors, axis=1).median(axis=1)
                              if donors else pd.Series(dtype=float))
    return loso, prior


def pool(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Order-count-weighted decile means across sessions."""
    frame = pd.concat(tables, ignore_index=True)
    out = (frame.assign(wp=lambda d: d.predicted * d.n, wr=lambda d: d.realised * d.n)
           .groupby("decile").agg(n=("n", "sum"), wp=("wp", "sum"), wr=("wr", "sum")))
    out["predicted"] = out.wp / out.n
    out["realised"] = out.wr / out.n
    out["ratio"] = out.realised / out.predicted
    return out[["predicted", "realised", "ratio", "n"]]


def band(values: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% band for the mean, resampling whole symbol-days."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = [float(values[rng.integers(0, len(values), len(values))].mean())
             for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def build(out_dir: Path) -> None:
    """Rebuild the corrected reports from committed offline inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    loso, prior = build_profiles()
    rows, tables = [], {name: [] for name in MODEL_ORDER}
    for key in panel.session_keys():
        scales = panel.scales(key)
        result = evaluate_session(key, panel.bars(key), panel.metaorders(key),
                                  float(scales.session_volume),
                                  float(scales.sigma_daily_20d),
                                  tod_profile_loso=loso[key],
                                  tod_profile_prior=prior[key])
        row = {"session": key, "symbol": scales.symbol,
               "split_second": result.split_second,
               "n_train_orders": result.n_train,
               "n_test_orders": result.n_test,
               "n_crossing_orders_excluded": result.n_crossing_excluded,
               "delta": result.calibration.delta,
               "n_lags": result.calibration.n_lags, **result.params}
        for name in MODEL_ORDER:
            score = result.scores.get(name)
            row[f"{name}_r2"] = score["r2_no_refit"] if score else np.nan
            row[f"{name}_slope"] = score["slope"] if score else np.nan
            if name in result.tables:
                table = result.tables[name].copy()
                table.insert(0, "session", key)
                table.insert(1, "model", name)
                tables[name].append(table)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.concat([t for name in MODEL_ORDER for t in tables[name]],
              ignore_index=True).to_csv(
        out_dir / "calibration_by_decile.csv", index=False)

    print("CONDITIONAL IMPACT ACCURACY, held-out 30% of each session")
    print("R2 is of realised on predicted with NO refit: the model's own "
          "prediction, not a line through it.\n")
    show = ["session", "n_test_orders"] + [f"{m}_r2" for m in MODEL_ORDER]
    print(summary[show].rename(columns=lambda c: c.replace("_r2", ""))
          .to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    pooled = {name: pool(tables[name]) for name in MODEL_ORDER if tables[name]}
    pd.concat([t.assign(model=n) for n, t in pooled.items()]).to_csv(
        out_dir / "calibration_pooled.csv")

    print("\n\nMODEL COMPARISON, median over the 15 symbol-days, with a "
          "bootstrap band by symbol-day")
    print("slope 1.000 is calibrated; top-decile ratio 1.000 is calibrated "
          "where a desk cares.\n")
    lines = []
    for name in MODEL_ORDER:
        if name not in pooled:
            continue
        r2 = summary[f"{name}_r2"].to_numpy(float)
        slope = summary[f"{name}_slope"].to_numpy(float)
        lo, hi = band(r2)
        lines.append({
            "model": name,
            "n_sessions": int(np.isfinite(r2).sum()),
            "median_r2": float(np.nanmedian(r2)),
            "mean_r2_band": f"[{lo:.3f}, {hi:.3f}]",
            "median_slope": float(np.nanmedian(slope)),
            "top_decile_ratio": float(pooled[name].ratio.iloc[-1]),
            "decile_1_to_8_ratio": float(pooled[name].ratio.iloc[:-1].abs().mean()),
        })
    comparison = pd.DataFrame(lines)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    historical_path = (
        panel.DATA.parent
        / "reports"
        / "conditional_impact"
        / "model_comparison.csv"
    )
    historical = pd.read_csv(historical_path)
    paired = historical.merge(
        comparison,
        on="model",
        how="outer",
        suffixes=("_historical", "_corrected"),
        validate="one_to_one",
    )
    for column in ("median_r2", "median_slope", "top_decile_ratio"):
        paired[f"{column}_change"] = (
            paired[f"{column}_corrected"]
            - paired[f"{column}_historical"]
        )
    paired.to_csv(out_dir / "historical_vs_corrected.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    for name in MODEL_ORDER:
        if name in pooled:
            print(f"\nPOOLED CALIBRATION, {name}, by predicted-impact decile")
            print(pooled[name].to_string(float_format=lambda v: f"{v:0.6f}"))

    # ------------------------------------------------------------- rate term
    k = summary.rate_k.to_numpy(float)
    lo, hi = band(k)
    print(f"\n\nPARTICIPATION-RATE TERM  (1 + k log rate)")
    print(f"  k, mean over 15 sessions   {k.mean():+.4f}  band by symbol-day "
          f"[{lo:+.4f}, {hi:+.4f}]")
    print(f"  k negative on              {int((k < 0).sum())} of {len(k)} sessions")
    print(f"  median execution rate      {summary.rate_median_test.median():.4f}")
    print(f"  distinguishable from zero  "
          f"{'YES' if (lo > 0 or hi < 0) else 'NO, the band contains zero'}")
    print(f"  top-decile calibration ratio: sqrt {pooled['sqrt'].ratio.iloc[-1]:.4f}"
          f"   sqrt_rate {pooled['sqrt_rate'].ratio.iloc[-1]:.4f}"
          f"   propagator_scaled {pooled['propagator_scaled'].ratio.iloc[-1]:.4f}")

    print(f"\nblend alpha: mean {summary.blend_alpha.mean():.3f}, "
          f"range {summary.blend_alpha.min():.3f} to {summary.blend_alpha.max():.3f} "
          f"(1 is all daily sigma, 0 is all trailing sigma)")
    input_manifest().to_csv(out_dir / "input_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "report_version": "outcome-end-v3",
                "decile_policy": "12-significant-digit ties, integer midpoint ranks",
                "float_check_rtol": FLOAT_RTOL,
                "float_check_atol": FLOAT_ATOL,
                "train_rule": "t_start < split_second and t_end < split_second",
                "test_rule": "t_start >= split_second",
                "crossing_rule": "excluded from both train and test",
                "train_fraction": panel.TRAIN_FRAC,
                "n_sessions": len(summary),
                "n_crossing_orders_excluded": int(
                    summary["n_crossing_orders_excluded"].sum()
                ),
            }
        ]
    ).to_csv(out_dir / "methodology.csv", index=False)
    print(f"\nsaved -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/conditional_impact_corrected"),
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="rebuild in a temporary directory and compare committed artifacts",
    )
    args = ap.parse_args()
    if not args.check:
        build(args.out_dir)
        return 0

    with tempfile.TemporaryDirectory(prefix="impact-conditional-check-") as raw:
        rebuilt = Path(raw)
        build(rebuilt)
        verify_artifacts(args.out_dir, rebuilt)
    print(f"Verified {len(ARTIFACTS)} corrected conditional-impact artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
