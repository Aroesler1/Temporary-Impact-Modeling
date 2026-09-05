#!/usr/bin/env python3
"""Restate the two-day propagator and metaorder numbers across all 15 sessions.

The published figures -- explanatory R2 0.366 and 0.432, predictive 0.004 and
0.005, metaorder exponent 0.370 -- are each ONE symbol-day. This recomputes them
on the same code over fifteen and reports per-stock ranges with a bootstrap band
by symbol-day, plus the count of sessions on which the explanatory-versus-
predictive gap actually holds.

Usage:
    python scripts/run_propagator_panel.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import panel  # noqa: E402
from metaorder_impact import fit_impact_law  # noqa: E402
from propagator import calibrate  # noqa: E402


def band(values: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% band for the mean, resampling whole symbol-days."""
    rng = np.random.default_rng(seed)
    draws = [float(values[rng.integers(0, len(values), len(values))].mean())
             for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/panel"))
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key in panel.session_keys():
        scales = panel.scales(key)
        bars = panel.bars(key)
        explanatory = calibrate(bars)
        predictive = calibrate(bars, drop_contemporaneous=True)
        law = fit_impact_law(panel.metaorders(key), float(scales.session_volume),
                             float(scales.sigma_daily_20d), n_bins=args.bins)
        rows.append({
            "session": key, "symbol": scales.symbol, "n_bars": len(bars),
            "explanatory_r2": explanatory.best.r2_out,
            "explanatory_delta": explanatory.best.delta,
            "explanatory_lags": explanatory.best.n_lags,
            "memoryless_r2": (explanatory.memoryless.r2_out
                              if explanatory.memoryless else np.nan),
            "predictive_r2": predictive.best.r2_out,
            "predictive_delta": predictive.best.delta,
            "predictive_lags": predictive.best.n_lags,
            "ratio": (explanatory.best.r2_out / predictive.best.r2_out
                      if predictive.best.r2_out > 0 else np.inf),
            "metaorder_exponent": law.exponent,
            "metaorder_r2": law.r_squared,
            "n_metaorders": law.n_metaorders,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "per_session.csv", index=False)

    print("PROPAGATOR AND METAORDER NUMBERS ACROSS 15 SYMBOL-DAYS")
    print("The published pair (0.36599 / 0.43229 explanatory, 0.00423 / 0.00455 "
          "predictive)\nand the exponent 0.370 were each one session. Here is "
          "the same code on fifteen.\n")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))

    print("\nPER STOCK, range across five sessions, with a bootstrap band for "
          "the mean\n")
    out = []
    for symbol in sorted(frame.symbol.unique()):
        sub = frame[frame.symbol == symbol]
        for col in ("explanatory_r2", "predictive_r2", "metaorder_exponent"):
            lo, hi = band(sub[col].to_numpy(float))
            out.append({"symbol": symbol, "quantity": col,
                        "min": float(sub[col].min()), "max": float(sub[col].max()),
                        "mean": float(sub[col].mean()),
                        "boot_lo": lo, "boot_hi": hi})
    summary = pd.DataFrame(out)
    summary.to_csv(args.out_dir / "by_symbol.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))

    gap = int((frame.explanatory_r2 > 10 * frame.predictive_r2.abs()).sum())
    positive_pred = int((frame.predictive_r2 > 0).sum())
    print(f"\nthe explanatory-versus-predictive gap (explanatory at least 10x the "
          f"absolute\npredictive R2) holds on {gap} of {len(frame)} sessions.")
    print(f"predictive out-of-sample R2 is POSITIVE on {positive_pred} of "
          f"{len(frame)} sessions; on the rest the\nbest lagged model is worse "
          f"than predicting the mean.")
    print(f"history gain over the memoryless model, median "
          f"{float((frame.explanatory_r2 - frame.memoryless_r2).median()):+.5f}, "
          f"max {float((frame.explanatory_r2 - frame.memoryless_r2).max()):+.5f}")
    print(f"\nmetaorder exponent across all 15: "
          f"{frame.metaorder_exponent.min():.3f} to "
          f"{frame.metaorder_exponent.max():.3f}, "
          f"mean {frame.metaorder_exponent.mean():.3f}")
    lo, hi = band(frame.metaorder_exponent.to_numpy(float))
    print(f"bootstrap band for the mean exponent by symbol-day: [{lo:.3f}, {hi:.3f}]")
    print(f"sessions with exponent below 0.5: "
          f"{int((frame.metaorder_exponent < 0.5).sum())} of {len(frame)}")
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
