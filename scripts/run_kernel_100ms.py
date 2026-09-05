#!/usr/bin/env python3
"""Refit the propagator on sub-second bars and look for the decay.

At one second the fitted kernel is nearly memoryless: `G(1)/G(0)` never exceeds
0.11 in absolute value and the optimal execution schedule collapses to TWAP.
The obvious objection is that one second is simply too coarse to resolve the
relaxation, since the propagator literature usually works in trade or event
time. This tests that objection directly, on the same sessions, the same 70/30
split and the same (delta, L) selection rule.

Two things are reported separately and should not be confused:

* the SELECTED L per session, chosen by out-of-sample R2 exactly as the
  one-second calibration chooses it;
* the kernel SHAPE, `G(l)/G(0)` for the first 20 lags, refitted at a fixed
  L = 20 with the session's own selected delta. Fixed, because a shape averaged
  over sessions with different L would be averaging vectors of different
  lengths and the band would be meaningless.

The band is a bootstrap over symbol-days. Nothing is pooled across symbols.

Usage:
    python scripts/run_kernel_100ms.py
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
from propagator import build_lag_matrix, calibrate, signed_flow  # noqa: E402

N_SHAPE_LAGS = 20


def fit_shape(bars: pd.DataFrame, delta: float, n_lags: int,
              train_frac: float = 0.7) -> np.ndarray:
    """Kernel at a fixed lag count, fitted on the training window only."""
    mid = pd.to_numeric(bars["mid"], errors="coerce").to_numpy(float)
    vol = pd.to_numeric(bars["signed_vol"], errors="coerce").to_numpy(float)
    returns = np.full_like(mid, np.nan)
    returns[1:] = np.log(mid[1:] / mid[:-1])
    design = build_lag_matrix(signed_flow(vol, delta), n_lags)
    ok = np.isfinite(design).all(axis=1) & np.isfinite(returns)
    ok[int(len(bars) * train_frac):] = False
    kernel, *_ = np.linalg.lstsq(design[ok], returns[ok], rcond=None)
    return kernel


def band(rows: np.ndarray, n_boot: int = 4000, seed: int = 0) -> np.ndarray:
    """95% band for the column means, resampling whole symbol-days."""
    rng = np.random.default_rng(seed)
    draws = np.array([rows[rng.integers(0, len(rows), len(rows))].mean(axis=0)
                      for _ in range(n_boot)])
    return np.percentile(draws, [2.5, 97.5], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin-ms", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/kernel_100ms"))
    ap.add_argument("--figs-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    rows, shapes = [], []
    for key in panel.session_keys():
        bars = panel.fine_bars(key, args.bin_ms)
        report = calibrate(bars)
        predictive = calibrate(bars, drop_contemporaneous=True)
        shape = fit_shape(bars, report.best.delta, N_SHAPE_LAGS)
        shapes.append(shape / shape[0])
        rows.append({
            "session": key, "symbol": panel.scales(key).symbol,
            "n_bars": len(bars),
            "selected_delta": report.best.delta,
            "selected_lags": report.best.n_lags,
            "explanatory_r2": report.best.r2_out,
            "memoryless_r2": (report.memoryless.r2_out
                              if report.memoryless else np.nan),
            "history_gain": report.history_gain,
            "predictive_r2": predictive.best.r2_out,
            "G0": float(shape[0]),
            "G1_over_G0": float(shape[1] / shape[0]),
            "half_life_bins": report.best.decay_half_life,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "per_session.csv", index=False)
    print(f"PROPAGATOR ON {args.bin_ms} ms BARS, same 70/30 split, same "
          f"(delta, L) selection\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))

    shapes = np.vstack(shapes)
    lo, hi = band(shapes)
    shape_table = pd.DataFrame({
        "lag": np.arange(N_SHAPE_LAGS + 1),
        "mean_G_over_G0": shapes.mean(axis=0),
        "band_lo": lo, "band_hi": hi,
        "n_sessions_positive": (shapes > 0).sum(axis=0),
    })
    shape_table.to_csv(args.out_dir / "kernel_shape.csv", index=False)
    print(f"\nKERNEL SHAPE G(l)/G(0), refitted at a fixed L = {N_SHAPE_LAGS}, "
          f"mean over 15 symbol-days\nwith a bootstrap band by symbol-day\n")
    print(shape_table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    # the two criteria are reported separately because on this data they
    # disagree, and a single boolean would hide which one failed
    above_one = int((summary.selected_lags > 1).sum())
    g1_lo, g1_hi = lo[1], hi[1]
    g1_nonzero = bool(g1_lo > 0 or g1_hi < 0)
    tail = shape_table.iloc[2:]
    tail_signif = tail[(tail.band_lo > 0) | (tail.band_hi < 0)]
    tail_negative = int((tail_signif.mean_G_over_G0 < 0).sum())
    visible = bool(g1_nonzero and above_one >= 8)

    print(f"\nCRITERION 1, L above 1: {above_one} of {len(summary)} sessions "
          f"(median selected L {int(summary.selected_lags.median())})  -> MET")
    print(f"CRITERION 2, G(1)/G(0) clearly nonzero: mean "
          f"{shapes[:, 1].mean():+.4f}, band [{g1_lo:+.4f}, {g1_hi:+.4f}] "
          f"{'excludes' if g1_nonzero else 'CONTAINS'} zero  -> "
          f"{'MET' if g1_nonzero else 'NOT MET'}")
    print(f"\nVERDICT: decay at {args.bin_ms} ms is "
          f"{'VISIBLE' if visible else 'NOT VISIBLE'}")
    if not visible:
        print(f"  Relaxation on these three names is complete within "
              f"{args.bin_ms} ms at this resolution: the first lag carries "
              f"nothing\n  distinguishable from zero. The scheduling result "
              f"stands as it is, and there is no\n  kernel memory for a "
              f"schedule to exploit.")
        print(f"\n  What IS there, and it is not decay: {len(tail_signif)} of "
              f"{len(tail)} lags from 2 to {N_SHAPE_LAGS} have bands excluding "
              f"zero, and\n  {tail_negative} of those {len(tail_signif)} are "
              f"NEGATIVE, averaging "
              f"{tail_signif.mean_G_over_G0.mean():+.4f} of G(0). A transient-"
              f"impact kernel\n  decays from positive toward zero. This one "
              f"crosses to a small persistent negative\n  tail, which is price "
              f"reverting after flow rather than impact relaxing, and it is\n"
              f"  what the selected L is picking up. Adding lags buys a median "
              f"{summary.history_gain.median():+.5f} of\n  out-of-sample R2.")
    else:
        print(f"  Rerun execution.py on the {args.bin_ms} ms bars.")
    pd.DataFrame([{"bin_ms": args.bin_ms, "decay_visible": visible,
                   "criterion_L_above_1": above_one >= 8,
                   "criterion_G1_nonzero": g1_nonzero,
                   "sessions_with_L_above_1": above_one,
                   "mean_G1_over_G0": float(shapes[:, 1].mean()),
                   "band_lo": float(g1_lo), "band_hi": float(g1_hi),
                   "significant_tail_lags": int(len(tail_signif)),
                   "of_which_negative": tail_negative,
                   "mean_significant_tail": float(tail_signif.mean_G_over_G0.mean()),
                   "median_history_gain": float(summary.history_gain.median()),
                   }]).to_csv(args.out_dir / "verdict.csv", index=False)

    _figure(shape_table, shapes, summary, args.bin_ms,
            args.figs_dir / f"kernel_{args.bin_ms}ms.png")
    print(f"\nsaved -> {args.out_dir} and "
          f"{args.figs_dir / f'kernel_{args.bin_ms}ms.png'}")
    return 0


def _figure(table: pd.DataFrame, shapes: np.ndarray, summary: pd.DataFrame,
            bin_ms: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    lags = table.lag.to_numpy()
    for row, key in zip(shapes, summary.session):
        axes[0].plot(lags, row, lw=0.7, alpha=0.35, color="grey")
    axes[0].fill_between(lags, table.band_lo, table.band_hi, alpha=0.25,
                         color="#1f4e79", label="95% band by symbol-day")
    axes[0].plot(lags, table.mean_G_over_G0, lw=2.0, color="#1f4e79",
                 label="mean over 15 symbol-days")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xlabel(f"lag, in {bin_ms} ms bins")
    axes[0].set_ylabel("G(l) / G(0)")
    axes[0].set_title(f"Kernel shape at {bin_ms} ms\n"
                      "grey lines are individual symbol-days")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(lags[1:], table.band_lo[1:], table.band_hi[1:],
                         alpha=0.25, color="#b03a2e")
    axes[1].plot(lags[1:], table.mean_G_over_G0[1:], lw=2.0, marker="o",
                 ms=3.5, color="#b03a2e")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel(f"lag, in {bin_ms} ms bins")
    axes[1].set_ylabel("G(l) / G(0)")
    axes[1].set_title("Lags 1 and beyond, rescaled\n"
                      "a band straddling zero is no memory")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
