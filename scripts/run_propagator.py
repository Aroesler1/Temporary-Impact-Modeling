#!/usr/bin/env python3
"""Calibrate the transient-impact propagator and report the two R^2 regimes.

Usage:
    python scripts/run_propagator.py --data data/MSFT_2024-06-03_1s.csv
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

from propagator import calibrate, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/MSFT_2024-06-03_1s.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/propagator"))
    args = parser.parse_args()

    frame = load(args.data)
    print(f"{args.data.name}: {len(frame):,} one-second observations\n")

    explanatory = calibrate(frame)
    predictive = calibrate(frame, drop_contemporaneous=True)

    print("EXPLANATORY  (contemporaneous flow included)")
    print(explanatory.grid.head(5).to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
    print(f"\n  best: delta={explanatory.best.delta}  lags={explanatory.best.n_lags}"
          f"  OOS R2={explanatory.best.r2_out:.5f}")
    print(f"  memoryless (L=0, same delta) OOS R2={explanatory.memoryless.r2_out:.5f}")
    print(f"  gain from lagged history: {explanatory.history_gain:+.5f}")

    print("\nPREDICTIVE  (lags >= 1 only)")
    print(predictive.grid.head(5).to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
    print(f"\n  best: delta={predictive.best.delta}  lags={predictive.best.n_lags}"
          f"  OOS R2={predictive.best.r2_out:.5f}")

    ratio = (explanatory.best.r2_out / predictive.best.r2_out
             if predictive.best.r2_out > 0 else float("inf"))
    print(f"\nexplanatory / predictive R2 ratio: {ratio:.0f}x")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    explanatory.grid.to_csv(args.out_dir / "grid_explanatory.csv", index=False)
    predictive.grid.to_csv(args.out_dir / "grid_predictive.csv", index=False)
    pd.DataFrame({"lag": np.arange(explanatory.best.kernel.size),
                  "G": explanatory.best.kernel}).to_csv(
        args.out_dir / "kernel_explanatory.csv", index=False)
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
