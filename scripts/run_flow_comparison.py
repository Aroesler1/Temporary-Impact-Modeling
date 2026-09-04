#!/usr/bin/env python3
"""Order flow imbalance against signed trade volume, alone and together.

Usage:
    python scripts/run_flow_comparison.py
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
from orderflow import FIXED_DELTA, compare_flows, delta_sensitivity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/flow"))
    ap.add_argument("--ofi", default="ofi_integrated",
                    choices=("ofi_best", "ofi_sum", "ofi_integrated"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key in panel.session_keys():
        for variant in ("ofi_best", args.ofi) if args.ofi != "ofi_best" else ("ofi_best",):
            for result in compare_flows(panel.bars_with_ofi(key), key, ofi_col=variant):
                rows.append({"session": key, "symbol": panel.scales(key).symbol,
                             "ofi_variant": variant, "relation": result.relation,
                             "r2_trade": result.r2_trade, "r2_ofi": result.r2_ofi,
                             "r2_both": result.r2_both,
                             "incremental_trade": result.incremental_trade,
                             "incremental_ofi": result.incremental_ofi,
                             "delta_trade": result.delta_trade,
                             "delta_ofi": result.delta_ofi, "n": result.n})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "per_session.csv", index=False)

    sens = pd.concat([delta_sensitivity(panel.bars_with_ofi(k), col).assign(session=k)
                      for k in panel.session_keys()
                      for col in ("signed_vol", "ofi_best")], ignore_index=True)
    sens.to_csv(args.out_dir / "delta_sensitivity.csv", index=False)
    print("WHY delta IS FIXED AT 0.5 AND NOT SELECTED")
    print("mean R2 across the 15 sessions at each concavity, contemporaneous:\n")
    print(sens.groupby(["column", "delta"])[["r2_in", "r2_out"]].mean()
          .to_string(float_format=lambda v: f"{v:0.4f}"))
    print("\nFor OFI the in-sample fit rises with delta all the way to linear "
          "while the held-out\nfit collapses; selecting delta on any inner split "
          "picks the specification that fails.\nBoth flows below use delta = "
          f"{FIXED_DELTA}, fixed in advance.\n\n")

    print("ORDER FLOW versus TRADE FLOW, one-second bins, out of sample on the "
          "same 70/30 split")
    print("incremental_ofi is what OFI adds GIVEN signed trade volume, and "
          "incremental_trade\nis what trade volume adds given OFI. Both are out-"
          "of-sample R2 differences.\n")
    for variant in frame.ofi_variant.unique():
        for relation in ("contemporaneous", "predictive"):
            sub = frame[(frame.ofi_variant == variant) & (frame.relation == relation)]
            print(f"-- {relation}, OFI = {variant} --")
            print(sub[["session", "delta_trade", "delta_ofi", "r2_trade",
                       "r2_ofi", "r2_both", "incremental_trade", "incremental_ofi"]]
                  .to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
            print()

    def band(values: np.ndarray, sessions: np.ndarray, seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        uniq = np.unique(sessions)
        by = {u: np.flatnonzero(sessions == u) for u in uniq}
        draws = [float(values[np.concatenate([by[u] for u in
                 rng.choice(uniq, size=len(uniq), replace=True)])].mean())
                 for _ in range(args.boot)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return f"{values.mean():.5f} [{lo:.5f}, {hi:.5f}]"

    pooled = []
    for variant in frame.ofi_variant.unique():
        for relation in ("contemporaneous", "predictive"):
            sub = frame[(frame.ofi_variant == variant) & (frame.relation == relation)]
            row = {"ofi_variant": variant, "relation": relation}
            for col in ("r2_trade", "r2_ofi", "r2_both", "incremental_trade",
                        "incremental_ofi"):
                row[col] = band(sub[col].to_numpy(float), sub.session.to_numpy())
            row["ofi_beats_trade_on"] = f"{int((sub.r2_ofi > sub.r2_trade).sum())}/15"
            pooled.append(row)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(args.out_dir / "pooled.csv", index=False)
    print("POOLED, mean across the 15 symbol-days with a bootstrap band by "
          "symbol-day\n")
    print(pooled.to_string(index=False))

    by_symbol = (frame.groupby(["ofi_variant", "relation", "symbol"])
                 [["r2_trade", "r2_ofi", "r2_both", "incremental_trade",
                   "incremental_ofi"]].agg(["min", "max"]))
    by_symbol.to_csv(args.out_dir / "by_symbol.csv")
    print("\nPER SYMBOL, range across its five sessions")
    print(by_symbol.to_string(float_format=lambda v: f"{v:0.5f}"))
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
