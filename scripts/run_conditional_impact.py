#!/usr/bin/env python3
"""Conditional impact accuracy on held-out metaorders, all fifteen sessions.

This is the headline number of the propagator section: given an order's size
and the seconds it executed over, how close was the predicted impact to the
realised one, on orders the model never saw.

Usage:
    python scripts/run_conditional_impact.py
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
from conditional_impact import evaluate_session  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/conditional_impact"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, prop_tables, scaled_tables, sqrt_tables, local_tables = [], [], [], [], []
    for key in panel.session_keys():
        scales = panel.scales(key)
        result = evaluate_session(key, panel.bars(key), panel.metaorders(key),
                                  float(scales.session_volume),
                                  float(scales.sigma_daily_20d))
        rows.append({
            "session": key,
            "n_test_orders": result.propagator["n"],
            "delta": result.calibration.delta,
            "n_lags": result.calibration.n_lags,
            "prop_r2": result.propagator["r2_no_refit"],
            "prop_slope": result.propagator["slope"],
            "prop_scale": result.propagator_scale,
            "scaled_r2": result.propagator_scaled["r2_no_refit"],
            "scaled_slope": result.propagator_scaled["slope"],
            "sqrt_c": result.sqrt_c,
            "sqrt_r2": result.sqrt_model["r2_no_refit"],
            "sqrt_r2_refit": result.sqrt_model["r2_refit"],
            "sqrt_slope": result.sqrt_model["slope"],
            "local_c": result.sqrt_local_c,
            "local_r2": result.sqrt_local["r2_no_refit"],
            "local_slope": result.sqrt_local["slope"],
        })
        for table, store, name in (
                (result.propagator_table, prop_tables, "propagator"),
                (result.propagator_scaled_table, scaled_tables, "propagator_scaled"),
                (result.sqrt_table, sqrt_tables, "sqrt"),
                (result.sqrt_local_table, local_tables, "sqrt_trailing_sigma")):
            table = table.copy()
            table.insert(0, "session", key)
            table.insert(1, "model", name)
            store.append(table)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    prop = pd.concat(prop_tables, ignore_index=True)
    scaled = pd.concat(scaled_tables, ignore_index=True)
    sqrt = pd.concat(sqrt_tables, ignore_index=True)
    local = pd.concat(local_tables, ignore_index=True)
    pd.concat([prop, scaled, sqrt, local], ignore_index=True).to_csv(
        args.out_dir / "calibration_by_decile.csv", index=False)

    def pool(table: pd.DataFrame) -> pd.DataFrame:
        out = (table.assign(wp=lambda d: d.predicted * d.n, wr=lambda d: d.realised * d.n)
               .groupby("decile").agg(n=("n", "sum"), wp=("wp", "sum"), wr=("wr", "sum")))
        out["predicted"] = out.wp / out.n
        out["realised"] = out.wr / out.n
        out["ratio"] = out.realised / out.predicted
        return out[["predicted", "realised", "ratio", "n"]]

    print("CONDITIONAL IMPACT ACCURACY, held-out 30% of each session")
    print("R2 is of realised on predicted with NO refit: the model's own "
          "prediction, not a line through it.\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    pooled = {name: pool(t) for name, t in
              (("propagator", prop), ("propagator_scaled", scaled),
               ("sqrt", sqrt), ("sqrt_trailing_sigma", local))}
    for name, table in pooled.items():
        print(f"\nPOOLED CALIBRATION, {name}, by predicted-impact decile")
        print(table.to_string(float_format=lambda v: f"{v:0.6f}"))
    pd.concat([t.assign(model=n) for n, t in pooled.items()]).to_csv(
        args.out_dir / "calibration_pooled.csv")

    print(f"\nmedian R2, no refit:  propagator {summary.prop_r2.median():.4f}   "
          f"propagator rescaled {summary.scaled_r2.median():.4f}   "
          f"square-root {summary.sqrt_r2.median():.4f}   "
          f"square-root with trailing sigma {summary.local_r2.median():.4f}")
    print(f"median slope of realised on predicted:  square-root "
          f"{summary.sqrt_slope.median():.3f}   with trailing sigma "
          f"{summary.local_slope.median():.3f}   (1.000 is calibrated)")
    print(f"rescaled propagator beats square-root on "
          f"{int((summary.scaled_r2 > summary.sqrt_r2).sum())} of {len(summary)} sessions")
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
