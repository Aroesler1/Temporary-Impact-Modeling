#!/usr/bin/env python3
"""Replay the propagator-optimal schedule, TWAP and Almgren-Chriss out of sample.

Usage:
    python scripts/run_schedule_oos.py
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
from execution import (bootstrap_saving, fit_linear_kernel, kernel_matrix,  # noqa: E402
                       optimal_schedule, replay_session, select_lags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/schedule"))
    ap.add_argument("--horizon", type=int, default=600, help="seconds")
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--kappa", type=float, default=0.005,
                    help="Almgren-Chriss urgency; 0 is TWAP")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results, diagnostics = [], []
    for key in panel.session_keys():
        bars = panel.bars(key)
        scales = panel.scales(key)
        results = replay_session(key, bars, float(scales.session_volume),
                                 horizon=args.horizon, n_starts=args.starts,
                                 kappa=args.kappa)
        all_results.extend(results)

        mid = bars.mid.to_numpy(float)
        ret = np.full(len(mid), np.nan)
        ret[1:] = np.log(mid[1:] / mid[:-1])
        vol = bars.signed_vol.to_numpy(float)
        train_end = int(len(bars) * panel.TRAIN_FRAC)
        n_lags = select_lags(ret, vol, train_end)
        kernel = fit_linear_kernel(ret, vol, n_lags, train_end)
        km = kernel_matrix(kernel, args.horizon)
        x = optimal_schedule(km, 1.0)
        diagnostics.append({
            "session": key, "n_lags": n_lags, "G0": float(kernel[0]),
            "G1_over_G0": float(kernel[1] / kernel[0]) if len(kernel) > 1 else np.nan,
            "min_eigenvalue": km.min_eigenvalue, "psd_projected": km.projected,
            "first_second_share": float(x[0]), "last_second_share": float(x[-1]),
            "middle_rate_share": float(np.median(x[1:-1])),
            "min_share": float(x.min()),
            "any_negative": bool((x < 0).any()),
        })
        print(f"  {key:<16} L={n_lags:<3} min eig {km.min_eigenvalue:+.3e}"
              f"{'  PSD-PROJECTED' if km.projected else ''}")

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(args.out_dir / "kernel_diagnostics.csv", index=False)
    print("\nKERNEL DIAGNOSTICS (delta fixed at 1, as the GSS solution requires)")
    print(diag.to_string(index=False, float_format=lambda v: f"{v:0.5g}"))
    print(f"\nindefinite kernel matrices: {int(diag.psd_projected.sum())} of "
          f"{len(diag)} sessions. An indefinite M means the fitted kernel admits "
          "a round trip with negative expected cost, so it is projected onto the "
          "PSD cone before inversion.")

    rows = [{"session": r.session, "start_second": r.start_second,
             "order_fraction": r.order_fraction,
             **{f"{n}_{k}": v for n, c in r.costs.items() for k, v in c.items()},
             **{f"{n}_inventory_var": v for n, v in r.inventory_variance.items()}}
            for r in all_results]
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "replay.csv", index=False)

    print(f"\nREPLAY over the held-out 30%, horizon {args.horizon}s, "
          f"{args.starts} start times a session, {len(frame):,} replays")
    print("cost per share in dollars, and the saving versus TWAP with a "
          "bootstrap band over symbol-days\n")

    out = []
    for fraction in sorted(frame.order_fraction.unique()):
        subset = [r for r in all_results if r.order_fraction == fraction]
        sub = frame[frame.order_fraction == fraction]
        for component in ("total", "impact", "drift"):
            row = {"order_fraction": fraction, "component": component,
                   "median_shares": fraction * float(panel.meta().session_volume.median()),
                   "twap_cost": float(sub[f"TWAP_{component}"].mean())}
            for name in ("propagator_optimal", "almgren_chriss"):
                mean, lo, hi = bootstrap_saving(subset, name, component=component)
                row[f"{name}_cost"] = float(sub[f"{name}_{component}"].mean())
                row[f"{name}_saving"] = mean
                row[f"{name}_lo"] = lo
                row[f"{name}_hi"] = hi
            out.append(row)
    table = pd.DataFrame(out)
    table.to_csv(args.out_dir / "saving_vs_twap.csv", index=False)
    for component in ("impact", "drift", "total"):
        print(f"\n-- {component} cost per share, dollars --")
        print(table[table.component == component].drop(columns="component")
              .to_string(index=False, float_format=lambda v: f"{v:0.4g}"))

    print("\nsaving versus TWAP as a fraction of TWAP's IMPACT cost "
          "(the part a schedule controls):")
    impact = table[table.component == "impact"].set_index("order_fraction")
    for fraction, r in impact.iterrows():
        for name in ("propagator_optimal", "almgren_chriss"):
            print(f"  {fraction:>6.3%}  {name:<20} "
                  f"{r[f'{name}_saving'] / r.twap_cost:+8.3%}  "
                  f"[{r[f'{name}_lo'] / r.twap_cost:+.3%}, "
                  f"{r[f'{name}_hi'] / r.twap_cost:+.3%}]")

    inv = frame.groupby("order_fraction")[[c for c in frame.columns
                                           if c.endswith("inventory_var")]].mean()
    print("\nmean remaining-inventory variance (share^2 seconds), the risk AC buys:")
    print((inv.T / inv["TWAP_inventory_var"]).T.to_string(
        float_format=lambda v: f"{v:0.4f}"))
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
