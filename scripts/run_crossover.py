#!/usr/bin/env python3
"""Crossover between linear and square-root impact, and the published comparison.

Usage:
    python scripts/run_crossover.py
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
from crossover import (bin_metaorders, crossover_in_ticks, fit_published,  # noqa: E402
                       fit_two_regime)

N_BINS = 40


def pooled_frame(keys: list[str]) -> pd.DataFrame:
    """Every reconstructed metaorder, with participation and impact in sigma."""
    rows = []
    for key in keys:
        scales = panel.scales(key)
        orders = panel.metaorders(key)
        orders = orders[(orders.mid_start > 0) & (orders.shares > 0)]
        rows.append(pd.DataFrame({
            "session": key,
            "symbol": scales.symbol,
            "q": orders.shares.to_numpy(float) / float(scales.session_volume),
            "impact_sigma": (orders["sign"].to_numpy(float)
                             * (np.log(orders.mid_end.to_numpy(float))
                                - np.log(orders.mid_start.to_numpy(float)))
                             / float(scales.sigma_daily_20d)),
        }))
    out = pd.concat(rows, ignore_index=True)
    return out[np.isfinite(out.q) & np.isfinite(out.impact_sigma) & (out.q > 0)]


def binned(frame: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    frame = frame.assign(bin=pd.qcut(frame.q, n_bins, labels=False, duplicates="drop"))
    return frame.groupby("bin").agg(q=("q", "mean"),
                                    impact=("impact_sigma", "mean"),
                                    n=("impact_sigma", "size")).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/crossover"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keys = panel.session_keys()
    meta = panel.meta().set_index("session")
    raw = pooled_frame(keys)

    print("CROSSOVER: linear below q*, square-root above, q* by profile likelihood")
    print(f"pooled over {len(keys)} symbol-days, {len(raw):,} reconstructed "
          f"metaorders, {N_BINS} bins\n")

    rows = []
    for label, sub in [("pooled", raw)] + [(s, raw[raw.symbol == s])
                                           for s in sorted(raw.symbol.unique())]:
        table = binned(sub)
        fit = fit_two_regime(table.q.to_numpy(), table.impact.to_numpy(),
                             table.n.to_numpy())
        sel = meta if label == "pooled" else meta[meta.symbol == label]
        ticks = crossover_in_ticks(fit, float(sel.sigma_daily_20d.median()),
                                   float(sel.mid_median.median()))
        rows.append({"panel": label, "n_metaorders": int(len(sub)),
                     "a": fit.a, "r2_weighted": fit.r2_weighted, **ticks})
        if label == "pooled":
            fit.profile.to_csv(args.out_dir / "crossover_profile_pooled.csv",
                               index=False)

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "crossover.csv", index=False)
    show = table[["panel", "n_metaorders", "q_star_fraction_of_daily_volume",
                  "q_star_ci_high", "impact_at_crossover_ticks",
                  "above_one_tick", "q_star_at_grid_boundary", "r2_weighted"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:0.6g}"))

    print("\nq* in shares of a median session, and the impact there:")
    for _, r in table.iterrows():
        sel = meta if r.panel == "pooled" else meta[meta.symbol == r.panel]
        shares = r.q_star_fraction_of_daily_volume * float(sel.session_volume.median())
        print(f"  {r.panel:<8} q* = {r.q_star_fraction_of_daily_volume:.3g} of daily "
              f"volume = {shares:,.0f} shares;  impact there "
              f"{r.impact_at_crossover_ticks:.2f} ticks "
              f"({'above' if r.above_one_tick else 'BELOW'} one tick)")

    # ---------------------------------------------------------------- published
    print("\n\nPUBLISHED RECIPE (arXiv 2606.24019): 30s bins, dominance > 0.3, "
          f"duration >= 60s, size >= 1e-4 of daily volume")
    pub_rows, pub_orders = [], []
    for key in keys:
        scales = panel.scales(key)
        orders = bin_metaorders(panel.bars(key), float(scales.session_volume))
        orders.insert(0, "session", key)
        orders.insert(1, "symbol", scales.symbol)
        orders["sigma_d"] = float(scales.sigma_daily_20d)
        pub_orders.append(orders)
        pub_rows.append({"session": key, "n_metaorders": len(orders),
                         "median_participation": float(orders.participation.median())
                         if len(orders) else np.nan})
    pub = pd.concat(pub_orders, ignore_index=True)
    pd.DataFrame(pub_rows).to_csv(args.out_dir / "published_counts.csv", index=False)
    print(pd.DataFrame(pub_rows).to_string(index=False,
                                           float_format=lambda v: f"{v:0.6g}"))

    fits = []
    for label, sub in [("pooled", pub)] + [(s, pub[pub.symbol == s])
                                           for s in sorted(pub.symbol.unique())]:
        if len(sub) < 50:
            continue
        fit = fit_published(sub, sub.sigma_d, groups=sub.session)
        fits.append({"panel": label, "n": fit.n_metaorders,
                     "delta_free": fit.delta,
                     "delta_ci_low": fit.delta_ci[0], "delta_ci_high": fit.delta_ci[1],
                     "c_free": fit.c_free,
                     "c_delta_half": fit.c_half,
                     "c_ci_low": fit.c_half_ci[0], "c_ci_high": fit.c_half_ci[1]})
    fit_table = pd.DataFrame(fits)
    fit_table.to_csv(args.out_dir / "published_fit.csv", index=False)

    # the same crossover question asked of the published recipe's metaorders,
    # which sit two orders of magnitude higher in participation than the
    # fill-run reconstruction and so probe a different part of the curve
    pub_bins = binned(pub.assign(q=pub.participation,
                                 impact_sigma=pub.impact / pub.sigma_d), n_bins=12)
    pub_fit = fit_two_regime(pub_bins.q.to_numpy(), pub_bins.impact.to_numpy(),
                             pub_bins.n.to_numpy())
    pub_ticks = crossover_in_ticks(pub_fit, float(meta.sigma_daily_20d.median()),
                                   float(meta.mid_median.median()))
    print(f"\ncrossover on the published-recipe metaorders (12 bins, "
          f"{len(pub):,} orders): q* = "
          f"{pub_ticks['q_star_fraction_of_daily_volume']:.3g} of daily volume, "
          f"impact there {pub_ticks['impact_at_crossover_ticks']:.2f} ticks, "
          f"{'at the grid boundary' if pub_ticks['q_star_at_grid_boundary'] else 'interior'}")
    pd.DataFrame([{"panel": "published_recipe", "n_metaorders": len(pub),
                   "a": pub_fit.a, "r2_weighted": pub_fit.r2_weighted,
                   **pub_ticks}]).to_csv(
        args.out_dir / "crossover_published_recipe.csv", index=False)
    print("\nI/sigma_D = c (Q/V_D)^delta")
    print(fit_table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    print("\npublished AAPL, 178 days: c_raw 0.69 [0.63, 0.77], c_eff 0.34 "
          "(bias-corrected), delta 0.50 [0.32, 0.66]")
    aapl = fit_table[fit_table.panel == "AAPL"]
    if len(aapl):
        r = aapl.iloc[0]
        print(f"ours AAPL,   5 days: c_raw {r.c_delta_half:.2f} "
              f"[{r.c_ci_low:.2f}, {r.c_ci_high:.2f}], delta {r.delta_free:.2f} "
              f"[{r.delta_ci_low:.2f}, {r.delta_ci_high:.2f}]")
    print("c_eff is not recomputed: the paper's abstract states the "
          "bias-corrected prefactor but not the correction.")
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
