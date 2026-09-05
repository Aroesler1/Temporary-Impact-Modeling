#!/usr/bin/env python3
"""Fit the square-root law per stock, then look across the cross-section.

Per stock: I/sigma_D = c (Q/V_D)^delta with delta free and with delta fixed at
1/2, bootstrapped BY DAY; the linear-to-square-root crossover; and one extra
column using this repository's time-of-day volatility profile in place of the
daily constant.

Across stocks: the distribution of delta and c, and regressions of each on
relative tick size, a spread proxy, dollar volume and volatility, with
heteroskedasticity-robust standard errors.

The hypothesis, stated before the table: delta FALLS with relative tick size,
because one tick floors the impact of small orders and flattens the fitted
slope. That is the tick-floor explanation the three-name study supported but
could not test, since three names give no cross-sectional variation in tick
size to test it against.

Usage:
    python scripts/run_cross_section.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cross_section as cs  # noqa: E402
from crossover import crossover_in_ticks, fit_published, fit_two_regime  # noqa: E402

MIN_METAORDERS = 200        # below this a per-stock fit is noise
N_BINS = 12
COMPARISON = ("AAPL", "MSFT", "INTC")
# the three-name study's published-recipe numbers, for the comparison rows
TIM3 = {"AAPL": (1.6783, 0.3375), "MSFT": (2.5762, 0.6196), "INTC": (1.5212, 0.3628)}


def load_symbol(symbol: str, meta_dir: Path):
    orders = meta_dir / f"{symbol}.csv"
    daily = meta_dir / f"{symbol}_daily.csv"
    if not orders.exists() or not daily.exists():
        return None, None
    return pd.read_csv(orders), pd.read_csv(daily)


def profile_sigma(orders: pd.DataFrame, daily: pd.DataFrame) -> np.ndarray:
    """sigma_D scaled by the symbol's own median half-hour volatility shape.

    Labelled as this repository's variant throughout: the published recipe uses
    a daily constant and the comparison with it stays like for like without
    this column.
    """
    columns = [c for c in daily.columns if c.startswith("hh_")]
    profile = daily[columns].median().to_numpy(float)
    bucket = ((orders.bin_start.to_numpy(float) - cs.RTH_OPEN_SEC)
              // cs.HALF_HOUR).astype(int)
    factor = np.where((bucket >= 0) & (bucket < len(profile)),
                      profile[np.clip(bucket, 0, len(profile) - 1)], 1.0)
    factor = np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)
    return orders.sigma_d.to_numpy(float) * factor


def fit_one(symbol: str, orders: pd.DataFrame, daily: pd.DataFrame,
            n_boot: int) -> dict:
    fit = fit_published(orders, orders.sigma_d, n_boot=n_boot, groups=orders.date)
    tod = fit_published(orders, profile_sigma(orders, daily), n_boot=n_boot,
                        groups=orders.date)

    frame = pd.DataFrame({"q": orders.participation.to_numpy(float),
                          "impact": orders.impact.to_numpy(float)
                          / orders.sigma_d.to_numpy(float)})
    frame = frame[np.isfinite(frame.q) & np.isfinite(frame.impact) & (frame.q > 0)]
    binned = frame.assign(bin=pd.qcut(frame.q, N_BINS, labels=False,
                                      duplicates="drop")).groupby("bin").agg(
        q=("q", "mean"), impact=("impact", "mean"), n=("impact", "size")
        ).reset_index(drop=True)
    cross = fit_two_regime(binned.q.to_numpy(), binned.impact.to_numpy(),
                           binned.n.to_numpy())
    ticks = crossover_in_ticks(cross, float(daily.sigma_5min.median()),
                               float(daily.close.median()))
    return {
        "symbol": symbol,
        "n_metaorders": fit.n_metaorders,
        "n_sessions": int(daily.shape[0]),
        "delta": fit.delta, "delta_lo": fit.delta_ci[0], "delta_hi": fit.delta_ci[1],
        "c_free": fit.c_free,
        "c_half": fit.c_half, "c_lo": fit.c_half_ci[0], "c_hi": fit.c_half_ci[1],
        "delta_tod": tod.delta, "delta_tod_lo": tod.delta_ci[0],
        "delta_tod_hi": tod.delta_ci[1], "c_half_tod": tod.c_half,
        "brackets_half": bool(fit.delta_ci[0] <= 0.5 <= fit.delta_ci[1]),
        "brackets_half_tod": bool(tod.delta_ci[0] <= 0.5 <= tod.delta_ci[1]),
        "q_star": ticks["q_star_fraction_of_daily_volume"],
        "q_star_interior": not ticks["q_star_at_grid_boundary"],
        "impact_at_crossover_ticks": ticks["impact_at_crossover_ticks"],
        "median_participation": float(orders.participation.median()),
        "mean_sigma": float(daily.sigma_5min.mean()),
        "median_close": float(daily.close.median()),
        "mean_volume": float(daily.volume_rth.mean()),
        "unsided_share": float(daily.unsided_share.mean()),
    }, binned.assign(symbol=symbol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path,
                    default=Path("data/cross_section/metaorders"))
    ap.add_argument("--sample", type=Path,
                    default=Path("data/cross_section/sample.csv"))
    ap.add_argument("--june", type=Path,
                    default=Path("data/cross_section/june_stats.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/cross_section"))
    ap.add_argument("--figs-dir", type=Path, default=Path("figs"))
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(args.sample)
    june = pd.read_csv(args.june).set_index("symbol")

    rows, bins, skipped = [], [], []
    for i, symbol in enumerate(sample.symbol, 1):
        orders, daily = load_symbol(symbol, args.meta_dir)
        if orders is None or len(orders) < MIN_METAORDERS:
            skipped.append((symbol, 0 if orders is None else len(orders)))
            continue
        row, binned = fit_one(symbol, orders, daily, args.boot)
        rows.append(row)
        bins.append(binned)
        print(f"  [{i:>3}/{len(sample)}] {symbol:<6} "
              f"{row['n_metaorders']:>6,} metaorders  delta {row['delta']:.3f}  "
              f"c {row['c_half']:.3f}", flush=True)

    fits = pd.DataFrame(rows).merge(sample[["symbol", "role"]], on="symbol")
    fits = fits.merge(june[["relative_tick", "dollar_volume",
                            "mean_high_low_bp"]], left_on="symbol",
                      right_index=True, how="left")
    # The trades schema carries no quotes, so there is no bid-ask spread to
    # measure. This is the June mean daily HIGH-LOW RANGE expressed in ticks,
    # which is a range proxy and not a spread; it is named for what it is.
    fits["daily_range_ticks"] = (fits.mean_high_low_bp / 1e4
                                 * fits.median_close / 0.01)
    fits.to_csv(args.out_dir / "per_stock_fits.csv", index=False)
    pd.concat(bins, ignore_index=True).to_csv(
        args.out_dir / "binned_metaorders.csv", index=False)

    strat = fits[fits.role == "stratified"]
    print(f"CROSS-SECTION: {len(strat)} stratified names fitted, "
          f"{len(fits) - len(strat)} comparison names, {len(skipped)} skipped "
          f"for fewer than {MIN_METAORDERS} metaorders")
    if skipped:
        print("  skipped: " + ", ".join(f"{s}({n})" for s, n in skipped))

    print(f"\nDISTRIBUTION OF delta ACROSS {len(strat)} NAMES "
          f"(delta free, bootstrap by day)")
    q = strat.delta.quantile([0.25, 0.5, 0.75])
    print(f"  median {q[0.5]:.3f}   quartiles [{q[0.25]:.3f}, {q[0.75]:.3f}]   "
          f"range [{strat.delta.min():.3f}, {strat.delta.max():.3f}]")
    print(f"  band brackets 0.5 on {int(strat.brackets_half.sum())} of "
          f"{len(strat)} names ({strat.brackets_half.mean():.0%})")
    print(f"  delta below 0.5 on {int((strat.delta < 0.5).sum())} of {len(strat)}")
    qt = strat.delta_tod.quantile([0.25, 0.5, 0.75])
    print(f"  with the time-of-day sigma (this repo's variant): median "
          f"{qt[0.5]:.3f}, quartiles [{qt[0.25]:.3f}, {qt[0.75]:.3f}], "
          f"brackets 0.5 on {int(strat.brackets_half_tod.sum())}")
    changed = int((strat.brackets_half != strat.brackets_half_tod).sum())
    print(f"  the time-of-day sigma changes the 0.5 verdict on {changed} of "
          f"{len(strat)} names")

    print(f"\nDISTRIBUTION OF c (delta fixed at 1/2)")
    qc = strat.c_half.quantile([0.25, 0.5, 0.75])
    print(f"  median {qc[0.5]:.3f}   quartiles [{qc[0.25]:.3f}, {qc[0.75]:.3f}]   "
          f"range [{strat.c_half.min():.3f}, {strat.c_half.max():.3f}]")

    print(f"\nCROSSOVER, per stock")
    print(f"  q* interior on {int(strat.q_star_interior.sum())} of {len(strat)} "
          f"names; median impact at q* "
          f"{strat.impact_at_crossover_ticks.median():.2f} ticks")
    print(f"  above one tick on "
          f"{int((strat.impact_at_crossover_ticks > 1).sum())} of {len(strat)}")

    # ------------------------------------------------------------ regressions
    raw_design = pd.DataFrame({
        "log_relative_tick": np.log(strat.relative_tick.to_numpy(float)),
        "log_daily_range_ticks": np.log(strat.daily_range_ticks.to_numpy(float)),
        "log_dollar_volume": np.log(strat.dollar_volume.to_numpy(float)),
        "volatility": strat.mean_sigma.to_numpy(float),
    })
    # standardised regressors: every coefficient then reads as the change in the
    # dependent per ONE STANDARD DEVIATION of that regressor, which is the only
    # way four variables on scales from 1e-2 to 1e1 can be compared by eye
    design = (raw_design - raw_design.mean()) / raw_design.std(ddof=0)
    raw_design.assign(symbol=strat.symbol.to_numpy()).to_csv(
        args.out_dir / "regression_inputs.csv", index=False)
    print("\nREGRESSIONS, heteroskedasticity-robust (HC1) standard errors")
    print("regressors are standardised, so each coefficient is the change in "
          "the dependent\nper one standard deviation of that regressor. There "
          "is no spread here: the trades\nschema carries no quotes, so the "
          "third row is the June mean daily HIGH-LOW RANGE\nin ticks, which is "
          "a range proxy and not a spread.\n")
    print("HYPOTHESIS STATED IN ADVANCE: the coefficient on log relative tick "
          "size is NEGATIVE,\nbecause one tick floors the impact of small "
          "orders and flattens the fitted slope.\n")
    tables = []
    for label, target in (("delta", strat.delta.to_numpy(float)),
                          ("c (delta fixed at 1/2)", strat.c_half.to_numpy(float))):
        result = cs.ols_robust(target, design)
        table = result.table()
        table.insert(0, "dependent", label)
        tables.append(table)
        print(f"-- {label} --")
        print(table.drop(columns="dependent").to_string(
            index=False, float_format=lambda v: f"{v:0.4f}"))
        print(f"   R2 {result.r_squared:.4f}, n {result.n}\n")
    pd.concat(tables, ignore_index=True).to_csv(
        args.out_dir / "regressions.csv", index=False)

    # --------------------------------------------------------------- pooled
    # the pooled bootstrap refits a nonlinear model on about a million points
    # once per draw, so it gets a smaller budget than the per-stock fits; the
    # pooled interval is narrow enough that the extra draws buy nothing
    pooled_boot = min(args.boot, 200)
    print(f"POOLED FITS (every metaorder from every name in one fit, "
          f"{pooled_boot} bootstrap draws)", flush=True)
    pooled_rows = []
    for label, names in (("all names", strat.symbol.tolist()),
                         ("small-tick tercile only",
                          strat.nsmallest(len(strat) // 3, "relative_tick"
                                          ).symbol.tolist())):
        frames = []
        for symbol in names:
            orders, _ = load_symbol(symbol, args.meta_dir)
            if orders is not None:
                frames.append(orders)
        pooled = pd.concat(frames, ignore_index=True)
        fit = fit_published(pooled, pooled.sigma_d, n_boot=pooled_boot,
                            groups=pooled.symbol + "_" + pooled.date)
        pooled_rows.append({"panel": label, "n_names": len(names),
                            "n_metaorders": fit.n_metaorders,
                            "delta": fit.delta, "delta_lo": fit.delta_ci[0],
                            "delta_hi": fit.delta_ci[1],
                            "c_half": fit.c_half, "c_lo": fit.c_half_ci[0],
                            "c_hi": fit.c_half_ci[1]})
        print(f"  {label:<26} n={fit.n_metaorders:>7,}  delta {fit.delta:.3f} "
              f"[{fit.delta_ci[0]:.3f}, {fit.delta_ci[1]:.3f}]   "
              f"c {fit.c_half:.3f} [{fit.c_half_ci[0]:.3f}, {fit.c_half_ci[1]:.3f}]")
    pd.DataFrame(pooled_rows).to_csv(args.out_dir / "pooled_fits.csv", index=False)

    # ---------------------------------------------------------- comparisons
    print("\nCOMPARISON WITH THE THREE-NAME STUDY (TIM-3, 5 sessions each, "
          "MBO-derived)")
    print(f"{'symbol':<7}{'here: c':>10}{'band':>18}{'here: delta':>13}"
          f"{'band':>18}{'TIM-3 c':>10}{'TIM-3 delta':>13}")
    for symbol in COMPARISON:
        row = fits[fits.symbol == symbol]
        if row.empty:
            continue
        r = row.iloc[0]
        c3, d3 = TIM3[symbol]
        print(f"{symbol:<7}{r.c_half:>10.3f}"
              f"{f'[{r.c_lo:.2f}, {r.c_hi:.2f}]':>18}{r.delta:>13.3f}"
              f"{f'[{r.delta_lo:.2f}, {r.delta_hi:.2f}]':>18}"
              f"{c3:>10.3f}{d3:>13.3f}")
    aapl = fits[fits.symbol == "AAPL"]
    if not aapl.empty:
        r = aapl.iloc[0]
        print(f"\nAAPL beside arXiv 2606.24019 (178 days, single venue):")
        print(f"  published   c_raw 0.690 [0.63, 0.77]      delta 0.500 "
              f"[0.32, 0.66]")
        print(f"  here        c_raw {r.c_half:.3f} [{r.c_lo:.2f}, {r.c_hi:.2f}]"
              f"      delta {r.delta:.3f} [{r.delta_lo:.2f}, {r.delta_hi:.2f}]"
              f"   ({int(r.n_metaorders):,} metaorders, "
              f"{int(r.n_sessions)} sessions)")

    _figure(strat, args.figs_dir / "cross_section_delta.png")
    print(f"\nsaved -> {args.out_dir} and "
          f"{args.figs_dir / 'cross_section_delta.png'}")
    return 0


def _figure(strat: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(strat.delta, bins=22, color="#1f4e79", alpha=0.85)
    axes[0].axvline(0.5, color="crimson", ls="--", lw=1.4, label="0.5")
    axes[0].axvline(strat.delta.median(), color="black", lw=1.4,
                    label=f"median {strat.delta.median():.3f}")
    axes[0].set_xlabel("fitted exponent delta")
    axes[0].set_ylabel("names")
    axes[0].set_title(f"delta across {len(strat)} S&P 500 names\n"
                      "Nasdaq, April to September 2024")
    axes[0].set_xlim(0.15, 0.62)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(np.log10(strat.relative_tick), strat.delta,
                     yerr=[strat.delta - strat.delta_lo,
                           strat.delta_hi - strat.delta], fmt="o", ms=3.5,
                     lw=0.7, alpha=0.7, color="#1f4e79")
    slope, intercept = np.polyfit(np.log10(strat.relative_tick), strat.delta, 1)
    grid = np.linspace(np.log10(strat.relative_tick).min(),
                       np.log10(strat.relative_tick).max(), 50)
    axes[1].plot(grid, intercept + slope * grid, color="crimson", lw=1.8,
                 label=f"slope {slope:+.3f} per decade")
    axes[1].axhline(0.5, color="grey", ls="--", lw=1.0)
    axes[1].set_xlabel("log10 relative tick size (one cent over price)")
    axes[1].set_ylabel("delta")
    axes[1].set_title("The tick-floor hypothesis, and its outcome\n"
                      "it predicts a FALL; the raw slope rises")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].hist(strat.c_half, bins=22, color="#b03a2e", alpha=0.85)
    axes[2].axvline(0.69, color="black", ls="--", lw=1.4,
                    label="published AAPL c_raw 0.69")
    axes[2].axvline(strat.c_half.median(), color="#1f4e79", lw=1.4,
                    label=f"median {strat.c_half.median():.3f}")
    axes[2].set_xlabel("prefactor c, delta fixed at 1/2")
    axes[2].set_ylabel("names")
    axes[2].set_title("Prefactor across the cross-section")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
