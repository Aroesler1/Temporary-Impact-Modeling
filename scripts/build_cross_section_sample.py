#!/usr/bin/env python3
"""Draw the cross-sectional sample, and record it before any trade is pulled.

Order matters here. The stratification variables come from June 2024 daily bars,
the draw is seeded and written down, and only then are trades requested. Drawing
after looking at any impact result would make the cross-section a selected
sample rather than a stratified one.

Membership is point in time, from the alpha repository's local
`sp500_membership_daily.parquet`. No WRDS query, and the path is supplied rather
than committed.

Usage:
    export SP500_MEMBERSHIP_PARQUET=~/path/to/sp500_membership_daily.parquet
    python scripts/build_cross_section_sample.py               # price only
    python scripts/build_cross_section_sample.py --confirm
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cross_section as cs  # noqa: E402
from sessions import DATASET  # noqa: E402

MEMBERSHIP_DATE = "2024-06-28"
JUNE_START, JUNE_END = "2024-06-01", "2024-07-01"
_EXCHANGE_TZ = ZoneInfo("America/New_York")


def membership_path(arg: Path | None) -> Path:
    path = arg or os.environ.get("SP500_MEMBERSHIP_PARQUET")
    if not path:
        raise SystemExit(
            "point-in-time membership file not given. Pass --membership or set "
            "SP500_MEMBERSHIP_PARQUET; it lives in the alpha repository under "
            "data/us_equities/reference/sp500_membership_daily.parquet")
    return Path(path).expanduser()


def members(path: Path, date: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    on_date = frame[(frame.date == date) & frame.active]
    if on_date.empty:
        raise SystemExit(f"no active members on {date} in {path}")
    return on_date[["symbol", "permno"]].drop_duplicates().sort_values("symbol")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--membership", type=Path, default=None)
    ap.add_argument("--confirm", action="store_true",
                    help="actually pull the June daily bars and draw the sample")
    ap.add_argument("--out-dir", type=Path, default=Path("data/cross_section"))
    ap.add_argument("--names-per-cell", type=int, default=cs.NAMES_PER_CELL)
    args = ap.parse_args()

    universe = members(membership_path(args.membership), MEMBERSHIP_DATE)
    symbols = universe.symbol.tolist()
    print(f"S&P 500 members on {MEMBERSHIP_DATE}: {len(symbols)}")

    if not os.environ.get("DATABENTO_API_KEY"):
        print("DATABENTO_API_KEY is not set", file=sys.stderr)
        return 1
    import databento as db

    client = db.Historical()
    start = datetime.strptime(JUNE_START, "%Y-%m-%d").replace(tzinfo=_EXCHANGE_TZ)
    end = datetime.strptime(JUNE_END, "%Y-%m-%d").replace(tzinfo=_EXCHANGE_TZ)
    cost = client.metadata.get_cost(dataset=DATASET, symbols=symbols,
                                    schema="ohlcv-1d", start=start, end=end,
                                    stype_in="raw_symbol")
    print(f"ohlcv-1d, {len(symbols)} symbols, {JUNE_START}..{JUNE_END}: "
          f"${cost:.4f}")
    if cost > 0:
        print("get_cost is not $0; aborting.", file=sys.stderr)
        return 2
    if not args.confirm:
        print("dry run; nothing downloaded. re-run with --confirm.")
        return 0

    store = client.timeseries.get_range(dataset=DATASET, symbols=symbols,
                                        schema="ohlcv-1d", start=start, end=end,
                                        stype_in="raw_symbol")
    daily = store.to_df()
    daily = pd.DataFrame({
        "symbol": daily["symbol"].to_numpy(),
        "date": pd.to_datetime(daily.index).strftime("%Y-%m-%d"),
        "close": pd.to_numeric(daily["close"], errors="coerce").to_numpy(float),
        "high": pd.to_numeric(daily["high"], errors="coerce").to_numpy(float),
        "low": pd.to_numeric(daily["low"], errors="coerce").to_numpy(float),
        "volume": pd.to_numeric(daily["volume"], errors="coerce").to_numpy(float),
    })
    stats = daily.groupby("symbol").agg(
        june_days=("close", "size"),
        mean_close=("close", "mean"),
        mean_volume=("volume", "mean"),
        mean_high_low_bp=("close", "size")).reset_index()
    hl = daily.assign(bp=(daily.high - daily.low) / daily.close * 1e4)
    stats["mean_high_low_bp"] = stats.symbol.map(hl.groupby("symbol").bp.mean())
    stats["relative_tick"] = 0.01 / stats.mean_close
    stats["dollar_volume"] = stats.mean_close * stats.mean_volume
    # a name that barely printed on Nasdaq in June has no venue liquidity to
    # measure impact against, and would be a different study
    stats = stats[stats.june_days >= 15]

    missing = sorted(set(symbols) - set(stats.symbol))
    print(f"resolved {len(stats)} of {len(symbols)} symbols on {DATASET}; "
          f"{len(missing)} missing or too thin")
    if missing:
        print("  missing:", ", ".join(missing[:20])
              + (" ..." if len(missing) > 20 else ""))

    strat = cs.stratify(stats, names_per_cell=args.names_per_cell)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out_dir / "june_stats.csv", index=False)
    strat.frame.to_csv(args.out_dir / "universe_with_cells.csv", index=False)
    strat.sample.to_csv(args.out_dir / "sample.csv", index=False)
    strat.cell_counts.to_csv(args.out_dir / "cell_counts.csv", index=False)
    pd.DataFrame([{"seed": strat.seed, "names_per_cell": args.names_per_cell,
                   "membership_date": MEMBERSHIP_DATE,
                   "n_universe": len(stats), "n_sample": len(strat.sample),
                   "short_cells": ";".join(strat.short_cells),
                   "n_missing_symbols": len(missing)}]).to_csv(
        args.out_dir / "sample_provenance.csv", index=False)

    print(f"\nseed {strat.seed}, {args.names_per_cell} names a cell\n")
    print(strat.cell_counts.to_string(index=False))
    print(f"\ndrawn: {len(strat.sample)} names")
    if strat.short_cells:
        print(f"short cells (took all available): {', '.join(strat.short_cells)}")
    print("\n" + ", ".join(strat.sample.symbol))
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
