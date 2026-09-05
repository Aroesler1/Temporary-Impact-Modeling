#!/usr/bin/env python3
"""Reconstruct proxy metaorders for every sampled name, from trades alone.

One pass per symbol over its six-month trades extract. For each session:

  1. one-second bars of signed volume, total volume and last trade price;
  2. V_D, the day's total Nasdaq RTH volume, sided prints and unsided prints
     together, since an unsided hidden print is still traded volume;
  3. sigma_D from five-minute trade prices, scaled to a session;
  4. proxy metaorders from `crossover.bin_metaorders`, which is the
     arXiv 2606.24019 recipe already written for the three-name study and is
     reused here rather than reimplemented.

The raw metaorders go to a gitignored working directory. What is committed is
the per-symbol fits and the binned aggregates, under `reports/cross_section/`.

Usage:
    python scripts/build_cross_section_metaorders.py
    python scripts/build_cross_section_metaorders.py --symbol AAPL --symbol MSFT
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cross_section as cs  # noqa: E402
from crossover import bin_metaorders  # noqa: E402
from fetch_cross_section_trades import raw_path  # noqa: E402
from sessions import raw_root  # noqa: E402

MIN_TRADES_PER_SESSION = 500     # below this a session has no flow to speak of


def load(symbol: str, root: Path) -> pd.DataFrame:
    """Every RTH trade in the pull window, with a session date and a second."""
    import databento as db

    path = raw_path(symbol, root)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = db.DBNStore.from_file(str(path)).to_ndarray()
    local = pd.to_datetime(arr["ts_event"].astype(np.int64), unit="ns", utc=True
                           ).tz_convert("America/New_York")
    sec = (local.hour * 3600 + local.minute * 60 + local.second).to_numpy(np.int64)
    keep = (sec >= cs.RTH_OPEN_SEC) & (sec < cs.RTH_CLOSE_SEC)
    return pd.DataFrame({
        "date": local.strftime("%Y-%m-%d").to_numpy()[keep],
        "sec": sec[keep],
        "price": arr["price"].astype(float)[keep] * 1e-9,
        "size": arr["size"].astype(float)[keep],
        "sign": cs.aggressor_sign(arr["side"])[keep],
    })


def build_symbol(symbol: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = load(symbol, root)
    metaorders, daily = [], []
    for date, day in trades.groupby("date", sort=True):
        if len(day) < MIN_TRADES_PER_SESSION:
            continue
        bars = cs.trade_bars(day.sec.to_numpy(), day.price.to_numpy(),
                             day["size"].to_numpy(), day.sign.to_numpy())
        volume = float(day["size"].sum())
        sigma = cs.realised_vol_5min(day.sec.to_numpy(), day.price.to_numpy())
        if not np.isfinite(sigma) or sigma <= 0 or volume <= 0:
            continue
        profile = cs.halfhour_vol_profile(day.sec.to_numpy(), day.price.to_numpy())
        daily.append({"symbol": symbol, "date": date, "n_trades": len(day),
                      "volume_rth": volume, "sigma_5min": sigma,
                      "unsided_share": float(
                          day.loc[day.sign == 0, "size"].sum() / volume),
                      "close": float(day.price.iloc[-1]),
                      **{f"hh_{i}": profile[i] for i in range(len(profile))}})
        orders = bin_metaorders(bars, volume)
        if orders.empty:
            continue
        orders.insert(0, "symbol", symbol)
        orders.insert(1, "date", date)
        orders["sigma_d"] = sigma
        metaorders.append(orders)

    return (pd.concat(metaorders, ignore_index=True) if metaorders
            else pd.DataFrame()), pd.DataFrame(daily)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", action="append", default=None)
    ap.add_argument("--sample", type=Path,
                    default=Path("data/cross_section/sample.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/cross_section/metaorders"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sample = pd.read_csv(args.sample)
    symbols = args.symbol or sample.symbol.tolist()
    root = raw_root()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, symbol in enumerate(symbols, 1):
        out = args.out_dir / f"{symbol}.csv"
        daily_out = args.out_dir / f"{symbol}_daily.csv"
        if out.exists() and daily_out.exists() and not args.force:
            print(f"  [{i:>3}/{len(symbols)}] {symbol:<6} already built")
            continue
        started = time.time()
        try:
            orders, daily = build_symbol(symbol, root)
        except FileNotFoundError:
            print(f"  [{i:>3}/{len(symbols)}] {symbol:<6} MISSING extract",
                  file=sys.stderr)
            continue
        if orders.empty:
            print(f"  [{i:>3}/{len(symbols)}] {symbol:<6} no metaorders passed "
                  f"the filters", file=sys.stderr)
        else:
            orders.to_csv(out, index=False)
        daily.to_csv(daily_out, index=False)
        print(f"  [{i:>3}/{len(symbols)}] {symbol:<6} {len(daily):>3} sessions  "
              f"{len(orders):>6,} metaorders  {time.time() - started:>5.1f}s",
              flush=True)
    print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
