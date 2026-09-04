#!/usr/bin/env python3
"""Pull daily bars for the three panel names and derive ADV and sigma_D.

Part 1 needs sizes normalised by 20-day average daily volume, and Parts 1 and 2
need impact in units of daily volatility. Neither can come from a single
session: a 20-day ADV needs 20 days, and a daily volatility estimated from one
session's intraday path is a different object with a different bias. So this
pulls `ohlcv-1d` over a trailing window and computes both from CLOSED days
strictly BEFORE each session, which also keeps the normaliser free of
look-ahead into the day being measured.

The daily bars are tiny, so only the derived table is kept:
`data/daily_reference.csv` carries one row per symbol-day in the panel with the
trailing 20-day ADV and the trailing 20-day close-to-close volatility. The raw
daily DBN is streamed to memory and never written to disk.

These daily bars span the whole UTC day, so their close is the last Nasdaq
print by 20:00 exchange time and includes the after-hours session. That is
visible in the panel: INTC's post-earnings collapse on the evening of
2024-08-01 lands in the 2024-08-01 bar, not the 2024-08-02 one.

Volumes are XNAS ONLY, not consolidated tape: `ohlcv-1d` on `XNAS.ITCH` counts
what printed on Nasdaq, which is roughly a third of MSFT's consolidated volume.
Every participation rate in this repo is therefore a share of Nasdaq volume,
and the column names say so. Using it consistently on both sides of the ratio
is what matters for the exponent; the level of c does depend on the choice.

Usage:
    python scripts/fetch_daily_reference.py            # price only
    python scripts/fetch_daily_reference.py --confirm
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sessions import DATASET, sessions  # noqa: E402

_EXCHANGE_TZ = ZoneInfo("America/New_York")
# wide enough that every panel session has 20 closed days behind it, including
# the earliest (2024-02-01)
_START = "2023-10-01"
_END = "2025-01-01"
_WINDOW = 20


def derive(daily: pd.DataFrame) -> pd.DataFrame:
    """Trailing ADV and close-to-close sigma for each panel session.

    Both windows END on the last close STRICTLY BEFORE the session, so the
    normaliser never contains the day whose impact it normalises.
    """
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    rows = []
    for sess in sessions():
        hist = daily[(daily.symbol == sess.symbol)
                     & (daily.date < sess.date)].tail(_WINDOW)
        same = daily[(daily.symbol == sess.symbol) & (daily.date == sess.date)]
        if len(hist) < _WINDOW:
            raise SystemExit(f"{sess.key}: only {len(hist)} prior closes, need {_WINDOW}")
        logret = np.diff(np.log(hist.close.to_numpy(float)))
        rows.append({
            "symbol": sess.symbol,
            "date": sess.date,
            "adv_20d_xnas": float(hist.volume.mean()),
            "sigma_daily_20d": float(logret.std(ddof=1)),
            "close_prev": float(hist.close.iloc[-1]),
            "session_volume_xnas": float(same.volume.iloc[0]) if len(same) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/daily_reference.csv"))
    args = ap.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        print("DATABENTO_API_KEY is not set", file=sys.stderr)
        return 1

    import databento as db

    client = db.Historical()
    symbols = sorted({s.symbol for s in sessions()})
    start = datetime.strptime(_START, "%Y-%m-%d").replace(tzinfo=_EXCHANGE_TZ)
    end = datetime.strptime(_END, "%Y-%m-%d").replace(tzinfo=_EXCHANGE_TZ)

    cost = client.metadata.get_cost(dataset=DATASET, symbols=symbols,
                                    schema="ohlcv-1d", start=start, end=end,
                                    stype_in="raw_symbol")
    print(f"  ohlcv-1d {','.join(symbols)} {_START}..{_END}  ${cost:.4f}")
    if cost > 0:
        print("get_cost is not $0; aborting.", file=sys.stderr)
        return 2
    if not args.confirm:
        print("dry run; nothing downloaded. re-run with --confirm.")
        return 0

    store = client.timeseries.get_range(dataset=DATASET, symbols=symbols,
                                        schema="ohlcv-1d", start=start, end=end,
                                        stype_in="raw_symbol")
    frame = store.to_df()
    daily = pd.DataFrame({
        "symbol": frame["symbol"].to_numpy(),
        # ohlcv-1d stamps each bar at UTC MIDNIGHT OF THE SESSION DATE, so the
        # label is read off the UTC index directly. Converting it to exchange
        # time first would move every bar to 20:00 the previous evening and
        # shift the whole series back one trading day.
        "date": pd.to_datetime(frame.index).strftime("%Y-%m-%d"),
        "close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(float),
        "volume": pd.to_numeric(frame["volume"], errors="coerce").to_numpy(float),
    })
    out = derive(daily)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
