#!/usr/bin/env python3
"""Pull the trades schema for the cross-sectional sample, at $0.

The trades schema carries price, size and aggressor side and nothing else. That
is exactly what the square-root law needs and it is a fraction of the size of
MBO or MBP-10, which is what makes a hundred names for two quarters possible at
all.

Two gates before any byte moves, both enforced rather than left to the operator:
`metadata.get_billable_size` reports what the request weighs, and
`metadata.get_cost` must be $0.0000 or the pull aborts.

LAYOUT DEVIATION, DELIBERATE. The rest of this repository stores raw extracts as
$DATABENTO_RAW_DIR/<SYMBOL>/<YYYY-MM-DD>.<schema>.dbn.zst, one file per
symbol-day. Here that would be 110 symbols times 126 trading days, near 14,000
files and 14,000 requests. So the trades are stored one file per SYMBOL over the
whole range, <SYMBOL>/<START>_<END>.trades.dbn.zst, and DATA.md says so.

MSFT and INTC are appended as named COMPARISON stocks so the cross-section can
be put beside the three-name study. They were not drawn by the stratified
sampler and are excluded from every cross-sectional statistic; the flag travels
with them in sample.csv.

Usage:
    python scripts/fetch_cross_section_trades.py            # price and weigh
    python scripts/fetch_cross_section_trades.py --confirm
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sessions import DATASET, raw_root  # noqa: E402

START, END = "2024-04-01", "2024-10-01"     # end is exclusive
SCHEMA = "trades"
COMPARISON_SYMBOLS = ("MSFT", "INTC")
_EXCHANGE_TZ = ZoneInfo("America/New_York")


def window() -> tuple[datetime, datetime]:
    return (datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=_EXCHANGE_TZ),
            datetime.strptime(END, "%Y-%m-%d").replace(tzinfo=_EXCHANGE_TZ))


def raw_path(symbol: str, root: Path) -> Path:
    return root / symbol / f"{START}_{END}.{SCHEMA}.dbn.zst"


def sample_symbols(sample_csv: Path) -> pd.DataFrame:
    """The drawn names plus the comparison names, idempotently.

    This runs twice, once to price and once to fetch, so it must not relabel
    anything on the second pass. Assigning `role` unconditionally would mark the
    comparison names as stratified the moment the file was read back, and they
    would then enter every cross-sectional statistic they are explicitly
    excluded from.
    """
    frame = pd.read_csv(sample_csv)
    if "role" not in frame.columns:
        frame["role"] = "stratified"
    frame["role"] = frame.role.fillna("stratified")
    extra = [s for s in COMPARISON_SYMBOLS if s not in set(frame.symbol)]
    if extra:
        frame = pd.concat([frame, pd.DataFrame({"symbol": extra,
                                                "role": "comparison"})],
                          ignore_index=True)
    return frame.sort_values("symbol").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path,
                    default=Path("data/cross_section/sample.csv"))
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--allow-cost", action="store_true",
                    help="permit a non-zero get_cost; off by default on purpose")
    ap.add_argument("--out", type=Path,
                    default=Path("data/cross_section/pull_manifest.csv"))
    args = ap.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        print("DATABENTO_API_KEY is not set", file=sys.stderr)
        return 1
    import databento as db

    frame = sample_symbols(args.sample)
    symbols = frame.symbol.tolist()
    frame.to_csv(args.sample, index=False)
    client = db.Historical()
    root = raw_root()
    start, end = window()

    size = client.metadata.get_billable_size(
        dataset=DATASET, symbols=symbols, schema=SCHEMA, start=start, end=end,
        stype_in="raw_symbol")
    cost = client.metadata.get_cost(
        dataset=DATASET, symbols=symbols, schema=SCHEMA, start=start, end=end,
        stype_in="raw_symbol")
    print(f"{DATASET} {SCHEMA}, {len(symbols)} symbols "
          f"({int((frame.role == 'comparison').sum())} comparison), "
          f"{START} to {END}")
    print(f"  billable size  {size:,} bytes  ({size / 1e9:.2f} GB uncompressed)")
    print(f"  get_cost       ${cost:.4f}")
    pd.DataFrame([{"dataset": DATASET, "schema": SCHEMA, "start": START,
                   "end": END, "n_symbols": len(symbols),
                   "billable_size_bytes": int(size), "get_cost_usd": float(cost)}
                  ]).to_csv(args.out, index=False)

    if cost > 0 and not args.allow_cost:
        print("\nget_cost is not $0. Aborting: the entitlement covers XNAS.ITCH "
              "outright, so a non-zero price means the request is wrong.",
              file=sys.stderr)
        return 2
    if not args.confirm:
        print("\ndry run; nothing downloaded. re-run with --confirm to fetch.")
        return 0

    print()
    done = 0
    for symbol in symbols:
        out = raw_path(symbol, root)
        if out.exists():
            done += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        client.timeseries.get_range(dataset=DATASET, symbols=[symbol],
                                    schema=SCHEMA, start=start, end=end,
                                    stype_in="raw_symbol", path=str(out))
        done += 1
        print(f"  [{done:>3}/{len(symbols)}] {symbol:<6} "
              f"{out.stat().st_size / 1e6:>7.1f} MB", flush=True)
    total = sum(raw_path(s, root).stat().st_size for s in symbols
                if raw_path(s, root).exists())
    print(f"\n{done} symbols on disk, {total / 1e9:.2f} GB compressed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
