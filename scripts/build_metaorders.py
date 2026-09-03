#!/usr/bin/env python3
"""Reconstruct proxy metaorders from a LOBSTER message stream and an L1 book.

`data/*_metaorders.csv` was committed with no builder behind it, the same gap
`build_1s_bars.py` closed for the one-second series. This closes it here.

A proxy metaorder is a maximal run of consecutive same-signed fills, following
the public-data approach of arXiv:2503.18199. A fill's LOBSTER `direction` is
the side of the RESTING order it executed against, so the aggressor's sign is
its negation: hitting a resting sell is a buy.

TWO DEFECTS IN THE ORIGINAL FILE, FOUND WHILE RECOVERING THE CONSTRUCTION
------------------------------------------------------------------------
Neither is reproduced here, and both change the published exponent.

1. Fills sharing a timestamp were reordered. The committed file contains a run
   whose `t_start` precedes the previous run's by 59 microseconds -- it is not
   sorted in time at all -- and splits one same-signed run into three because a
   later fill was sorted ahead of two earlier ones. Order within a timestamp is
   the sequence order and is meaningful; this builder keeps message order.

2. `mid_start` was measured AFTER the run's first fill, not before it. The
   module docstring says impact is "the signed mid-price change from immediately
   before the run to immediately after it", and measuring from after the first
   fill drops that fill's own impact from every metaorder. `--mid-convention at`
   reproduces the original for comparison; the default `before` is what the
   method actually calls for.

Inputs are the two artefacts the sibling `lob-engine-cpp` repo produces:

    python scripts/databento_to_lobster.py <mbo.dbn.zst> --out <message.csv>
    build/lob_engine <message.csv> --backend map --depth 1 --book-out <l1.csv>

Usage:
    python scripts/build_metaorders.py \
        --messages MSFT_2024-06-03_message.csv \
        --book MSFT_2024-06-03_l1.csv \
        --out data/MSFT_2024-06-03_metaorders.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOBSTER_DISPLAYED_FILL = 4
_PRICE_SCALE = 10_000.0
_MESSAGE_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


def build(messages_path: Path, book_path: Path, *, mid_convention: str = "before"
          ) -> pd.DataFrame:
    messages = pd.read_csv(messages_path, names=_MESSAGE_COLS, header=None)
    book = pd.read_csv(book_path, usecols=["bid_px_0", "ask_px_0"])
    if len(book) != len(messages):
        raise SystemExit(f"book has {len(book):,} rows but the message file has "
                         f"{len(messages):,}; they must be the same conversion")

    mid = ((book.bid_px_0 + book.ask_px_0) / 2.0 / _PRICE_SCALE).to_numpy()
    # a one-sided book leaves the mid undefined; carry the last known quote
    # rather than dropping the metaorder, which would bias toward calm periods
    mid = pd.Series(mid).ffill().to_numpy()

    event_type = messages.event_type.to_numpy()
    fills = np.flatnonzero(event_type == LOBSTER_DISPLAYED_FILL)
    if fills.size == 0:
        raise SystemExit("no displayed fills in the message stream")

    sign = -messages.direction.to_numpy()[fills]
    size = messages["size"].to_numpy()[fills]
    time = messages.time.to_numpy()[fills]

    # maximal runs of consecutive same-signed fills, in message order
    starts = np.flatnonzero(np.r_[True, sign[1:] != sign[:-1]])
    ends = np.r_[starts[1:] - 1, len(fills) - 1]
    first_row, last_row = fills[starts], fills[ends]

    if mid_convention == "before":
        mid_start = mid[np.maximum(first_row - 1, 0)]
    elif mid_convention == "at":
        mid_start = mid[first_row]
    else:
        raise SystemExit(f"unknown mid convention {mid_convention!r}")

    return pd.DataFrame({
        "sign": sign[starts].astype(float),
        "shares": np.add.reduceat(size, starts),
        "n_fills": np.diff(np.r_[starts, len(fills)]),
        "t_start": time[starts],
        "t_end": time[ends],
        "mid_start": mid_start,
        "mid_end": mid[last_row],
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--messages", type=Path, required=True)
    ap.add_argument("--book", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mid-convention", choices=("before", "at"), default="before",
                    help="'before' measures impact from the mid before the run's "
                         "first fill, as the method calls for; 'at' measures from "
                         "after it, reproducing the original file's behaviour")
    args = ap.parse_args()

    frame = build(args.messages, args.book, mid_convention=args.mid_convention)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"{args.out}: {len(frame):,} proxy metaorders from "
          f"{int(frame.n_fills.sum()):,} fills "
          f"(mid convention: {args.mid_convention})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
