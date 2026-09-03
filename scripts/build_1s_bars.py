#!/usr/bin/env python3
"""Build the one-second series the propagator is calibrated on.

`data/*_1s.csv` used to be committed with no script behind it, which made the
headline R^2 unreproducible from source: the input existed but the path from
raw vendor data to that input did not. This closes that gap, and it is also what
lets a second session be built on exactly the same convention as the first
rather than on a hand-rolled variant.

Inputs are the two artefacts the sibling `lob-engine-cpp` repo produces from one
Databento MBO extract:

    python scripts/databento_to_lobster.py <mbo.dbn.zst> --out <message.csv>
    build/lob_engine <message.csv> --backend map --depth 1 --book-out <l1.csv>

Convention (recovered from, and verified against, the committed MSFT series):

* Regular trading hours only, 09:30:00 to 16:00:00 exchange time -> LOBSTER
  seconds [34200, 57600).
* One row per second that carries at least one message. Seconds with no message
  at all are absent rather than zero-filled, so the bar grid is event-supported;
  on MSFT 2024-06-03 that drops 10 of 23,400 seconds.
* `signed_vol` sums book-affecting displayed fills only (LOBSTER event type 4).
  A fill's `direction` is the side of the RESTING order it executed against, so
  the aggressor's sign is its negation: hitting a resting sell is a buy.
  Hidden prints (type 5) are excluded -- they move no displayed liquidity, and
  including them changes the series materially (max per-second difference of
  366,836 shares on MSFT 2024-06-03).
* `mid` is the LAST mid within the second, from the engine's reconstructed L1
  book. Last rather than mean or first: the propagator regresses one-second
  returns on flow, and a close-to-close return needs the state at the boundary,
  not an average over the interval.

Usage:
    python scripts/build_1s_bars.py \
        --messages INTC_2024-08-02_message.csv \
        --book INTC_2024-08-02_l1.csv \
        --out data/INTC_2024-08-02_1s.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# LOBSTER seconds-since-exchange-midnight for the regular session.
RTH_OPEN = 34_200      # 09:30:00
RTH_CLOSE = 57_600     # 16:00:00
LOBSTER_DISPLAYED_FILL = 4
_PRICE_SCALE = 10_000.0  # LOBSTER integer prices are 1e-4 dollars

_MESSAGE_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


def build(messages_path: Path, book_path: Path) -> pd.DataFrame:
    messages = pd.read_csv(messages_path, names=_MESSAGE_COLS, header=None)
    messages = messages[(messages.time >= RTH_OPEN) & (messages.time < RTH_CLOSE)]
    if messages.empty:
        raise SystemExit(f"{messages_path} has no messages inside RTH")
    message_second = messages.time.astype(np.int64)

    # the bar grid: every second the feed actually spoke in
    seconds = np.unique(message_second)

    fills = messages[messages.event_type == LOBSTER_DISPLAYED_FILL]
    signed = (-fills.direction * fills["size"]).groupby(
        fills.time.astype(np.int64)).sum().reindex(seconds, fill_value=0)

    book = pd.read_csv(book_path)
    book = book[(book.timestamp >= RTH_OPEN) & (book.timestamp < RTH_CLOSE)]
    if len(book) != len(messages):
        raise SystemExit(
            f"book has {len(book):,} RTH rows but the message file has "
            f"{len(messages):,}; they must come from the same conversion")
    mid = (book.bid_px_0 + book.ask_px_0) / 2.0 / _PRICE_SCALE
    mid = mid.groupby(book.timestamp.astype(np.int64)).last().reindex(seconds)

    frame = pd.DataFrame({"sec": seconds,
                          "signed_vol": signed.to_numpy(dtype=float),
                          "mid": mid.to_numpy(dtype=float)})
    # A one-sided book leaves mid undefined; carry the last known quote rather
    # than dropping the second, which would silently shorten the return series.
    if frame.mid.isna().any():
        n = int(frame.mid.isna().sum())
        frame["mid"] = frame.mid.ffill()
        print(f"  warning: {n:,} seconds had no two-sided book; forward-filled mid",
              file=sys.stderr)
    return frame.dropna(subset=["mid"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--messages", type=Path, required=True,
                    help="LOBSTER message CSV from databento_to_lobster.py")
    ap.add_argument("--book", type=Path, required=True,
                    help="engine L1 book CSV from lob_engine --depth 1 --book-out")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None,
                    help="existing series to diff against; exits non-zero if it differs")
    args = ap.parse_args()

    frame = build(args.messages, args.book)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"{args.out}: {len(frame):,} one-second bars "
          f"[{int(frame.sec.min())}, {int(frame.sec.max())}]")

    if args.compare is not None:
        ref = pd.read_csv(args.compare)
        same = (len(ref) == len(frame)
                and np.array_equal(ref.sec.to_numpy(), frame.sec.to_numpy())
                and np.array_equal(ref.signed_vol.to_numpy(), frame.signed_vol.to_numpy())
                and np.array_equal(ref.mid.to_numpy(), frame.mid.to_numpy()))
        print(f"compare vs {args.compare}: {'IDENTICAL' if same else 'DIFFERS'}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
