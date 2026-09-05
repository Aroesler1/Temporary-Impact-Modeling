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
* One row per BIN that carries at least one message. The bin is one second by
  default; `--bin-ms` makes it finer, and `sec` then carries the bin's start
  time in seconds. At the default of 1000 ms the output is byte for byte what
  it was before the option existed, which a test asserts. Seconds with no message
  at all are absent rather than zero-filled, so the bar grid is event-supported;
  on MSFT 2024-06-03 that drops 10 of 23,400 seconds.
* `volume` is the UNSIGNED sum of the same displayed fills, so a second's
  one-sidedness can be measured; `signed_vol` alone cannot distinguish a quiet
  one-way second from a busy two-way one.
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


def _bin_index(time: pd.Series, bin_ms: int) -> np.ndarray:
    """Which bin each timestamp falls in, as an integer count of bins.

    LOBSTER times carry microsecond resolution, so the index is computed in
    integer microseconds rather than by multiplying a float by 1000, which would
    put a timestamp exactly on a bin edge into the wrong bin often enough to
    matter over four million messages.
    """
    micros = np.rint(time.to_numpy(float) * 1e6).astype(np.int64)
    return micros // (int(bin_ms) * 1000)


def _bin_start(index: np.ndarray, bin_ms: int) -> np.ndarray:
    """Bin start in seconds; integer seconds when the bin divides a second.

    The multiply happens in INTEGER milliseconds before the divide. Scaling by a
    precomputed 0.1 instead gives 34201.200000000004 for the twelfth bin of a
    second, which then reaches the CSV and breaks any join on the column.
    """
    if int(bin_ms) % 1000 == 0:
        return index * (int(bin_ms) // 1000)
    return (index * int(bin_ms)) / 1000.0


def build(messages_path: Path, book_path: Path, bin_ms: int = 1000) -> pd.DataFrame:
    if int(bin_ms) <= 0:
        raise SystemExit("--bin-ms must be positive")
    messages = pd.read_csv(messages_path, names=_MESSAGE_COLS, header=None)
    messages = messages[(messages.time >= RTH_OPEN) & (messages.time < RTH_CLOSE)]
    if messages.empty:
        raise SystemExit(f"{messages_path} has no messages inside RTH")
    message_second = _bin_index(messages.time, bin_ms)

    # the bar grid: every bin the feed actually spoke in
    seconds = np.unique(message_second)

    fills = messages[messages.event_type == LOBSTER_DISPLAYED_FILL]
    fill_second = pd.Series(_bin_index(fills.time, bin_ms), index=fills.index)
    signed = (-fills.direction * fills["size"]).groupby(
        fill_second).sum().reindex(seconds, fill_value=0)
    # unsigned displayed volume, needed for the direction-dominance filter in
    # `crossover.py`; signed volume alone cannot say how one-sided a bin was
    volume = fills["size"].groupby(fill_second).sum().reindex(seconds, fill_value=0)

    book = pd.read_csv(book_path)
    book = book[(book.timestamp >= RTH_OPEN) & (book.timestamp < RTH_CLOSE)]
    if len(book) != len(messages):
        raise SystemExit(
            f"book has {len(book):,} RTH rows but the message file has "
            f"{len(messages):,}; they must come from the same conversion")
    mid = (book.bid_px_0 + book.ask_px_0) / 2.0 / _PRICE_SCALE
    mid = mid.groupby(_bin_index(book.timestamp, bin_ms)).last().reindex(seconds)

    frame = pd.DataFrame({"sec": _bin_start(seconds, bin_ms),
                          "signed_vol": signed.to_numpy(dtype=float),
                          "volume": volume.to_numpy(dtype=float),
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
    ap.add_argument("--bin-ms", type=int, default=1000,
                    help="bin width in milliseconds; 1000 is the original "
                         "one-second grid and reproduces it exactly")
    ap.add_argument("--compare", type=Path, default=None,
                    help="existing series to diff against; exits non-zero if it differs")
    args = ap.parse_args()

    frame = build(args.messages, args.book, bin_ms=args.bin_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"{args.out}: {len(frame):,} {args.bin_ms} ms bars "
          f"[{frame.sec.min():g}, {frame.sec.max():g}]")

    if args.compare is not None:
        ref = pd.read_csv(args.compare)
        # compares the three columns the original convention defined; `volume`
        # was added later and a reference file predating it must still match
        same = (len(ref) == len(frame)
                and np.array_equal(ref.sec.to_numpy(), frame.sec.to_numpy())
                and np.array_equal(ref.signed_vol.to_numpy(), frame.signed_vol.to_numpy())
                and np.array_equal(ref.mid.to_numpy(), frame.mid.to_numpy()))
        print(f"compare vs {args.compare}: {'IDENTICAL' if same else 'DIFFERS'}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
