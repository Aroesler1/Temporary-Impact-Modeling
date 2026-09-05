#!/usr/bin/env python3
"""Traded volume per session, deduplicated, straight from the MBO extract.

`run_metaorder_impact.py` used to hardcode one session's volume as a constant.
Participation rate is a ratio and the denominator has to be defined as carefully
as the numerator, so this derives it for every session on the same convention
the metaorder reconstruction uses.

Databento reports a displayed execution TWICE -- once as `T`, the trade print,
and once as `F`, the book-side fill -- sharing a `sequence`. Adding both double
counts by about 43%. So:

    displayed = sum of size over F
    hidden    = sum of size over T whose sequence has no F
    total     = displayed + hidden

The window is the whole extract, 04:00-20:00 exchange time, because
`build_metaorders.py` reconstructs metaorders over the whole extract too. Using
an RTH-only denominator under a full-window numerator would inflate every
participation rate by roughly a tenth.

Needs the `databento` SDK. Writes data/session_volume.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sessions import raw_root, sessions  # noqa: E402


def tally(path: Path) -> dict[str, float]:
    import databento as db

    arr = db.DBNStore.from_file(str(path)).to_ndarray()
    # the ndarray view types `action` as a one-byte string, not an int code
    action = arr["action"]
    fills = action == b"F"
    trades = action == b"T"
    f_sequences = np.unique(arr["sequence"][fills])
    hidden = trades & ~np.isin(arr["sequence"], f_sequences)
    displayed_vol = float(arr["size"][fills].sum())
    hidden_vol = float(arr["size"][hidden].sum())
    return {"displayed_volume_mbo": displayed_vol,
            "hidden_volume_mbo": hidden_vol,
            "total_volume_mbo": displayed_vol + hidden_vol,
            "n_fills": int(fills.sum()),
            "n_hidden_prints": int(hidden.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/session_volume.csv"))
    args = ap.parse_args()

    root = raw_root()
    rows = []
    for sess in sessions():
        path = sess.raw_path("mbo", root)
        if not path.exists():
            print(f"  {sess.key:<16} MISSING {path.name}", file=sys.stderr)
            continue
        row = {"session": sess.key, **tally(path)}
        rows.append(row)
        print(f"  {sess.key:<16} displayed {row['displayed_volume_mbo']:>12,.0f}  "
              f"hidden {row['hidden_volume_mbo']:>12,.0f}  "
              f"total {row['total_volume_mbo']:>12,.0f}")
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
