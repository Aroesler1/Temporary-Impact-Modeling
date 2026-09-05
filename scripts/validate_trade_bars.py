#!/usr/bin/env python3
"""Check the trades schema against the MBO-derived series, before trusting it.

The cross-section is built from the `trades` schema, which is a different feed
view from the MBO stream every earlier result in this repository used. Two
things could be silently wrong and both would still produce a plausible-looking
exponent:

1. the AGGRESSOR SIGN convention. If `side` were the aggressing side rather than
   the resting side, every metaorder in the study would be inverted.
2. DOUBLE COUNTING. MBO reports a displayed execution twice, once as `T` and
   once as `F` sharing a sequence, and `build_volume_tally.py` deduplicates on
   that. Whether the trades schema has already done so has to be measured, not
   assumed. It has: sequences are unique WITHIN a session. Sequence numbers
   reset daily, so a six-month file contains collisions across days that are not
   duplicates at all, and the check below is per session for that reason.

Both are checked against the five AAPL sessions whose MBO-derived one-second
bars are committed. The MBO series counts DISPLAYED fills only, so the two will
not be equal: the trades schema also carries hidden prints. Correlation and the
displayed share are what the check is on.

Usage:
    python scripts/validate_trade_bars.py --symbol AAPL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cross_section as cs  # noqa: E402
import panel  # noqa: E402
from fetch_cross_section_trades import raw_path  # noqa: E402
from sessions import raw_root, sessions  # noqa: E402


def load_trades(symbol: str) -> pd.DataFrame:
    import databento as db

    path = raw_path(symbol, raw_root())
    if not path.exists():
        raise SystemExit(f"{path} not found; run fetch_cross_section_trades.py")
    arr = db.DBNStore.from_file(str(path)).to_ndarray()
    ts = arr["ts_event"].astype(np.int64)
    return pd.DataFrame({
        "ts": ts,
        "price": arr["price"].astype(float) * 1e-9,
        "size": arr["size"].astype(float),
        "sign": cs.aggressor_sign(arr["side"]),
        "sequence": arr["sequence"].astype(np.int64),
    })


def session_slice(trades: pd.DataFrame, date: str) -> pd.DataFrame:
    midnight = pd.Timestamp(date, tz="America/New_York").tz_convert("UTC").value
    sec = (trades.ts.to_numpy() - midnight) // 1_000_000_000
    keep = (sec >= cs.RTH_OPEN_SEC) & (sec < cs.RTH_CLOSE_SEC)
    out = trades[keep].copy()
    out["sec"] = sec[keep]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/cross_section/trades_validation.csv"))
    args = ap.parse_args()

    trades = load_trades(args.symbol)
    print(f"{args.symbol}: {len(trades):,} trade records over the pull window\n")

    rows = []
    for sess in sessions():
        if sess.symbol != args.symbol:
            continue
        day = session_slice(trades, sess.date)
        if day.empty:
            continue
        bars = cs.trade_bars(day.sec.to_numpy(), day.price.to_numpy(),
                             day["size"].to_numpy(), day.sign.to_numpy())
        reference = panel.bars(sess.key)
        joined = reference.merge(bars, on="sec", how="inner",
                                 suffixes=("_mbo", "_trades"))
        corr = float(np.corrcoef(joined.signed_vol_mbo, joined.signed_vol_trades)[0, 1])
        dups = int(len(day) - day.sequence.nunique())
        rows.append({
            "session": sess.key,
            "n_seconds_joined": len(joined),
            "duplicate_sequences_in_session": dups,
            "signed_vol_correlation": corr,
            "displayed_share_of_trade_volume":
                float(reference.volume.sum() / bars.volume.sum()),
            "unsided_share": float((day.sign == 0).mean()),
            "trade_volume": float(bars.volume.sum()),
            "mbo_displayed_volume": float(reference.volume.sum()),
        })
        print(f"  {sess.key:<16} corr(signed volume) {corr:+.4f}   "
              f"sided share of volume "
              f"{rows[-1]['displayed_share_of_trade_volume']:.3f}   "
              f"unsided prints {rows[-1]['unsided_share']:.4f}   "
              f"dup sequences {dups}")

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    worst = float(table.signed_vol_correlation.min())
    print(f"\nlowest correlation across sessions: {worst:+.4f}")
    if worst > 0.9:
        print("VERDICT: the sign convention matches the MBO-derived series. "
              "On the trades schema\n         `side` is the AGGRESSING side "
              "and needs no negation, which is the OPPOSITE\n         of the "
              "MBO convention every earlier result here used.")
    elif worst < -0.9:
        print("VERDICT: THE SIGN IS INVERTED. `aggressor_sign` must be negated.")
    else:
        print("VERDICT: INCONCLUSIVE. Do not build the cross-section on this.")
    print(f"\nsaved -> {args.out}")
    return 0 if abs(worst) > 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
