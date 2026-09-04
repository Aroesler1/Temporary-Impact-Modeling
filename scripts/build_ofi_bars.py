#!/usr/bin/env python3
"""Derive one-second order flow imbalance for every session from MBP-10.

Per-event OFI is computed at each of the ten displayed levels from consecutive
vendor book states, then summed inside each RTH second onto the same grid the
propagator's bars use, so the two flows can be put in one regression without a
join that guesses.

Depth normalisation follows Cont, Cucuringu and Zhang: each level's per-second
OFI is divided by the session's average displayed size across the top ten
levels, which makes the levels commensurable before the principal component is
taken. The component is fitted on the first 70% of seconds only -- the same
chronological split every other out-of-sample number in this repo uses.

Output: data/ofi/<SYMBOL>_<DATE>_1s_ofi.csv with sec, ofi_best, ofi_sum,
ofi_integrated, plus the ten PCA weights in data/ofi/pca_weights.csv.

Needs the `databento` SDK; the committed OFI series reproduce Part 4 without it.

Usage:
    python scripts/build_ofi_bars.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import orderflow as of  # noqa: E402
from build_bookwalk import RTH_CLOSE_SEC, RTH_OPEN_SEC, _exchange_second  # noqa: E402
from sessions import raw_root, sessions  # noqa: E402

DEPTH = 10
TRAIN_FRAC = 0.7


def build(path: Path, date: str) -> tuple[pd.DataFrame, np.ndarray]:
    import databento as db

    arr = db.DBNStore.from_file(str(path)).to_ndarray()
    sec = _exchange_second(arr["ts_event"].astype(np.int64), date)
    keep = (sec >= RTH_OPEN_SEC) & (sec < RTH_CLOSE_SEC)
    arr, sec = arr[keep], sec[keep]
    if arr.size == 0:
        raise SystemExit(f"{path}: no records inside RTH")

    per_level, depths = [], []
    for i in range(DEPTH):
        bp = arr[f"bid_px_{i:02d}"].astype(float)
        ap = arr[f"ask_px_{i:02d}"].astype(float)
        bs = arr[f"bid_sz_{i:02d}"].astype(float)
        asz = arr[f"ask_sz_{i:02d}"].astype(float)
        per_level.append(of.level_ofi(np.nan_to_num(bp), np.nan_to_num(bs),
                                      np.nan_to_num(ap), np.nan_to_num(asz)))
        depths.append(np.nanmean(np.concatenate([bs, asz])))
    events = pd.DataFrame({f"ofi_{i}": v for i, v in enumerate(per_level)})
    events["sec"] = sec

    # sum inside the second, then normalise by average displayed size across the
    # top ten levels (Cont, Cucuringu and Zhang), which makes levels commensurable
    binned = events.groupby("sec").sum().sort_index()
    qm = float(np.mean(depths))
    cols = [f"ofi_{i}" for i in range(DEPTH)]
    normalised = binned[cols].to_numpy(float) / (DEPTH * qm)

    split = int(len(normalised) * TRAIN_FRAC)
    integrated, weights = of.integrate_pca(normalised, fit_rows=slice(0, split))
    return pd.DataFrame({
        "sec": binned.index.to_numpy(),
        "ofi_best": normalised[:, 0],
        "ofi_sum": normalised.sum(axis=1),
        "ofi_integrated": integrated,
    }), weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", action="append", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/ofi"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = raw_root()
    weight_rows = []
    for sess in sessions():
        if args.symbol and sess.symbol not in set(args.symbol):
            continue
        path = sess.raw_path("mbp-10", root)
        if not path.exists():
            print(f"  {sess.key:<16} MISSING {path.name}", file=sys.stderr)
            continue
        frame, weights = build(path, sess.date)
        frame.to_csv(args.out_dir / f"{sess.key}_1s_ofi.csv", index=False,
                     float_format="%.6g")
        weight_rows.append({"session": sess.key,
                            **{f"w_{i}": float(w) for i, w in enumerate(weights)}})
        print(f"  {sess.key:<16} {len(frame):>6,} seconds   "
              f"PCA w0={weights[0]:+.2f} w9={weights[-1]:+.2f}")

    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(args.out_dir / "pca_weights.csv", index=False)
        print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
