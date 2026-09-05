#!/usr/bin/env python3
"""Derive the displayed-ladder cost curve for every session from MBP-10.

One pass per symbol-day over the vendor's own ten-level book. Vendor depth
rather than a reconstruction, so the result is a statement about the market and
not about the book-building code.

Sampling: the LAST book state in each RTH second. That is one snapshot per
second the feed spoke in, matching the one-second grid the propagator uses, and
it is a time sample rather than an event sample -- an event sample would
over-weight the busiest seconds, which are also the thinnest, and would bias the
cost curve up.

Sizes: a fixed log grid in shares, common to all three names so the pooled fit
is not an artefact of three different grids. A snapshot contributes to size x
only if its ten displayed levels hold x shares; `participating` records the
fraction that do, at every bin.

Outputs, all derived aggregates small enough to commit:
    data/bookwalk/<SYMBOL>_<DATE>_bins.csv     bin table, one per size
                                               normalisation and side
    data/bookwalk/original_recipe.csv          the notebook's fit, and the same
                                               fit with each of its filters lifted
    data/bookwalk/session_scales.csv           mid, spread, depth and volatility
                                               scales used for the normalisations

Needs the `databento` SDK; the committed bin tables reproduce every fit in
`scripts/run_bookwalk.py` without it.

Usage:
    python scripts/build_bookwalk.py                    # every session
    python scripts/build_bookwalk.py --symbol AAPL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bookwalk as bw  # noqa: E402
from sessions import raw_root, sessions  # noqa: E402

RTH_OPEN_SEC = 34_200      # 09:30:00 exchange time
RTH_CLOSE_SEC = 57_600     # 16:00:00
_NS_PER_S = 1_000_000_000

# 20 points per decade from 1 to ~316k shares: fine enough that MSFT, whose ten
# levels usually hold only a few thousand shares, still gets ~70 grid points
# inside its displayed depth
SIZE_GRID = np.unique(np.round(np.logspace(0.0, 5.5, 111)))

SIZE_NORMALISATIONS = ("shares", "x_over_adv", "x_over_depth")
DEPTH = 10


def _exchange_second(ts_event_ns: np.ndarray, date: str) -> np.ndarray:
    """Seconds since exchange-local midnight, as LOBSTER counts them."""
    midnight = (pd.Timestamp(date, tz="America/New_York")
                .tz_convert("UTC").value)
    return (ts_event_ns - midnight) // _NS_PER_S


def load_snapshots(path: Path, date: str) -> pd.DataFrame:
    """Last displayed book state in each RTH second, prices in dollars."""
    import databento as db

    arr = db.DBNStore.from_file(str(path)).to_ndarray()
    sec = _exchange_second(arr["ts_event"].astype(np.int64), date)
    in_rth = (sec >= RTH_OPEN_SEC) & (sec < RTH_CLOSE_SEC)
    arr, sec = arr[in_rth], sec[in_rth]
    if arr.size == 0:
        raise SystemExit(f"{path}: no records inside RTH")

    # last index within each second; ts_event is non-decreasing in the file
    last = np.flatnonzero(np.r_[sec[1:] != sec[:-1], True])
    cols = {"sec": sec[last]}
    for i in range(DEPTH):
        for side in ("bid", "ask"):
            cols[f"{side}_px_{i:02d}"] = (
                arr[f"{side}_px_{i:02d}"][last].astype(float) * bw.PRICE_SCALE)
            cols[f"{side}_sz_{i:02d}"] = arr[f"{side}_sz_{i:02d}"][last].astype(float)
    frame = pd.DataFrame(cols)
    # an undefined or crossed touch is not a book to walk
    ok = ((frame.ask_px_00 > frame.bid_px_00) & (frame.bid_px_00 > 0)
          & (frame.ask_sz_00 > 0) & (frame.bid_sz_00 > 0))
    return frame[ok].reset_index(drop=True)


def session_bins(book: pd.DataFrame, adv: float, sigma_d: float,
                 n_bins: int = 40) -> tuple[pd.DataFrame, dict]:
    """Bin tables for both sides and all three size normalisations."""
    walk = bw.walk_costs(book, SIZE_GRID, depth=DEPTH)
    n_snap = len(book)
    grid = np.broadcast_to(SIZE_GRID[None, :], (n_snap, len(SIZE_GRID)))

    out = []
    for side, cost_key, depth_key in (("buy", "buy_cost", "depth_l1_ask"),
                                      ("sell", "sell_cost", "depth_l1_bid")):
        cost = walk[cost_key]                              # $/share above mid
        mid = walk["mid"][:, None]
        half = walk["half_spread"][:, None]
        depth_l1 = walk[depth_key][:, None]

        # `participating` is 1 where the snapshot's displayed depth reaches x.
        # Averaged inside a bin it is the fraction of the session whose book was
        # deep enough, which is the truncation the module docstring warns about.
        participating = np.isfinite(cost).astype(float)

        values = {
            "cost_bp": cost / mid * 1e4,
            "cost_sigma": cost / mid / sigma_d,
            "cost_half_spreads": cost / half,
            "participating_num": participating,
        }
        sizes = {"shares": grid,
                 "x_over_adv": grid / adv,
                 "x_over_depth": grid / np.where(depth_l1 > 0, depth_l1, np.nan)}

        for norm, size in sizes.items():
            # participation must count NON-participating snapshots too, so it is
            # binned on its own with the finite-cost mask lifted
            flat = {k: v.ravel() for k, v in values.items() if k != "participating_num"}
            table = bw.bin_walk(size.ravel(), flat, n_bins=n_bins)
            part = _participation_by_bin(size.ravel(), participating.ravel(),
                                         table["size"].to_numpy())
            table.insert(0, "side", side)
            table.insert(1, "size_normalisation", norm)
            table["participating"] = part
            out.append(table)
    scales = {
        "n_snapshots": n_snap,
        "mid_median": float(np.median(walk["mid"])),
        "half_spread_median": float(np.median(walk["half_spread"])),
        "depth_l1_ask_median": float(np.median(walk["depth_l1_ask"])),
        "depth_l1_bid_median": float(np.median(walk["depth_l1_bid"])),
        "depth_10_ask_median": float(np.median(walk["depth_10_ask"])),
        "depth_10_bid_median": float(np.median(walk["depth_10_bid"])),
        "one_tick_bp": float(0.01 / np.median(walk["mid"]) * 1e4),
        "share_one_tick_spread": float(np.mean(
            np.isclose(2 * walk["half_spread"], 0.01, atol=1e-9))),
    }
    return pd.concat(out, ignore_index=True), scales


def _participation_by_bin(size: np.ndarray, participating: np.ndarray,
                          bin_means: np.ndarray) -> np.ndarray:
    """Fraction of snapshots deep enough, at each bin's mean size.

    Assigned by nearest bin centre in log size rather than re-deriving the
    quantile edges, which would need the finite-cost mask that is exactly what
    this column exists to measure without.
    """
    ok = np.isfinite(size) & (size > 0)
    idx = np.abs(np.log(size[ok])[:, None] - np.log(bin_means)[None, :]).argmin(axis=1)
    frame = pd.DataFrame({"bin": idx, "p": participating[ok]})
    means = frame.groupby("bin").p.mean()
    return means.reindex(range(len(bin_means))).to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", action="append", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/bookwalk"))
    ap.add_argument("--daily", type=Path, default=Path("data/daily_reference.csv"))
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()

    daily = pd.read_csv(args.daily).set_index(["symbol", "date"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = raw_root()

    recipes, scales_rows = [], []
    for sess in sessions():
        if args.symbol and sess.symbol not in set(args.symbol):
            continue
        path = sess.raw_path("mbp-10", root)
        if not path.exists():
            print(f"  {sess.key:<16} MISSING {path.name}", file=sys.stderr)
            continue
        book = load_snapshots(path, sess.date)
        ref = daily.loc[(sess.symbol, sess.date)]
        table, scales = session_bins(book, float(ref.adv_20d_xnas),
                                     float(ref.sigma_daily_20d), n_bins=args.bins)
        table.insert(0, "session", sess.key)
        table.to_csv(args.out_dir / f"{sess.key}_bins.csv", index=False)

        recipe = bw.original_recipe(book, depth=DEPTH)
        recipe["session"] = sess.key
        recipes.append(recipe)
        scales.update({"session": sess.key, "symbol": sess.symbol, "date": sess.date,
                       "adv_20d_xnas": float(ref.adv_20d_xnas),
                       "sigma_daily_20d": float(ref.sigma_daily_20d)})
        scales_rows.append(scales)
        print(f"  {sess.key:<16} {scales['n_snapshots']:>6,} snapshots  "
              f"p_filtered={recipe['p_original_filtered']:.3f}  "
              f"p_unfiltered={recipe['p_unfiltered']:.3f}")

    if recipes:
        pd.DataFrame(recipes).to_csv(args.out_dir / "original_recipe.csv", index=False)
        pd.DataFrame(scales_rows).to_csv(args.out_dir / "session_scales.csv", index=False)
        print(f"\nsaved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
