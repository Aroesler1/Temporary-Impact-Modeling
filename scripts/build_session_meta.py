#!/usr/bin/env python3
"""Collect the per-session scale constants every part of the study normalises by.

Nothing here is a new measurement: it joins the daily reference (trailing ADV
and volatility), the book scales from the MBP-10 pass, and the traded volume
implied by the reconstructed metaorders, so that no run script has to hardcode
a session volume the way `run_metaorder_impact.py` used to.

`displayed_volume` is the sum of displayed fills inside RTH, which is what the
metaorder reconstruction sees and therefore the right denominator for a
participation rate built from it. It is NOT the consolidated tape and it is not
even the whole Nasdaq session, since hidden prints are excluded; the column name
says which.

Usage:
    python scripts/build_session_meta.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sessions import sessions  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    args = ap.parse_args()

    daily = pd.read_csv(args.data_dir / "daily_reference.csv").set_index(["symbol", "date"])
    book = pd.read_csv(args.data_dir / "bookwalk" / "session_scales.csv").set_index("session")
    volume = pd.read_csv(args.data_dir / "session_volume.csv").set_index("session")

    rows = []
    for sess in sessions():
        bars = pd.read_csv(args.data_dir / f"{sess.key}_1s.csv")
        metas = pd.read_csv(args.data_dir / f"{sess.key}_metaorders.csv")
        ref = daily.loc[(sess.symbol, sess.date)]
        scales = book.loc[sess.key]
        vol = volume.loc[sess.key]
        displayed = float(metas.shares.sum())
        if not np.isclose(displayed, float(vol.displayed_volume_mbo)):
            raise SystemExit(
                f"{sess.key}: metaorder shares {displayed:,.0f} disagree with the "
                f"MBO displayed tally {vol.displayed_volume_mbo:,.0f}; the two "
                f"builders are not reading the same fills")
        rows.append({
            "session": sess.key, "symbol": sess.symbol, "date": sess.date,
            "n_bars": int(len(bars)),
            "n_metaorders": int(len(metas)),
            "displayed_volume": displayed,
            "hidden_volume": float(vol.hidden_volume_mbo),
            # participation denominator: deduplicated traded volume over the same
            # 04:00-20:00 window the metaorder reconstruction covers
            "session_volume": float(vol.total_volume_mbo),
            "adv_20d_xnas": float(ref.adv_20d_xnas),
            "sigma_daily_20d": float(ref.sigma_daily_20d),
            "mid_median": float(scales.mid_median),
            "half_spread_median": float(scales.half_spread_median),
            "one_tick_bp": float(scales.one_tick_bp),
            "share_one_tick_spread": float(scales.share_one_tick_spread),
            "depth_l1_ask_median": float(scales.depth_l1_ask_median),
            "depth_l1_bid_median": float(scales.depth_l1_bid_median),
            "depth_10_ask_median": float(scales.depth_10_ask_median),
            # realised session volatility of the one-second mid, annualised to a
            # day, kept beside the 20-day close-to-close figure because the two
            # disagree and every impact number is quoted in one of them
            "sigma_realised_session": float(
                np.nanstd(np.diff(np.log(bars.mid.to_numpy(float))))
                * np.sqrt(len(bars))),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.data_dir / "session_meta.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
    print(f"\nsaved -> {args.data_dir / 'session_meta.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
