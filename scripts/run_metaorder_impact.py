#!/usr/bin/env python3
"""Fit the impact law on reconstructed metaorders.

Usage:
    python scripts/run_metaorder_impact.py
"""
from __future__ import annotations

import argparse, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metaorder_impact import fit_impact_law, load  # noqa: E402

# MSFT 2024-06-03, from the Databento MBO session used elsewhere in this repo.
SESSION_VOLUME = 5_576_188   # deduplicated: see the T/F double-count note
DAILY_VOL = 0.0106


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/MSFT_2024-06-03_metaorders.csv"))
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/metaorder"))
    args = ap.parse_args()

    frame = load(args.data)
    fit = fit_impact_law(frame, SESSION_VOLUME, DAILY_VOL, n_bins=args.bins)

    print(f"{args.data.name}: {len(frame):,} reconstructed metaorders")
    print(f"session volume {SESSION_VOLUME:,}   daily vol {DAILY_VOL:.4f}\n")
    print(fit.bins.to_string(index=False, float_format=lambda v: f"{v:0.6f}"))
    print()
    print(fit.describe())
    print(f"  fitted on {fit.n_bins} bin means; the square-root law predicts 0.5")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit.bins.to_csv(args.out_dir / "impact_bins.csv", index=False)
    print(f"\nsaved -> {args.out_dir / 'impact_bins.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
