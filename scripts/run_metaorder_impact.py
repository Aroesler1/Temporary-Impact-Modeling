#!/usr/bin/env python3
"""Fit the impact law on reconstructed metaorders.

Session volume and daily volatility come from `data/session_meta.csv`, the
same scales every other run script uses, keyed by the session in the metaorder
file's name.

Usage:
    python scripts/run_metaorder_impact.py
    python scripts/run_metaorder_impact.py --data data/AAPL_2024-06-03_metaorders.csv
"""
from __future__ import annotations

import argparse, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import panel  # noqa: E402
from metaorder_impact import fit_impact_law, load  # noqa: E402

_SUFFIX = "_metaorders"


def session_key(data: Path) -> str:
    """`data/MSFT_2024-06-03_metaorders.csv` -> `MSFT_2024-06-03`."""
    stem = Path(data).stem
    if not stem.endswith(_SUFFIX):
        raise ValueError(f"{data}: expected a <SYMBOL>_<DATE>{_SUFFIX}.csv file")
    return stem[: -len(_SUFFIX)]


def session_scales(data: Path) -> tuple[float, float]:
    """Deduplicated session volume and trailing 20-day daily volatility.

    Read from the session file rather than hardcoded: an earlier version
    carried a rounded sigma (0.0106) that did not match the 0.01038 every
    other script used for MSFT 2024-06-03.
    """
    scales = panel.scales(session_key(data))
    return float(scales.session_volume), float(scales.sigma_daily_20d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/MSFT_2024-06-03_metaorders.csv"))
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/metaorder"))
    args = ap.parse_args()

    frame = load(args.data)
    session_volume, daily_vol = session_scales(args.data)
    fit = fit_impact_law(frame, session_volume, daily_vol, n_bins=args.bins)

    print(f"{args.data.name}: {len(frame):,} reconstructed metaorders")
    print(f"session volume {session_volume:,.0f}   daily vol {daily_vol:.6f}\n")
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
