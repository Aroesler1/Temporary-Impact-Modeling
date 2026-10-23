#!/usr/bin/env python3
"""How much quieter the held-out window is, and whether that tracks the slope.

The plain square-root model holds sigma at a daily constant. Its calibration
slope (realised on predicted, held-out orders) is a median 0.52, and the
explanation offered is that the held-out afternoon is quieter than the morning
the coefficient was fitted on. This measures that directly: one-second return
volatility in the last 30% of each session over the first 70%, beside the
model's own slope from `reports/conditional_impact_corrected/summary.csv`.

Usage:
    python scripts/run_volatility_shift.py
    python scripts/run_volatility_shift.py --check
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import panel  # noqa: E402

SUMMARY = panel.DATA.parent / "reports" / "conditional_impact_corrected" / "summary.csv"
ARTIFACTS = ("per_session.csv", "summary.csv")


def per_session() -> pd.DataFrame:
    slopes = pd.read_csv(SUMMARY).set_index("session")["sqrt_slope"]
    rows = []
    for key in panel.session_keys():
        bars = panel.bars(key)
        returns = panel.returns(bars)
        split = panel.split_index(len(bars))
        train, test = float(np.nanstd(returns[:split])), float(np.nanstd(returns[split:]))
        rows.append({"session": key, "symbol": panel.scales(key).symbol,
                     "sigma_1s_train": train, "sigma_1s_test": test,
                     "test_over_train": test / train,
                     "sqrt_slope": float(slopes.loc[key])})
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "n_sessions": len(frame),
        "median_test_over_train": float(frame.test_over_train.median()),
        "median_sqrt_slope": float(frame.sqrt_slope.median()),
        "pearson_ratio_slope": float(frame.test_over_train.corr(frame.sqrt_slope)),
        # rank correlation, reported beside Pearson because two sessions whose
        # volatility did not fall carry much of the linear correlation
        "spearman_ratio_slope": float(frame.test_over_train.corr(
            frame.sqrt_slope, method="spearman")),
    }])


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = per_session()
    frame.to_csv(out_dir / "per_session.csv", index=False)
    summary = summarise(frame)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/volatility_shift"))
    ap.add_argument("--check", action="store_true",
                    help="rebuild in a temporary directory and compare committed files")
    args = ap.parse_args()
    if not args.check:
        build(args.out_dir)
        return 0
    with tempfile.TemporaryDirectory(prefix="volatility-shift-check-") as raw:
        build(Path(raw))
        for name in ARTIFACTS:
            pd.testing.assert_frame_equal(pd.read_csv(args.out_dir / name),
                                          pd.read_csv(Path(raw) / name),
                                          check_exact=False, rtol=1e-10)
    print(f"Verified {len(ARTIFACTS)} volatility-shift artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
