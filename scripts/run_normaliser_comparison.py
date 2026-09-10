#!/usr/bin/env python3
"""Compare venue and consolidated normalisers on identical proxy orders.

Phase A changes only volume and holds venue volatility fixed. Phase B changes
only volatility and holds venue volume fixed. A fourth row reports both
substitutions together. No missing consolidated observation is replaced.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crsp  # noqa: E402
from normaliser_comparison import fit_variants, prepare_variants  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=os.environ.get("IMPACT_METAORDER_DIR"),
        required=os.environ.get("IMPACT_METAORDER_DIR") is None,
    )
    parser.add_argument(
        "--crsp-cache",
        type=Path,
        default=os.environ.get("IMPACT_CRSP_CACHE_DIR"),
        required=os.environ.get("IMPACT_CRSP_CACHE_DIR") is None,
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/cross_section/sample.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/cross_section"),
    )
    parser.add_argument("--boot", type=int, default=0)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sample = sample[sample["role"] == "stratified"].copy()
    frames = []
    missing_files = []
    for symbol in sample["symbol"].astype(str):
        path = args.meta_dir / f"{symbol}.csv"
        if not path.exists():
            missing_files.append(symbol)
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            missing_files.append(symbol)
            continue
        frames.append(frame)
    if missing_files:
        raise FileNotFoundError(
            f"Missing proxy-order files for {len(missing_files)} symbols: "
            f"{missing_files[:10]}"
        )
    orders = pd.concat(frames, ignore_index=True)
    required_pairs = orders[["symbol", "date"]].drop_duplicates()
    consolidated = crsp.load_consolidated_cache(
        sample["symbol"],
        str(pd.to_datetime(orders["date"]).min().date()),
        str(pd.to_datetime(orders["date"]).max().date()),
        required_pairs=required_pairs,
        cache_dir=args.crsp_cache,
    )
    variants = prepare_variants(orders, consolidated)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled = fit_variants(variants, n_boot=args.boot, seed=20240628)
    pooled.insert(0, "sample", "all_proxy_orders")
    pooled.to_csv(
        args.out_dir / "normaliser_comparison_pooled.csv",
        index=False,
    )

    per_symbol = []
    baseline = variants["venue_volume_venue_sigma"]
    for symbol in sample["symbol"].astype(str):
        indices = baseline.index[baseline["symbol"] == symbol]
        subset = {name: frame.loc[indices] for name, frame in variants.items()}
        fitted = fit_variants(subset, n_boot=0, seed=20240628)
        fitted.insert(0, "symbol", symbol)
        per_symbol.append(fitted)
    by_symbol = pd.concat(per_symbol, ignore_index=True)
    by_symbol.to_csv(
        args.out_dir / "normaliser_comparison_by_symbol.csv",
        index=False,
    )

    daily = (
        baseline.groupby(["symbol", "date"], as_index=False)
        .agg(
            volume_venue=("volume_venue", "first"),
            volume_crsp=("volume_crsp", "first"),
            sigma_venue=("sigma_d", "first"),
            sigma_crsp=("sigma_crsp", "first"),
            n_proxy_orders=("Q", "size"),
        )
    )
    daily["crsp_over_venue_volume"] = (
        daily["volume_crsp"] / daily["volume_venue"]
    )
    volume_summary = (
        daily.groupby("symbol")["crsp_over_venue_volume"]
        .agg(
            n_sessions="size",
            median_ratio="median",
            p05_ratio=lambda values: values.quantile(0.05),
            p95_ratio=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )
    volume_summary.to_csv(
        args.out_dir / "venue_vs_consolidated_volume.csv",
        index=False,
    )

    coverage = pd.DataFrame(
        [
            {
                "n_symbols": orders["symbol"].nunique(),
                "n_proxy_orders": len(orders),
                "n_symbol_dates": len(required_pairs),
                "n_consolidated_rows_loaded": len(consolidated),
                "missing_proxy_orders": 0,
                "missing_symbol_dates": 0,
                "crsp_volume_unit": consolidated.attrs["volume_unit"],
                "crsp_daily_source": consolidated.attrs["daily_source"],
                "identifier_source": consolidated.attrs["name_source"],
                "equs_validation_median_ratio": consolidated.attrs[
                    "equs_validation_ratio"
                ],
            }
        ]
    )
    coverage.to_csv(args.out_dir / "normaliser_coverage.csv", index=False)
    print(pooled.to_string(index=False))
    print(coverage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
