#!/usr/bin/env python3
"""Rebuild compact normaliser comparisons from committed derived tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(report_dir: Path) -> pd.DataFrame:
    pooled = pd.read_csv(report_dir / "normaliser_comparison_pooled.csv").set_index(
        "normaliser"
    )
    by_symbol = pd.read_csv(
        report_dir / "normaliser_comparison_by_symbol.csv"
    )
    base_name = "venue_volume_venue_sigma"
    comparisons = (
        ("volume_only", "consolidated_volume_venue_sigma"),
        ("volatility_only", "venue_volume_consolidated_sigma"),
        ("volume_and_volatility", "consolidated_volume_consolidated_sigma"),
    )
    base = pooled.loc[base_name]
    rows = []
    base_symbols = by_symbol[by_symbol["normaliser"] == base_name].set_index("symbol")
    for channel, name in comparisons:
        changed = pooled.loc[name]
        changed_symbols = by_symbol[by_symbol["normaliser"] == name].set_index(
            "symbol"
        )
        rows.append(
            {
                "comparison": channel,
                "baseline": base_name,
                "changed": name,
                "n_metaorders": int(changed["n_metaorders"]),
                "baseline_delta": base["delta"],
                "changed_delta": changed["delta"],
                "delta_change": changed["delta"] - base["delta"],
                "baseline_c_half": base["c_half"],
                "changed_c_half": changed["c_half"],
                "c_half_change": changed["c_half"] - base["c_half"],
                "median_symbol_delta_change": (
                    changed_symbols["delta"] - base_symbols["delta"]
                ).median(),
                "median_symbol_c_half_change": (
                    changed_symbols["c_half"] - base_symbols["c_half"]
                ).median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/cross_section"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build(args.report_dir).to_csv(index=False).encode()
    target = args.report_dir / "normaliser_comparison_summary.csv"
    volumes = pd.read_csv(args.report_dir / "venue_vs_consolidated_volume.csv")
    volume_summary = pd.DataFrame(
        [
            {
                "n_symbols": len(volumes),
                "median_of_symbol_median_crsp_over_venue": volumes[
                    "median_ratio"
                ].median(),
                "p05_of_symbol_median_crsp_over_venue": volumes[
                    "median_ratio"
                ].quantile(0.05),
                "p95_of_symbol_median_crsp_over_venue": volumes[
                    "median_ratio"
                ].quantile(0.95),
                "minimum_symbol_median_crsp_over_venue": volumes[
                    "median_ratio"
                ].min(),
                "maximum_symbol_median_crsp_over_venue": volumes[
                    "median_ratio"
                ].max(),
            }
        ]
    ).to_csv(index=False).encode()
    volume_target = args.report_dir / "volume_source_summary.csv"
    if args.check:
        if (
            not target.exists()
            or target.read_bytes() != expected
            or not volume_target.exists()
            or volume_target.read_bytes() != volume_summary
        ):
            raise SystemExit(f"{target} is stale")
    else:
        target.write_bytes(expected)
        volume_target.write_bytes(volume_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
