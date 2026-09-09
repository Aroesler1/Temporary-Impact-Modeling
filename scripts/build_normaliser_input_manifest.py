#!/usr/bin/env python3
"""Record hashes and coverage for the external proxy-order inputs."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=os.environ.get("IMPACT_METAORDER_DIR"),
        required=os.environ.get("IMPACT_METAORDER_DIR") is None,
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/cross_section/sample.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cross_section/normaliser_input_manifest.csv"),
    )
    args = parser.parse_args()
    sample = pd.read_csv(args.sample)
    sample = sample[sample["role"] == "stratified"]
    rows = []
    for symbol in sample["symbol"].astype(str):
        path = args.meta_dir / f"{symbol}.csv"
        frame = pd.read_csv(path, usecols=["date"])
        rows.append(
            {
                "symbol": symbol,
                "input_file": path.name,
                "sha256": sha256(path),
                "n_metaorders": len(frame),
                "n_sessions": frame["date"].nunique(),
                "date_min": frame["date"].min(),
                "date_max": frame["date"].max(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
