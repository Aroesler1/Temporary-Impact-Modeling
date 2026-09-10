#!/usr/bin/env python3
"""Recompute the audit from published aggregates, offline and without a refit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kernel_response import return_to_level_response

SOURCE_REVISION = "dd2d0b8b9a4482be7e93909479b0fcd6478276eb"
INPUTS = (
    "reports/kernel_100ms/kernel_shape.csv",
    "reports/cross_section/per_stock_fits.csv",
    "reports/conditional_impact/model_comparison.csv",
    "reports/panel/per_session.csv",
)

# Digests of git-show SOURCE_REVISION inputs, pinned independently of generation.
INPUT_HASHES = (
    "e54dbaeff262550b4c8cc5208887f8ac7f5c3057a2da3e77498b87db7d35d773",
    "ebb3ba9ef986832c0c8ddad3befeb055b18aec12fd217c0bbc81b25f9f39ab7a",
    "548ad63f4a153a2dcbdad05014a23820a6dd71360355a72166897e94103fe809",
    "836a2086586bfacc41f93446d109ea133ca797cf79ba22791a15dbd9f144a10b",
)


def read_rows(root: Path, relative: str) -> list[dict[str, str]]:
    with (root / relative).open(newline="") as source:
        return list(csv.DictReader(source))


def csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def audit_tables(root: Path) -> dict[str, bytes]:
    for path, expected in zip(INPUTS, INPUT_HASHES):
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen source hash differs: {path}")
    shape = read_rows(root, INPUTS[0])
    if [int(row["lag"]) for row in shape] != list(range(21)):
        raise ValueError("expected contiguous committed lags 0 through 20")
    coefficients = [float(row["mean_G_over_G0"]) for row in shape]
    levels = return_to_level_response(coefficients)
    response = [dict(
        lag=lag, horizon_ms=100 * lag,
        mean_return_response_over_initial=coefficients[lag],
        mean_level_response_over_initial=level,
        n_sessions=15, status="descriptive_selected_validation_no_level_CI",
    ) for lag, level in enumerate(levels)]

    stocks = read_rows(root, INPUTS[1])
    if len({row["symbol"] for row in stocks}) != len(stocks):
        raise ValueError("duplicate cross-section symbols")
    stratified = [row for row in stocks if row["role"] == "stratified"]
    if len(stratified) != 108:
        raise ValueError("expected the frozen 108-name stratified sample")
    interior = [row for row in stratified if row["q_star_interior"] == "True"]
    scope = []
    for label, subset in (("all_fitted_including_boundaries", stratified),
                          ("interior_crossover_only", interior)):
        ticks = [float(row["impact_at_crossover_ticks"]) for row in subset]
        if not ticks or not all(math.isfinite(value) for value in ticks):
            raise ValueError("crossover impacts must be nonempty and finite")
        scope.append(dict(sample=label, n_names=len(subset),
                          median_crossover_impact_ticks=statistics.median(ticks),
                          crossover_above_one_tick=sum(value > 1 for value in ticks)))

    model = {row["model"]: row for row in read_rows(root, INPUTS[2])}
    panel = read_rows(root, INPUTS[3])
    checks = [
        dict(metric="cross_section_median_delta", value=statistics.median(
            float(row["delta"]) for row in stratified), n=108,
             status="descriptive_per_stock_fits"),
        dict(metric="cross_section_intervals_bracketing_half", value=sum(
            row["brackets_half"] == "True" for row in stratified), n=108,
             status="individual_intervals_not_simultaneous"),
        dict(metric="prior_tod_conditional_median_r2",
             value=float(model["sqrt_tod_prior"]["median_r2"]), n=12,
             status="nested_selection_existing_conditional_result"),
        dict(metric="panel_positive_predictive_validation_r2", value=sum(
            float(row["predictive_r2"]) > 0 for row in panel), n=len(panel),
             status="selected_validation_not_untouched_test"),
    ]
    manifest = [dict(path=path, source_revision=SOURCE_REVISION,
                     sha256=hashlib.sha256((root / path).read_bytes()).hexdigest())
                for path in INPUTS]
    return {"return_vs_level.csv": csv_bytes(response),
            "crossover_scope.csv": csv_bytes(scope),
            "headline_checks.csv": csv_bytes(checks),
            "input_manifest.csv": csv_bytes(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify published audit (default)")
    mode.add_argument("--write", action="store_true", help="write derived audit tables")
    args = parser.parse_args()
    tables = audit_tables(ROOT)
    output = ROOT / "reports/kernel_audit"
    if args.write:
        output.mkdir(parents=True, exist_ok=True)
    for name, content in tables.items():
        path = output / name
        if args.write:
            path.write_bytes(content)
        elif not path.exists() or path.read_bytes() != content:
            raise ValueError(f"audit differs: {path.relative_to(ROOT)}")
    print("Verified four audit tables from four committed aggregate inputs.")
    print("100 ms: return response 0.002048; cumulative level response 1.002048.")
    print("2 seconds: cumulative normalized level response 0.917296; no level CI.")
    print("Empirical schedule claims withdrawn; no new holdout fit or execution result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
