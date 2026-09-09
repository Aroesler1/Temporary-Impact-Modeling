"""Offline arithmetic and provenance checks for repaired conditional reports."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conditional_impact import MODEL_ORDER
from scripts.run_conditional_impact import band

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports" / "conditional_impact_corrected"


def test_corrected_model_comparison_recomputes_from_session_reports():
    summary = pd.read_csv(REPORTS / "summary.csv")
    pooled = pd.read_csv(REPORTS / "calibration_pooled.csv")
    reported = pd.read_csv(REPORTS / "model_comparison.csv").set_index("model")

    for model in MODEL_ORDER:
        if model not in reported.index:
            continue
        row = reported.loc[model]
        r2 = summary[f"{model}_r2"].to_numpy(float)
        slope = summary[f"{model}_slope"].to_numpy(float)
        group = pooled[pooled["model"] == model].sort_values("decile")
        lo, hi = band(r2)

        assert int(row["n_sessions"]) == int(np.isfinite(r2).sum())
        assert row["median_r2"] == pytest.approx(float(np.nanmedian(r2)))
        assert row["median_slope"] == pytest.approx(float(np.nanmedian(slope)))
        assert row["mean_r2_band"] == f"[{lo:.3f}, {hi:.3f}]"
        assert row["top_decile_ratio"] == pytest.approx(float(group["ratio"].iloc[-1]))
        assert row["decile_1_to_8_ratio"] == pytest.approx(
            float(group["ratio"].iloc[:-1].abs().mean())
        )


def test_corrected_split_and_input_manifest_are_self_checking():
    summary = pd.read_csv(REPORTS / "summary.csv")
    method = pd.read_csv(REPORTS / "methodology.csv").iloc[0]
    manifest = pd.read_csv(REPORTS / "input_manifest.csv")

    assert method["report_version"] == "outcome-end-v2"
    assert int(method["n_sessions"]) == len(summary) == 15
    assert int(method["n_crossing_orders_excluded"]) == 8
    assert int(summary["n_crossing_orders_excluded"].sum()) == 8

    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        assert path.stat().st_size == row.bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256
