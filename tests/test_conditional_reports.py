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

    assert method["report_version"] == "outcome-end-v3"
    assert int(method["n_sessions"]) == len(summary) == 15
    assert int(method["n_crossing_orders_excluded"]) == 8
    assert int(summary["n_crossing_orders_excluded"].sum()) == 8

    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        assert path.stat().st_size == row.bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256


def test_decile_ties_are_stable_under_roundoff_and_input_permutation():
    from conditional_impact import calibration_deciles
    predicted = np.repeat(np.arange(1., 11.), 4) * 1e-5
    perturbed = predicted * (1 + np.tile([-1, 1, -1, 1], 10) * 1e-14)
    expected = np.repeat(np.arange(10), 4)
    np.testing.assert_array_equal(calibration_deciles(predicted), expected)
    np.testing.assert_array_equal(calibration_deciles(perturbed), expected)
    order = np.random.default_rng(9).permutation(len(predicted))
    np.testing.assert_array_equal(calibration_deciles(perturbed[order]), expected[order])
    assert len(np.unique(calibration_deciles(np.ones(20)))) == 1


def test_deciles_keep_mean_predictions_unrounded():
    from conditional_impact import calibration_table
    predicted = np.array([1.00000000000001, 1.00000000000002])
    table = calibration_table(predicted * 2, predicted)
    assert table.n.sum() == 2
    assert len(table) == 1
    assert table.predicted.iloc[0] == predicted.mean()


@pytest.mark.parametrize('damage', ['count', 'float', 'hash', 'missing'])
def test_artifact_check_rejects_material_changes(tmp_path, damage):
    from scripts.run_conditional_impact import ARTIFACTS, verify_artifacts
    stored, rebuilt = tmp_path / 'stored', tmp_path / 'rebuilt'
    stored.mkdir(); rebuilt.mkdir()
    for name in ARTIFACTS:
        frame = pd.DataFrame({'n': [1000000], 'r2': [.2], 'sha256': ['fixture']})
        frame.to_csv(stored / name, index=False)
        if name == ARTIFACTS[0]:
            if damage == 'count': frame.loc[0, 'n'] += 1
            if damage == 'float': frame.loc[0, 'r2'] += 1e-4
            if damage == 'hash': frame.loc[0, 'sha256'] = 'changed'
            if damage == 'missing': frame.loc[0, 'r2'] = np.nan
        frame.to_csv(rebuilt / name, index=False)
    with pytest.raises(AssertionError):
        verify_artifacts(stored, rebuilt)


def test_artifact_check_accepts_only_negligible_float_roundoff(tmp_path):
    from scripts.run_conditional_impact import ARTIFACTS, verify_artifacts
    stored, rebuilt = tmp_path / 'stored', tmp_path / 'rebuilt'
    stored.mkdir(); rebuilt.mkdir()
    for name in ARTIFACTS:
        pd.DataFrame({'n': [100], 'r2': [.2]}).to_csv(stored / name, index=False)
        pd.DataFrame({'n': [100], 'r2': [.2 + 1e-12]}).to_csv(rebuilt / name, index=False)
    verify_artifacts(stored, rebuilt)
