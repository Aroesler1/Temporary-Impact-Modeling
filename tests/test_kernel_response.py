"""Economic impulse controls for the return-versus-level distinction."""
import numpy as np
import pytest

from kernel_response import return_to_level_response


def test_zero_lagged_returns_do_not_mean_price_reversion():
    assert return_to_level_response([1, 0, 0]) == [1, 1, 1]


def test_negative_returns_recover_exponential_level_decay():
    assert return_to_level_response([1, -0.5, -0.25]) == [1, 0.5, 0.25]


def test_convolution_agrees_with_accumulated_returns_within_support():
    flow = np.array([2, -1, 3, 0, -2], dtype=float)
    returns = np.array([1, -0.4, -0.1, 0.2, -0.1])
    expected = np.cumsum(np.convolve(flow, returns)[:len(flow)])
    levels = return_to_level_response(returns)
    np.testing.assert_allclose(np.convolve(flow, levels)[:len(flow)], expected)


def test_response_stops_at_identified_support():
    assert return_to_level_response([2]) == [2]


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")],
                                    [1e308, 1e308]])
def test_invalid_response_is_refused(values):
    with pytest.raises(ValueError, match="finite"):
        return_to_level_response(values)


def test_empirical_schedule_refuses_unaudited_level_costs():
    import execution
    # Fail before accessing data or refitting on a previously examined tail.
    with pytest.raises(RuntimeError, match="return coefficients"):
        execution.replay_session("fixture", None, 1000)


def test_level_half_life_does_not_treat_zero_return_as_reversion():
    from propagator import PropagatorFit
    fit = PropagatorFit(1, 2, np.array([1., 0., 0.]), 0, 0, 10, 5)
    assert np.isnan(fit.decay_half_life)
    fit.kernel = np.array([1., -0.5, -0.25])
    assert fit.decay_half_life == 1


def test_predictive_only_fit_has_no_identified_initial_level():
    from propagator import PropagatorFit
    fit = PropagatorFit(1, 2, np.array([1., -0.8]), 0, 0, 10, 5)
    assert np.isnan(fit.decay_half_life)


def test_published_audit_is_reproducible_without_raw_data():
    from pathlib import Path
    from scripts.audit_kernel_response import audit_tables
    root = Path(__file__).resolve().parents[1]
    for name, content in audit_tables(root).items():
        assert (root / "reports/kernel_audit" / name).read_bytes() == content


def test_altered_source_cannot_be_blessed_by_regenerating_audit(tmp_path):
    from scripts.audit_kernel_response import INPUTS, audit_tables
    path = tmp_path / INPUTS[0]
    path.parent.mkdir(parents=True)
    path.write_text("lag,mean_G_over_G0\n0,0.0\n")
    with pytest.raises(ValueError, match="frozen source hash differs"):
        audit_tables(tmp_path)


def test_schedule_cli_does_not_create_output(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "schedule"
    result = subprocess.run([sys.executable, str(root / "scripts/run_schedule_oos.py"),
                             "--out-dir", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "withdrawn" in result.stderr
    assert not output.exists()
