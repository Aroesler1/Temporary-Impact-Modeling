"""Tests for the transient-impact propagator.

These pin the properties that make the calibration meaningful rather than
merely runnable: that a known kernel is recovered, that the predictive variant
really excludes contemporaneous information, and that scoring is out-of-sample.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propagator import (  # noqa: E402
    build_lag_matrix,
    calibrate,
    fit_propagator,
    signed_flow,
)


def test_signed_flow_preserves_sign_and_compresses():
    v = np.array([-100.0, 0.0, 100.0])
    out = signed_flow(v, 0.5)
    assert out[0] == -10.0 and out[1] == 0.0 and out[2] == 10.0
    # delta = 1 is the identity
    assert np.allclose(signed_flow(v, 1.0), v)


def test_lag_matrix_is_strictly_causal():
    flow = np.arange(5.0)
    design = build_lag_matrix(flow, 2)
    # column 0 is contemporaneous, column l is flow shifted forward by l
    assert np.allclose(design[:, 0], flow)
    assert np.isnan(design[0, 1])
    assert design[2, 1] == flow[1]
    assert design[2, 2] == flow[0]


def test_recovers_a_known_kernel():
    """Synthetic data built from a known G must be recovered by OLS."""
    rng = np.random.default_rng(0)
    n = 6000
    vol = rng.normal(0, 500, n)
    true_kernel = np.array([3e-5, 1e-5, 4e-6])
    flow = signed_flow(vol, 1.0)
    returns = np.zeros(n)
    for lag, g in enumerate(true_kernel):
        returns[lag:] += g * flow[: n - lag]
    returns += rng.normal(0, 1e-6, n)

    fit = fit_propagator(returns, vol, n_lags=2, delta=1.0)
    assert np.allclose(fit.kernel, true_kernel, rtol=0.05)
    assert fit.r2_out > 0.95


def test_predictive_mode_excludes_contemporaneous_flow():
    """With impact ONLY contemporaneous, the predictive fit must learn nothing.

    This is the guard that matters: if the l=0 column leaked into the
    predictive design, this test would show a high R^2 and the headline
    explanatory-vs-predictive contrast would be an artefact.
    """
    rng = np.random.default_rng(1)
    n = 6000
    vol = rng.normal(0, 500, n)
    returns = 3e-5 * vol + rng.normal(0, 1e-6, n)  # purely contemporaneous

    explanatory = fit_propagator(returns, vol, n_lags=5, delta=1.0)
    predictive = fit_propagator(returns, vol, n_lags=5, delta=1.0,
                                drop_contemporaneous=True)
    assert explanatory.r2_out > 0.9
    assert predictive.r2_out < 0.05


def test_predictive_mode_requires_a_lag():
    rng = np.random.default_rng(2)
    vol = rng.normal(0, 100, 500)
    with pytest.raises(ValueError):
        fit_propagator(vol * 1e-5, vol, n_lags=0, delta=1.0, drop_contemporaneous=True)


def test_scoring_is_out_of_sample():
    """R^2_out must come from data the kernel never saw."""
    rng = np.random.default_rng(3)
    n = 4000
    vol = rng.normal(0, 500, n)
    returns = rng.normal(0, 1e-5, n)  # pure noise: no relationship to recover
    fit = fit_propagator(returns, vol, n_lags=10, delta=1.0, train_frac=0.7)
    assert fit.n_train + fit.n_test <= n
    # fitting noise gives positive in-sample R^2 and non-positive out-of-sample
    assert fit.r2_in > 0
    assert fit.r2_out < 0.02


def test_calibrate_selects_on_out_of_sample(monkeypatch):
    import pandas as pd

    rng = np.random.default_rng(4)
    n = 4000
    vol = rng.normal(0, 400, n)
    flow = signed_flow(vol, 1.0)
    mid = 100 * np.exp(np.cumsum(2e-5 * flow + rng.normal(0, 1e-6, n)))
    frame = pd.DataFrame({"mid": mid, "signed_vol": vol})

    report = calibrate(frame, lag_grid=(0, 2), delta_grid=(0.5, 1.0))
    assert not report.grid.empty
    # grid is ordered by out-of-sample score
    assert report.grid["r2_out"].is_monotonic_decreasing
    assert report.best.r2_out == report.grid.iloc[0]["r2_out"]
