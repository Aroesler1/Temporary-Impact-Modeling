"""Optimal execution under a fitted kernel, checked against closed forms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import execution as ex


def test_kernel_matrix_is_toeplitz_and_banded():
    kernel = np.array([1.0, 0.5, 0.25])
    km = ex.kernel_matrix(kernel, T=6)
    assert km.M[0, 0] == pytest.approx(km.M[3, 3])
    assert km.M[0, 1] == pytest.approx(0.5)
    assert km.M[0, 2] == pytest.approx(0.25)
    assert km.M[0, 3] == pytest.approx(0.0)      # beyond the kernel's reach
    np.testing.assert_allclose(km.M, km.M.T)


def test_kernel_matrix_reports_and_repairs_an_indefinite_fit():
    """A kernel with a large negative lag makes M indefinite, which means the
    model admits a profitable round trip. That has to be visible."""
    km = ex.kernel_matrix(np.array([1.0, -0.9]), T=8)
    assert km.min_eigenvalue < 0
    assert km.projected
    assert np.linalg.eigvalsh(km.M).min() >= 0


def test_memoryless_kernel_makes_twap_optimal():
    """G(l) = 0 for l > 0 gives M proportional to the identity, whose
    minimum-cost schedule under a sum constraint is the flat one."""
    km = ex.kernel_matrix(np.array([1.0]), T=25)
    x = ex.optimal_schedule(km, S=1000.0)
    np.testing.assert_allclose(x, ex.twap(25, 1000.0), rtol=1e-9)


def test_optimal_schedule_meets_the_size_constraint():
    km = ex.kernel_matrix(np.array([1.0, 0.6, 0.3, 0.1]), T=40)
    x = ex.optimal_schedule(km, S=5000.0)
    assert x.sum() == pytest.approx(5000.0)


def test_optimal_schedule_beats_twap_under_its_own_cost_function():
    """The whole claim of the GSS solution: no other schedule has lower model
    cost. Checked against TWAP and against a thousand random perturbations."""
    kernel = np.exp(-0.3 * np.arange(6))          # Obizhaeva-Wang decay
    km = ex.kernel_matrix(kernel, T=30)
    S = 100.0
    x = ex.optimal_schedule(km, S)

    def cost(v):
        return 0.5 * float(v @ km.M @ v)

    assert cost(x) <= cost(ex.twap(30, S)) + 1e-12
    rng = np.random.default_rng(0)
    for _ in range(1000):
        step = rng.standard_normal(30)
        step -= step.mean()                       # stay on the constraint
        assert cost(x) <= cost(x + 0.5 * step) + 1e-12


def test_obizhaeva_wang_shape_appears_for_an_exponential_kernel():
    """The published solution for an exponentially decaying kernel is a block
    at each end and a constant rate between."""
    km = ex.kernel_matrix(np.exp(-0.5 * np.arange(10)), T=50)
    x = ex.optimal_schedule(km, 1.0)
    middle = x[5:-5]
    assert x[0] > middle.max() * 1.05
    assert x[-1] > middle.max() * 1.05
    assert middle.std() / middle.mean() < 0.02


def test_almgren_chriss_front_loads_and_reduces_inventory():
    S, T = 1000.0, 60
    flat = ex.almgren_chriss(T, S, kappa=0.0)
    urgent = ex.almgren_chriss(T, S, kappa=0.06)
    np.testing.assert_allclose(flat, ex.twap(T, S))
    assert urgent.sum() == pytest.approx(S)
    assert urgent[0] > urgent[-1]
    inv = lambda v: float(np.sum((S - np.cumsum(v)) ** 2))
    assert inv(urgent) < inv(flat)


def test_replay_cost_splits_drift_from_impact():
    kernel = np.array([1e-6])
    mid = np.full(50, 100.0)
    x = ex.twap(50, 1000.0)
    out = ex.replay_cost(x, mid, kernel)
    assert out["drift"] == pytest.approx(0.0, abs=1e-12)
    # each second trades 20 shares, so own displacement at second t is
    # 1e-6 * 20 in log terms and the average paid price is above the mid
    assert out["impact"] > 0
    assert out["total"] == pytest.approx(out["drift"] + out["impact"])


def test_replay_cost_charges_drift_to_a_flat_schedule_only_once():
    mid = np.linspace(100.0, 101.0, 51)[:50]
    out = ex.replay_cost(ex.twap(50, 100.0), mid, np.array([0.0]))
    assert out["impact"] == pytest.approx(0.0)
    assert out["drift"] == pytest.approx(mid.mean() - mid[0])


def test_replay_cost_penalises_concentrating_into_one_second():
    kernel = np.array([1e-5])
    mid = np.full(40, 100.0)
    spread_out = ex.replay_cost(ex.twap(40, 4000.0), mid, kernel)["impact"]
    concentrated = np.zeros(40)
    concentrated[0] = 4000.0
    lumped = ex.replay_cost(concentrated, mid, kernel)["impact"]
    assert lumped > spread_out


def test_select_lags_and_fit_linear_kernel_recover_a_known_kernel():
    rng = np.random.default_rng(3)
    n = 5000
    vol = rng.standard_normal(n) * 500.0
    true = np.array([2e-6, 8e-7])
    ret = np.convolve(vol, true)[:n] + 1e-9 * rng.standard_normal(n)
    train_end = int(n * 0.7)
    n_lags = ex.select_lags(ret, vol, train_end, lag_grid=(1, 2, 5))
    kernel = ex.fit_linear_kernel(ret, vol, n_lags, train_end)
    np.testing.assert_allclose(kernel[:2], true, rtol=0.02)


def test_fit_linear_kernel_ignores_rows_past_the_training_end():
    n = 4000
    rng = np.random.default_rng(4)
    vol = rng.standard_normal(n) * 100.0
    ret = 5e-6 * vol
    train_end = int(n * 0.7)
    ret[train_end:] = 1.0                       # nonsense in the held-out tail
    kernel = ex.fit_linear_kernel(ret, vol, 1, train_end)
    assert kernel[0] == pytest.approx(5e-6, rel=1e-6)


def test_bootstrap_saving_brackets_the_mean():
    results = []
    rng = np.random.default_rng(5)
    for s in range(8):
        for _ in range(20):
            twap_cost = 0.01
            other = 0.01 - 0.002 + 0.0005 * rng.standard_normal()
            results.append(ex.ReplayResult(
                f"S{s}", 0, 0.01,
                {"TWAP": {"total": twap_cost, "impact": twap_cost, "drift": 0.0},
                 "other": {"total": other, "impact": other, "drift": 0.0}},
                {"TWAP": 1.0, "other": 1.0}))
    mean, lo, hi = ex.bootstrap_saving(results, "other", n_boot=500)
    assert mean == pytest.approx(0.002, abs=2e-4)
    assert lo < mean < hi
