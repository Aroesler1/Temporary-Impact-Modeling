"""Crossover fit and the published-recipe reconstruction, on known inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import crossover as cx


def test_two_regime_is_continuous_at_the_crossover():
    a, q_star = 2.0, 0.01
    left = cx.two_regime(np.array([q_star - 1e-12]), a, q_star)[0]
    right = cx.two_regime(np.array([q_star + 1e-12]), a, q_star)[0]
    assert left == pytest.approx(right, rel=1e-6)
    assert left == pytest.approx(a * q_star)


def test_two_regime_is_linear_below_and_square_root_above():
    a, q_star = 3.0, 0.02
    assert cx.two_regime(np.array([0.01]), a, q_star)[0] == pytest.approx(0.03)
    # doubling q above the crossover multiplies impact by sqrt(2)
    hi = cx.two_regime(np.array([0.08, 0.16]), a, q_star)
    assert hi[1] / hi[0] == pytest.approx(np.sqrt(2.0))


def test_fit_two_regime_recovers_a_known_crossover():
    a_true, q_true = 4.0, 5e-4
    q = np.exp(np.linspace(np.log(1e-6), np.log(1e-1), 200))
    y = cx.two_regime(q, a_true, q_true)
    fit = cx.fit_two_regime(q, y, np.ones_like(q), n_grid=400)
    assert fit.q_star == pytest.approx(q_true, rel=0.05)
    assert fit.a == pytest.approx(a_true, rel=0.02)
    assert fit.r2_weighted > 0.999
    assert not fit.at_grid_boundary


def test_fit_two_regime_flags_a_crossover_outside_the_data():
    """Pure square-root data has no linear branch to find, so the optimiser
    pushes q* to the bottom of the grid. That has to be reported as a boundary
    hit, not as an estimate."""
    q = np.exp(np.linspace(np.log(1e-5), np.log(1e-1), 200))
    y = 2.0 * np.sqrt(q)
    fit = cx.fit_two_regime(q, y, np.ones_like(q), n_grid=200)
    assert fit.at_grid_boundary


def test_crossover_in_ticks_converts_correctly():
    fit = cx.CrossoverFit(a=2.0, q_star=0.01, wsse=0.0, r2_weighted=1.0,
                          n_bins=10, profile=pd.DataFrame())
    # impact at the crossover is a*q* = 0.02 in sigma units; with sigma 0.01 and
    # a $100 mid that is 0.02 * 0.01 * 100 = $0.02, which is two cents
    out = cx.crossover_in_ticks(fit, sigma_d=0.01, mid=100.0, tick=0.01)
    assert out["impact_at_crossover_dollars"] == pytest.approx(0.02)
    assert out["impact_at_crossover_ticks"] == pytest.approx(2.0)
    assert out["above_one_tick"] is True
    assert cx.crossover_in_ticks(fit, 0.001, 100.0)["above_one_tick"] is False


def bars_with_one_metaorder(n=600, bin_seconds=30):
    """A quiet session with one sustained one-sided burst of known size."""
    sec = np.arange(34200, 34200 + n)
    signed = np.zeros(n)
    volume = np.full(n, 100.0)              # two-sided background
    burst = slice(120, 240)                 # 120 seconds = four 30s bins
    signed[burst] = 900.0
    volume[burst] = 1000.0
    mid = np.full(n, 100.0)
    mid[180:] = 101.0                       # the price moves during the burst
    return pd.DataFrame({"sec": sec, "signed_vol": signed, "volume": volume,
                         "mid": mid})


def test_bin_metaorders_finds_the_burst_and_measures_it():
    bars = bars_with_one_metaorder()
    out = cx.bin_metaorders(bars, session_volume=1e6, min_size_fraction=0.0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.sign == 1.0
    assert row.duration_seconds == 120
    assert row.Q == pytest.approx(900.0 * 120)
    assert row.impact > 0


def test_bin_metaorders_applies_every_filter():
    bars = bars_with_one_metaorder()
    # dominance of the burst bins is 900/1000 = 0.9; raise the bar past it
    assert len(cx.bin_metaorders(bars, 1e6, dominance_min=0.95,
                                 min_size_fraction=0.0)) == 0
    assert len(cx.bin_metaorders(bars, 1e6, min_duration=150,
                                 min_size_fraction=0.0)) == 0
    # the burst is 108,000 shares; a 1e6-share session makes that 0.108
    assert len(cx.bin_metaorders(bars, 1e6, min_size_fraction=0.2)) == 0
    assert len(cx.bin_metaorders(bars, 1e6, min_size_fraction=0.05)) == 1


def test_bin_metaorders_breaks_a_run_on_a_sign_change():
    bars = bars_with_one_metaorder()
    bars.loc[180:209, "signed_vol"] = -900.0
    out = cx.bin_metaorders(bars, 1e6, min_duration=0, min_size_fraction=0.0)
    assert len(out) == 3
    assert list(out["sign"]) == [1.0, -1.0, 1.0]


def test_fit_published_recovers_a_known_prefactor_and_exponent():
    rng = np.random.default_rng(0)
    n = 4000
    q = np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), n))
    c_true, delta_true, sigma = 0.7, 0.5, 0.02
    impact = c_true * sigma * q ** delta_true
    orders = pd.DataFrame({"participation": q, "impact": impact})
    fit = cx.fit_published(orders, sigma, n_boot=50)
    assert fit.delta == pytest.approx(delta_true, abs=0.01)
    assert fit.c_free == pytest.approx(c_true, rel=0.02)
    assert fit.c_half == pytest.approx(c_true, rel=0.02)
    assert fit.delta_ci[0] < delta_true < fit.delta_ci[1]


def test_fit_published_bootstrap_respects_session_blocks():
    """A session-blocked bootstrap on data with a per-session offset must be
    wider than one that treats the orders as independent."""
    rng = np.random.default_rng(1)
    frames = []
    for s in range(6):
        q = np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), 400))
        c = 0.7 * (1.0 + 0.4 * (s - 2.5) / 2.5)     # the prefactor shifts by day
        frames.append(pd.DataFrame({"participation": q,
                                    "impact": c * 0.02 * np.sqrt(q),
                                    "session": f"S{s}"}))
    orders = pd.concat(frames, ignore_index=True)
    blocked = cx.fit_published(orders, 0.02, n_boot=200, groups=orders.session)
    naive = cx.fit_published(orders, 0.02, n_boot=200)
    assert (blocked.c_half_ci[1] - blocked.c_half_ci[0]) > \
           (naive.c_half_ci[1] - naive.c_half_ci[0])
