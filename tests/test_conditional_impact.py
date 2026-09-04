"""Conditional impact accuracy, on flow whose impact is known by construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import conditional_impact as ci


def synthetic_bars(n=4000, g0=2e-6, delta=1.0, seed=0):
    """Bars whose returns ARE the kernel applied to the flow, plus a little noise."""
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal(n) * 300.0
    ret = g0 * np.sign(vol) * np.abs(vol) ** delta + 1e-7 * rng.standard_normal(n)
    mid = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"sec": np.arange(34200, 34200 + n), "signed_vol": vol,
                         "volume": np.abs(vol), "mid": mid})


def test_predict_propagator_matches_the_kernel_by_hand():
    cal = ci.Calibration(delta=0.5, n_lags=2, kernel=np.array([2.0, 1.0, 0.5]),
                         n_train=100, inner_r2_out=0.1)
    orders = pd.DataFrame({"sign": [1.0, -1.0, 1.0], "shares": [100.0, 400.0, 900.0],
                           "t_start": [10.0, 20.0, 30.0],
                           "t_end": [10.5, 20.2, 32.9]})
    got = ci.predict_propagator(orders, cal)
    # one-second orders pay G(0) only; the three-second order accumulates
    # (3-0)G0 + (3-1)G1 + (3-2)G2 = 8.5 against a rate of 300 shares a second.
    # All three are positive: impact is reported in the order's own direction,
    # so the sell is not negative.
    np.testing.assert_allclose(got, [2.0 * 10.0, 2.0 * 20.0, 8.5 * np.sqrt(300.0)],
                               rtol=1e-10)


def test_predict_propagator_is_signed_like_realised_impact():
    """A buy and a sell of the same size under the same kernel must both come
    back positive, or the R2 against realised impact scores the sign convention
    instead of the model."""
    cal = ci.Calibration(delta=1.0, n_lags=0, kernel=np.array([1e-6]),
                         n_train=10, inner_r2_out=0.0)
    orders = pd.DataFrame({"sign": [1.0, -1.0], "shares": [500.0, 500.0],
                           "t_start": [0.0, 1.0], "t_end": [0.0, 1.0]})
    got = ci.predict_propagator(orders, cal)
    assert got[0] == pytest.approx(got[1])
    assert got[0] > 0


def test_predict_propagator_never_looks_past_the_kernel():
    cal = ci.Calibration(delta=1.0, n_lags=1, kernel=np.array([1.0, 0.0]),
                         n_train=10, inner_r2_out=0.0)
    orders = pd.DataFrame({"sign": [1.0], "shares": [10.0],
                           "t_start": [0.0], "t_end": [9.0]})
    # ten seconds at one share a second, memoryless kernel: total impact is 10
    assert ci.predict_propagator(orders, cal)[0] == pytest.approx(10.0)


def test_calibrate_on_window_recovers_a_known_kernel_and_ignores_the_tail():
    bars = synthetic_bars()
    # corrupt the last 30% so a calibration that peeked would be visibly wrong
    bars.loc[len(bars) * 7 // 10:, "signed_vol"] *= 50.0
    cal = ci.calibrate_on_window(bars, train_end=int(len(bars) * 0.7))
    assert cal.delta == 1.0
    assert cal.kernel[0] == pytest.approx(2e-6, rel=0.05)


def test_realised_impact_is_signed_by_the_order_direction():
    orders = pd.DataFrame({"sign": [1.0, -1.0], "shares": [1.0, 1.0],
                           "mid_start": [100.0, 100.0], "mid_end": [101.0, 99.0]})
    got = ci.realised_impact(orders)
    assert got[0] > 0 and got[1] > 0            # both moved WITH their order
    assert got[0] == pytest.approx(np.log(1.01))


def test_fit_sqrt_coefficient_recovers_a_known_c():
    rng = np.random.default_rng(1)
    n, volume, sigma = 5000, 1e7, 0.02
    shares = np.exp(rng.uniform(np.log(10), np.log(1e5), n))
    c_true = 0.8
    impact = c_true * sigma * np.sqrt(shares / volume)
    orders = pd.DataFrame({"sign": np.ones(n), "shares": shares,
                           "mid_start": np.full(n, 100.0),
                           "mid_end": 100.0 * np.exp(impact)})
    assert ci.fit_sqrt_coefficient(orders, volume, sigma) == pytest.approx(
        c_true, rel=0.02)


def test_fit_scale_recovers_a_known_multiple():
    pred = np.linspace(1.0, 10.0, 200)
    assert ci.fit_scale(pred, 3.5 * pred) == pytest.approx(3.5)


def test_scores_are_negative_when_a_prediction_is_biased_high():
    """R2 without a refit must punish a level error; the refit R2 must not,
    which is exactly why both are reported."""
    rng = np.random.default_rng(2)
    realised = rng.standard_normal(1000) * 0.001
    predicted = 5.0 * realised
    scores = ci._scores(realised, predicted)
    assert scores["r2_no_refit"] < 0
    assert scores["r2_refit"] == pytest.approx(1.0, abs=1e-9)
    assert scores["slope"] == pytest.approx(0.2, rel=1e-6)


def test_calibration_table_is_monotone_for_a_perfect_model():
    rng = np.random.default_rng(3)
    predicted = rng.uniform(0.0001, 0.01, 5000)
    table = ci.calibration_table(predicted, predicted)
    assert table.ratio.between(0.999, 1.001).all()
    assert table.predicted.is_monotonic_increasing
    assert len(table) == 10


def test_trailing_sigma_is_causal_and_tracks_the_level():
    n = 6000
    rng = np.random.default_rng(4)
    # quiet first half, four times as volatile second half
    ret = np.r_[rng.standard_normal(n // 2) * 1e-5,
                rng.standard_normal(n // 2) * 4e-5]
    bars = pd.DataFrame({"sec": np.arange(34200, 34200 + n),
                         "mid": 100.0 * np.exp(np.cumsum(ret))})
    at = np.array([34200 + n // 2 - 100, 34200 + n - 100])
    sig = ci.trailing_sigma(bars, at, window_seconds=1800)
    assert sig[1] / sig[0] == pytest.approx(4.0, rel=0.15)
    # a window that ends before any volatility change cannot see it
    early = ci.trailing_sigma(bars, np.array([34200 + 2000]), window_seconds=1800)
    assert early[0] < sig[1]


def test_evaluate_session_scores_a_perfect_model_near_one():
    """Orders whose impact IS the kernel applied to their own flow: the
    propagator prediction should be nearly exact out of sample."""
    n = 6000
    g0 = 3e-6
    rng = np.random.default_rng(5)
    vol = rng.standard_normal(n) * 200.0
    ret = g0 * vol
    mid = 100.0 * np.exp(np.cumsum(ret))
    bars = pd.DataFrame({"sec": np.arange(34200, 34200 + n), "signed_vol": vol,
                         "volume": np.abs(vol), "mid": mid})
    sec = bars.sec.to_numpy(float)
    orders = pd.DataFrame({
        "sign": np.sign(vol), "shares": np.abs(vol),
        "t_start": sec, "t_end": sec,
        "mid_start": np.r_[100.0, mid[:-1]], "mid_end": mid,
    })
    result = ci.evaluate_session("SYNTH", bars, orders, session_volume=1e7,
                                 sigma_d=0.02)
    assert result.calibration.delta == 1.0
    assert result.propagator["r2_no_refit"] > 0.99
    assert result.propagator["slope"] == pytest.approx(1.0, abs=0.02)
    # the square-root model has the wrong shape here and must score worse
    assert result.sqrt_model["r2_no_refit"] < result.propagator["r2_no_refit"]
