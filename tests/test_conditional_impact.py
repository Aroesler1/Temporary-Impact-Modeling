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
    assert result.r2("propagator") > 0.99
    assert result.scores["propagator"]["slope"] == pytest.approx(1.0, abs=0.02)
    # the square-root model has the wrong shape here and must score worse
    assert result.r2("sqrt") < result.r2("propagator")


# --------------------------------------------------------------------------
# which volatility belongs in the square-root model
# --------------------------------------------------------------------------

def test_blend_sigma_reduces_to_each_end():
    trail = np.array([0.01, 0.02, 0.04])
    np.testing.assert_allclose(ci.blend_sigma(0.03, trail, 1.0), 0.03)
    np.testing.assert_allclose(ci.blend_sigma(0.03, trail, 0.0), trail)
    # the midpoint is the geometric mean, not the arithmetic one
    np.testing.assert_allclose(ci.blend_sigma(0.03, trail, 0.5),
                               np.sqrt(0.03 * trail))


def test_fit_blend_alpha_recovers_a_known_weight():
    """Orders whose impact was generated at a known blend must recover it."""
    rng = np.random.default_rng(0)
    n, volume, sigma_d = 4000, 1e7, 0.02
    trail = np.exp(rng.uniform(np.log(0.005), np.log(0.05), n))
    shares = np.exp(rng.uniform(np.log(10), np.log(1e5), n))
    alpha_true, c_true = 0.3, 0.9
    sigma = ci.blend_sigma(sigma_d, trail, alpha_true)
    impact = c_true * sigma * np.sqrt(shares / volume)
    orders = pd.DataFrame({"sign": np.ones(n), "shares": shares,
                           "mid_start": np.full(n, 100.0),
                           "mid_end": 100.0 * np.exp(impact)})
    alpha, c = ci.fit_blend_alpha(orders, volume, sigma_d, trail)
    assert alpha == pytest.approx(alpha_true, abs=0.02)
    assert c == pytest.approx(c_true, rel=0.02)


def test_halfhour_bucket_starts_at_the_open():
    got = ci.halfhour_bucket(np.array([34200.0, 35999.0, 36000.0, 57599.0]))
    np.testing.assert_array_equal(got, [0, 0, 1, 12])


def test_halfhour_ratios_finds_a_known_intraday_shape():
    """A session that is twice as volatile in its second half hour must show a
    ratio twice as large there."""
    n = 3600
    rng = np.random.default_rng(1)
    ret = np.r_[rng.standard_normal(1800) * 1e-5,
                rng.standard_normal(1800) * 2e-5]
    bars = pd.DataFrame({"sec": np.arange(34200, 34200 + n),
                         "mid": 100.0 * np.exp(np.cumsum(ret))})
    ratios = ci.halfhour_ratios(bars)
    assert set(ratios.index) == {0, 1}
    assert ratios[1] / ratios[0] == pytest.approx(2.0, rel=0.1)


def test_profile_sigma_applies_the_bucket_multiplier():
    profile = pd.Series({0: 1.5, 1: 0.5})
    got = ci.profile_sigma(0.02, profile, np.array([34200.0, 36000.0]))
    np.testing.assert_allclose(got, [0.03, 0.01])


def test_profile_sigma_falls_back_to_one_where_there_is_no_donor():
    """An uncovered half hour must be left at sigma_D rather than silently
    given a neighbour's multiplier or a NaN."""
    got = ci.profile_sigma(0.02, pd.Series({0: 1.5}), np.array([36000.0]))
    assert got[0] == pytest.approx(0.02)


# --------------------------------------------------------------------------
# the participation-rate term
# --------------------------------------------------------------------------

def test_execution_rate_is_one_when_the_order_was_the_only_trade():
    bars = pd.DataFrame({"sec": np.arange(34200, 34205), "volume": [0.0, 500.0, 0, 0, 0],
                         "mid": 100.0, "signed_vol": 0.0})
    orders = pd.DataFrame({"shares": [500.0], "t_start": [34201.2], "t_end": [34201.8]})
    assert ci.execution_rate(orders, bars)[0] == pytest.approx(1.0)


def test_execution_rate_falls_when_others_traded_in_the_window():
    bars = pd.DataFrame({"sec": np.arange(34200, 34205),
                         "volume": [0.0, 500.0, 500.0, 0.0, 0.0],
                         "mid": 100.0, "signed_vol": 0.0})
    orders = pd.DataFrame({"shares": [250.0], "t_start": [34201.0], "t_end": [34202.5]})
    assert ci.execution_rate(orders, bars)[0] == pytest.approx(0.25)


def test_execution_rate_is_clipped_into_the_unit_interval():
    bars = pd.DataFrame({"sec": np.arange(34200, 34203), "volume": [10.0, 10.0, 10.0],
                         "mid": 100.0, "signed_vol": 0.0})
    orders = pd.DataFrame({"shares": [1e9, 1e-9], "t_start": [34200.0, 34200.0],
                           "t_end": [34200.5, 34200.5]})
    rate = ci.execution_rate(orders, bars)
    assert rate.max() <= 1.0
    assert rate.min() > 0.0


def test_fit_rate_model_recovers_known_parameters():
    rng = np.random.default_rng(2)
    n, volume, sigma = 6000, 1e7, 0.02
    shares = np.exp(rng.uniform(np.log(100), np.log(1e5), n))
    rate = rng.uniform(0.02, 1.0, n)
    c_true, delta_true, k_true = 0.8, 0.45, 0.15
    impact = sigma * c_true * (shares / volume) ** delta_true * (
        1.0 + k_true * np.log(rate))
    orders = pd.DataFrame({"sign": np.ones(n), "shares": shares,
                           "mid_start": np.full(n, 100.0),
                           "mid_end": 100.0 * np.exp(impact)})
    fit = ci.fit_rate_model(orders, volume, sigma, rate)
    assert fit.c == pytest.approx(c_true, rel=0.03)
    assert fit.delta == pytest.approx(delta_true, abs=0.02)
    assert fit.k == pytest.approx(k_true, abs=0.02)


def test_fit_rate_model_returns_k_near_zero_when_rate_does_not_matter():
    rng = np.random.default_rng(3)
    n, volume, sigma = 6000, 1e7, 0.02
    shares = np.exp(rng.uniform(np.log(100), np.log(1e5), n))
    rate = rng.uniform(0.02, 1.0, n)
    impact = sigma * 0.8 * np.sqrt(shares / volume)
    orders = pd.DataFrame({"sign": np.ones(n), "shares": shares,
                           "mid_start": np.full(n, 100.0),
                           "mid_end": 100.0 * np.exp(impact)})
    assert abs(ci.fit_rate_model(orders, volume, sigma, rate).k) < 0.01


def test_rate_model_prediction_stays_positive_over_the_fitted_range():
    """k is bounded so the correction factor cannot flip the sign of predicted
    impact for the slowest orders in the training set."""
    rng = np.random.default_rng(4)
    n, volume, sigma = 3000, 1e7, 0.02
    shares = np.exp(rng.uniform(np.log(100), np.log(1e5), n))
    rate = rng.uniform(1e-4, 1.0, n)
    impact = sigma * 0.8 * np.sqrt(shares / volume) * (1 + 5.0 * np.log(rate))
    orders = pd.DataFrame({"sign": np.ones(n), "shares": shares,
                           "mid_start": np.full(n, 100.0),
                           "mid_end": 100.0 * np.exp(impact)})
    fit = ci.fit_rate_model(orders, volume, sigma, rate)
    assert (ci.predict_rate_model(orders, fit, volume, sigma, rate) > 0).all()


def test_evaluate_session_scores_every_model_it_is_given_data_for():
    n, g0 = 6000, 3e-6
    rng = np.random.default_rng(5)
    vol = rng.standard_normal(n) * 200.0
    mid = 100.0 * np.exp(np.cumsum(g0 * vol))
    bars = pd.DataFrame({"sec": np.arange(34200, 34200 + n), "signed_vol": vol,
                         "volume": np.abs(vol), "mid": mid})
    sec = bars.sec.to_numpy(float)
    orders = pd.DataFrame({"sign": np.sign(vol), "shares": np.abs(vol),
                           "t_start": sec, "t_end": sec,
                           "mid_start": np.r_[100.0, mid[:-1]], "mid_end": mid})
    profile = pd.Series({b: 1.0 for b in range(13)})
    result = ci.evaluate_session("SYNTH", bars, orders, 1e7, 0.02,
                                 tod_profile_loso=profile)
    for name in ("propagator", "propagator_scaled", "sqrt", "sqrt_trailing_sigma",
                 "sqrt_blend", "sqrt_tod_loso", "sqrt_rate"):
        assert name in result.scores
        assert np.isfinite(result.r2(name))
    # no prior-session donor was supplied, so that model must be absent rather
    # than silently scored against a default
    assert "sqrt_tod_prior" not in result.scores
    assert result.n_test > 0
