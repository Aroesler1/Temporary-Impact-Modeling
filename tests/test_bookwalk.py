"""Book-walk cost and the fitter suite, on ladders and curves with known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bookwalk as bw


def ladder(prices, sizes, n=1):
    """A DataFrame of `n` identical MBP-10 snapshots from one ask/bid ladder."""
    depth = len(prices)
    cols = {}
    for i in range(depth):
        cols[f"ask_px_{i:02d}"] = [prices[i]] * n
        cols[f"ask_sz_{i:02d}"] = [sizes[i]] * n
        cols[f"bid_px_{i:02d}"] = [prices[0] - 0.02 - 0.01 * i] * n
        cols[f"bid_sz_{i:02d}"] = [sizes[i]] * n
    return pd.DataFrame(cols)


def test_walk_vwap_fills_the_marginal_level_partially():
    px = np.array([[10.00, 10.01, 10.02]])
    sz = np.array([[100.0, 200.0, 300.0]])
    got = bw.walk_vwap(px, sz, np.array([50.0, 100.0, 150.0, 300.0, 600.0]))
    want = [10.00, 10.00,
            (100 * 10.00 + 50 * 10.01) / 150,
            (100 * 10.00 + 200 * 10.01) / 300,
            (100 * 10.00 + 200 * 10.01 + 300 * 10.02) / 600]
    np.testing.assert_allclose(got[0], want, rtol=1e-12)


def test_walk_vwap_is_nan_beyond_displayed_depth():
    px = np.array([[10.0, 10.1]])
    sz = np.array([[10.0, 10.0]])
    got = bw.walk_vwap(px, sz, np.array([20.0, 20.0001, 50.0]))
    assert np.isfinite(got[0, 0])
    assert np.isnan(got[0, 1]) and np.isnan(got[0, 2])


def test_walk_costs_measures_from_the_mid_not_the_touch():
    book = ladder([10.00, 10.01], [100.0, 100.0])
    walk = bw.walk_costs(book, np.array([1.0]), depth=2)
    # best bid is 9.98, so mid is 9.99 and the half spread is 0.01
    assert walk["half_spread"][0] == pytest.approx(0.01)
    assert walk["buy_cost"][0, 0] == pytest.approx(0.01)      # one half spread
    assert walk["sell_cost"][0, 0] == pytest.approx(0.01)


def test_original_recipe_reports_the_filter_as_three_numbers():
    rng = np.random.default_rng(0)
    n = 400
    cols = {}
    for i in range(10):
        cols[f"ask_px_{i:02d}"] = 10.0 + 0.01 * i
        cols[f"ask_sz_{i:02d}"] = rng.integers(50, 400, n).astype(float)
        cols[f"bid_px_{i:02d}"] = 9.99 - 0.01 * i
        cols[f"bid_sz_{i:02d}"] = rng.integers(50, 400, n).astype(float)
    out = bw.original_recipe(pd.DataFrame(cols), cap=1000.0)
    assert out["n_buckets_kept"] < out["n_buckets_all"]
    assert 0.0 < out["share_of_snapshot_points_dropped"] < 1.0
    # count weighting is a separate choice from the size cap, so the two
    # unfiltered variants must not be forced to agree
    assert out["p_unfiltered"] != out["p_unfiltered_count_weighted"]


def synthetic_panel(form, theta, n_sessions=15, n_points=40, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        u = np.exp(rng.uniform(np.log(1e-3), np.log(30.0), n_points))
        y = form.predict(np.asarray(theta, float), u) * (
            1 + noise * rng.standard_normal(n_points))
        rows.append(pd.DataFrame({"session": f"S{s}", "u": u, "y": y,
                                  "w": rng.integers(50, 5000, n_points).astype(float)}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("exponent", [0.25, 0.45, 0.6])
def test_wnls_recovers_a_known_power_law(exponent):
    p = synthetic_panel(bw.POWER, [1.7, exponent])
    fit = bw.fit_wnls(bw.POWER, p.u.to_numpy(), p.y.to_numpy(), p.w.to_numpy())
    assert fit.exponent == pytest.approx(exponent, abs=0.01)
    assert fit.params[0] == pytest.approx(1.7, rel=0.03)
    assert fit.se_robust[1] > 0


def test_wnls_recovers_a_known_piecewise_curve():
    p = synthetic_panel(bw.PIECEWISE, [1.0, 0.8, 0.4], noise=0.01)
    fit = bw.fit_wnls(bw.PIECEWISE, p.u.to_numpy(), p.y.to_numpy(), p.w.to_numpy())
    c, u0, exponent = fit.params
    assert c == pytest.approx(1.0, rel=0.05)
    assert u0 == pytest.approx(0.8, rel=0.20)
    assert exponent == pytest.approx(0.4, abs=0.03)


def test_wnls_recovers_a_known_logarithmic_curve():
    p = synthetic_panel(bw.LOGARITHMIC, [2.0, 0.5], noise=0.01)
    fit = bw.fit_wnls(bw.LOGARITHMIC, p.u.to_numpy(), p.y.to_numpy(), p.w.to_numpy())
    assert fit.params[0] == pytest.approx(2.0, rel=0.05)
    assert fit.params[1] == pytest.approx(0.5, rel=0.15)


def test_athl_with_beta_fixed_does_not_move_it():
    form = bw.athl(0.6)
    p = synthetic_panel(bw.POWER, [1.0, 0.35])
    fit = bw.fit_wnls(form, p.u.to_numpy(), p.y.to_numpy(), p.w.to_numpy())
    assert fit.param_names == ("eta",)
    assert np.isnan(fit.exponent)


def test_loso_cv_prefers_the_generating_form():
    p = synthetic_panel(bw.POWER, [1.2, 0.4])
    power = bw.loso_cv(bw.POWER, p)["loso_rmse"]
    linear = bw.loso_cv(bw.LINEAR, p)["loso_rmse"]
    assert power < linear
    assert bw.loso_cv(bw.POWER, p)["n_folds"] == 15


def test_aic_penalises_the_extra_parameter_when_it_buys_nothing():
    """Power data: the three-parameter piecewise form fits no better, so AIC
    must rank it worse even though its SSE cannot be higher."""
    p = synthetic_panel(bw.POWER, [1.0, 0.4], noise=0.005)
    power = bw.fit_wnls(bw.POWER, p.u.to_numpy(), p.y.to_numpy(), p.w.to_numpy())
    piecewise = bw.fit_wnls(bw.PIECEWISE, p.u.to_numpy(), p.y.to_numpy(),
                            p.w.to_numpy())
    assert piecewise.wsse <= power.wsse * 1.001
    assert power.aic < piecewise.aic


def test_block_bootstrap_collapses_on_noiseless_data():
    p = synthetic_panel(bw.POWER, [1.0, 0.42], noise=0.0)
    lo, hi, draws = bw.block_bootstrap_exponent(bw.POWER, p, n_boot=200)
    assert lo == pytest.approx(0.42, abs=1e-5)
    assert hi == pytest.approx(0.42, abs=1e-5)
    assert len(draws) == 200


def test_block_bootstrap_widens_with_noise_and_stays_centred():
    """Multiplicative noise on a level-space weighted fit biases the exponent up
    a little, so this checks the two properties that are the estimator's job:
    the interval grows with the noise, and it stays near the truth."""
    narrow = bw.block_bootstrap_exponent(
        bw.POWER, synthetic_panel(bw.POWER, [1.0, 0.42], noise=0.01), n_boot=200)
    wide = bw.block_bootstrap_exponent(
        bw.POWER, synthetic_panel(bw.POWER, [1.0, 0.42], noise=0.05), n_boot=200)
    assert (wide[1] - wide[0]) > 3 * (narrow[1] - narrow[0])
    for lo, hi, _ in (narrow, wide):
        assert abs(0.5 * (lo + hi) - 0.42) < 0.02


def test_block_bootstrap_brackets_the_full_sample_estimate():
    p = synthetic_panel(bw.POWER, [1.0, 0.42], noise=0.05)
    fit = bw.fit_wnls(bw.POWER, p.u.to_numpy(), p.y.to_numpy(), p.w.to_numpy())
    lo, hi, _ = bw.block_bootstrap_exponent(bw.POWER, p, n_boot=200)
    assert lo <= fit.exponent <= hi


def test_profile_is_minimised_at_the_true_exponent():
    p = synthetic_panel(bw.POWER, [1.0, 0.45], noise=0.01)
    grid = np.linspace(0.3, 0.6, 31)
    profile = bw.profile_exponent(bw.POWER, p.u.to_numpy(), p.y.to_numpy(),
                                  p.w.to_numpy(), grid)
    assert profile.loc[profile.wsse.idxmin(), "exponent"] == pytest.approx(0.45,
                                                                          abs=0.02)
    assert profile.deviance.min() == pytest.approx(0.0)
    # the profile must rise away from the optimum in both directions
    assert profile.deviance.iloc[0] > 3.841
    assert profile.deviance.iloc[-1] > 3.841


def test_bin_walk_drops_rows_where_any_column_is_missing():
    size = np.array([1.0, 2.0, 3.0, 4.0] * 30)
    good = size ** 0.5
    bad = good.copy()
    bad[0] = np.nan
    table = bw.bin_walk(size, {"a": good, "b": bad}, n_bins=4)
    assert int(table.n.sum()) == len(size) - 1


def test_bin_count_weights_change_the_fit():
    """A weighted fit must actually use the weights, or the whole
    heteroskedasticity argument in the module docstring is decoration."""
    u = np.array([0.01, 0.1, 1.0, 10.0] * 10)
    y = 1.0 * u ** 0.4
    y[0] = y[0] * 3.0                       # one badly measured point
    heavy = np.where(np.arange(len(u)) == 0, 1e6, 1.0)
    light = np.where(np.arange(len(u)) == 0, 1e-6, 1.0)
    a = bw.fit_wnls(bw.POWER, u, y, heavy).exponent
    b = bw.fit_wnls(bw.POWER, u, y, light).exponent
    assert abs(a - b) > 0.05
    assert b == pytest.approx(0.4, abs=0.02)
