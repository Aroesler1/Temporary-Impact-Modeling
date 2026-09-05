"""Cross-section: the sampler, the trade bars, and the regression step."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cross_section as cs


# --------------------------------------------------------------------------
# stratified sampling
# --------------------------------------------------------------------------

def universe(n=180, seed=0, coupled=False) -> pd.DataFrame:
    """A synthetic universe of `n` names.

    By default the two stratification axes are independent, which is the clean
    case. `coupled=True` lets dollar volume carry the price through, giving the
    strong negative correlation between relative tick size and dollar volume
    that the real S&P 500 has and that leaves off-diagonal cells thin. That
    coupling is the whole reason for stratifying rather than sampling at random.
    """
    rng = np.random.default_rng(seed)
    price = np.exp(rng.uniform(np.log(5), np.log(900), n))
    share_volume = np.exp(rng.uniform(np.log(2e5), np.log(4e7), n))
    dollar_volume = price * share_volume if coupled else share_volume
    return pd.DataFrame({"symbol": [f"S{i:04d}" for i in range(n)],
                         "relative_tick": 0.01 / price,
                         "dollar_volume": dollar_volume})


def test_stratify_fills_nine_cells_evenly():
    strat = cs.stratify(universe(), names_per_cell=12)
    assert len(strat.cell_counts) == 9
    assert set(strat.cell_counts.drawn) == {12}
    assert len(strat.sample) == 108
    assert not strat.short_cells


def test_stratify_is_deterministic_under_the_seed():
    a = cs.stratify(universe(), names_per_cell=10)
    b = cs.stratify(universe(), names_per_cell=10)
    pd.testing.assert_series_equal(a.sample.symbol, b.sample.symbol)
    c = cs.stratify(universe(), names_per_cell=10, seed=cs.SAMPLE_SEED + 1)
    assert list(c.sample.symbol) != list(a.sample.symbol)


def test_stratify_does_not_depend_on_input_row_order():
    """A draw that changed with the order rows happened to arrive in would not
    be reproducible from the recorded seed."""
    frame = universe()
    a = cs.stratify(frame, names_per_cell=10)
    b = cs.stratify(frame.sample(frac=1.0, random_state=7), names_per_cell=10)
    assert sorted(a.sample.symbol) == sorted(b.sample.symbol)


def test_stratify_reports_a_short_cell_instead_of_padding_it():
    frame = universe(n=60)
    strat = cs.stratify(frame, names_per_cell=12)
    assert strat.short_cells
    assert (strat.cell_counts.drawn <= 12).all()
    assert (strat.cell_counts.drawn == strat.cell_counts[["available", "drawn"]]
            .min(axis=1)).all()
    # nothing is drawn twice, and nothing is invented
    assert strat.sample.symbol.is_unique
    assert set(strat.sample.symbol).issubset(set(frame.symbol))


def test_stratify_handles_a_universe_whose_axes_are_correlated():
    """Relative tick size and dollar volume are negatively correlated in the
    real S&P 500, so some crossed cells are genuinely thin. Those must be
    reported short, never padded from a neighbouring cell."""

    strat = cs.stratify(universe(coupled=True), names_per_cell=12)
    assert len(strat.cell_counts) == 9
    assert strat.short_cells
    assert (strat.cell_counts.drawn == strat.cell_counts[["available", "drawn"]]
            .min(axis=1)).all()


def test_stratify_drops_unusable_rows():
    frame = universe(n=90)
    frame.loc[0, "relative_tick"] = np.nan
    frame.loc[1, "dollar_volume"] = 0.0
    strat = cs.stratify(frame, names_per_cell=5)
    assert frame.symbol[0] not in set(strat.sample.symbol)
    assert frame.symbol[1] not in set(strat.sample.symbol)


def test_tercile_splits_by_rank_not_by_value():
    """Dollar volume spans orders of magnitude; splitting on raw value would
    put nearly everything in one bucket."""
    skewed = pd.Series(np.r_[np.arange(90, dtype=float), [1e9, 2e9, 3e9]])
    counts = cs.tercile(skewed).value_counts()
    assert counts.max() - counts.min() <= 1


# --------------------------------------------------------------------------
# trades to bars
# --------------------------------------------------------------------------

def test_aggressor_sign_uses_the_trades_convention():
    """On the trades schema `side` is the AGGRESSING side, the opposite of MBO.
    Getting this backwards inverts every metaorder in the study."""
    got = cs.aggressor_sign(np.array([b"B", b"A", b"N"]))
    np.testing.assert_array_equal(got, [1.0, -1.0, 0.0])


def test_trade_bars_sum_signed_and_unsigned_volume_separately():
    bars = cs.trade_bars(np.array([1, 1, 1, 2]), np.array([10.0, 10.1, 10.2, 11.0]),
                         np.array([100.0, 50.0, 30.0, 20.0]),
                         np.array([1.0, -1.0, 0.0, 1.0]))
    assert list(bars.sec) == [1, 2]
    # the unsided print counts toward volume and not toward signed volume
    assert bars.volume.iloc[0] == pytest.approx(180.0)
    assert bars.signed_vol.iloc[0] == pytest.approx(50.0)
    # the price column is the LAST trade in the bin, not a mid
    assert bars.mid.iloc[0] == pytest.approx(10.2)


def test_realised_vol_5min_recovers_a_known_volatility():
    rng = np.random.default_rng(1)
    n_bins = cs.BINS_PER_SESSION
    target = 0.02
    per_bin = target / np.sqrt(n_bins)
    # one trade at the end of each five-minute bin, on a known random walk
    sec = cs.RTH_OPEN_SEC + np.arange(n_bins) * cs.FIVE_MINUTES + cs.FIVE_MINUTES - 1
    price = 100.0 * np.exp(np.cumsum(rng.normal(0, per_bin, n_bins)))
    got = cs.realised_vol_5min(sec, price)
    assert got == pytest.approx(target, rel=0.25)


def test_realised_vol_5min_is_nan_on_a_session_with_almost_no_trades():
    assert np.isnan(cs.realised_vol_5min(np.array([34200, 34500]),
                                         np.array([10.0, 10.1])))


def test_halfhour_vol_profile_finds_a_planted_u_shape():
    rng = np.random.default_rng(2)
    n_bins = cs.BINS_PER_SESSION
    sec = cs.RTH_OPEN_SEC + np.arange(n_bins) * cs.FIVE_MINUTES + 1
    bucket = (np.arange(n_bins) * cs.FIVE_MINUTES) // cs.HALF_HOUR
    scale = np.where(bucket == 0, 3.0, 1.0)          # a loud opening half hour
    price = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, n_bins) * scale))
    profile = cs.halfhour_vol_profile(sec, price)
    assert len(profile) == cs.N_HALF_HOURS
    assert profile[0] > 1.5
    assert np.nanmedian(profile[1:]) < profile[0]


def test_halfhour_vol_profile_leaves_a_thin_bucket_as_nan():
    sec = np.array([34200, 34500, 34800, 35100, 35400, 35700, 36000,
                    36300, 36600, 36900, 37200])
    profile = cs.halfhour_vol_profile(sec, np.linspace(100.0, 101.0, len(sec)))
    assert np.isnan(profile[-1])


# --------------------------------------------------------------------------
# the cross-sectional regression
# --------------------------------------------------------------------------

def test_ols_robust_recovers_a_planted_tick_size_slope():
    """The whole cross-sectional claim is one coefficient. This plants it."""
    rng = np.random.default_rng(3)
    n = 120
    log_tick = rng.uniform(np.log(1e-5), np.log(3e-3), n)
    log_volume = rng.normal(20.0, 1.2, n)
    true_slope = -0.08
    delta = 0.5 + true_slope * (log_tick - log_tick.mean()) + \
        0.02 * (log_volume - log_volume.mean()) + rng.normal(0, 0.03, n)
    result = cs.ols_robust(delta, pd.DataFrame({"log_tick": log_tick,
                                                "log_volume": log_volume}))
    i = result.names.index("log_tick")
    assert result.coef[i] == pytest.approx(true_slope, abs=0.02)
    assert result.coef[i] - 1.96 * result.se[i] < true_slope
    assert true_slope < result.coef[i] + 1.96 * result.se[i]
    assert result.tstat[i] < -1.96


def test_ols_robust_finds_no_slope_when_there_is_none():
    rng = np.random.default_rng(4)
    n = 150
    x = rng.normal(0, 1, n)
    y = 0.4 + rng.normal(0, 0.05, n)
    result = cs.ols_robust(y, pd.DataFrame({"x": x}))
    assert abs(result.tstat[1]) < 1.96


def test_ols_robust_standard_errors_widen_under_heteroskedasticity():
    """HC1 exists for this case; if the errors did not react to it the robust
    standard error would be decoration."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(0, 1, n)
    homo = cs.ols_robust(1.0 + 0.5 * x + rng.normal(0, 0.3, n),
                         pd.DataFrame({"x": x}))
    hetero = cs.ols_robust(1.0 + 0.5 * x + rng.normal(0, 0.3, n) * np.abs(x) * 3,
                           pd.DataFrame({"x": x}))
    assert hetero.se[1] > homo.se[1]


def test_ols_robust_drops_non_finite_rows():
    x = np.arange(50, dtype=float)
    y = 2.0 * x
    y[0] = np.nan
    x_frame = pd.DataFrame({"x": x})
    x_frame.loc[1, "x"] = np.nan
    result = cs.ols_robust(y, x_frame)
    assert result.n == 48
    assert result.coef[1] == pytest.approx(2.0, rel=1e-9)


def test_ols_robust_refuses_an_underdetermined_fit():
    with pytest.raises(ValueError):
        cs.ols_robust(np.array([1.0, 2.0]),
                      pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))


def test_consolidated_normaliser_is_a_raising_stub():
    """It must fail loudly rather than fall back to venue volume and let a
    consolidated claim be made from single-venue data."""
    with pytest.raises(NotImplementedError, match="pending row"):
        cs.consolidated_normalisers(["AAPL"], "2024-04-01", "2024-09-30")
