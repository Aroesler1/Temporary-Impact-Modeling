"""Multi-level OFI and the two-flow regressions, on books with known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import orderflow as of


def test_level_ofi_counts_a_bid_improvement_as_its_whole_new_size():
    bid_px = np.array([10.00, 10.01])
    bid_sz = np.array([100.0, 300.0])
    ask_px = np.array([10.02, 10.02])
    ask_sz = np.array([100.0, 100.0])
    e = of.level_ofi(bid_px, bid_sz, ask_px, ask_sz)
    # the bid improved, so its full new size enters; the ask did not move, so
    # its unchanged size cancels
    assert e[1] == pytest.approx(300.0)


def test_level_ofi_counts_a_pulled_bid_as_minus_the_old_size():
    e = of.level_ofi(np.array([10.00, 9.99]), np.array([100.0, 50.0]),
                     np.array([10.02, 10.02]), np.array([100.0, 100.0]))
    assert e[1] == pytest.approx(-100.0)


def test_level_ofi_is_the_change_when_the_price_holds():
    e = of.level_ofi(np.array([10.0, 10.0]), np.array([100.0, 140.0]),
                     np.array([10.02, 10.02]), np.array([100.0, 100.0]))
    assert e[1] == pytest.approx(40.0)


def test_level_ofi_signs_the_ask_side_opposite_the_bid():
    bid = of.level_ofi(np.array([10.0, 10.0]), np.array([100.0, 200.0]),
                       np.array([10.02, 10.02]), np.array([100.0, 100.0]))
    ask = of.level_ofi(np.array([10.0, 10.0]), np.array([100.0, 100.0]),
                       np.array([10.02, 10.02]), np.array([100.0, 200.0]))
    assert bid[1] == pytest.approx(-ask[1])


def test_integrate_pca_recovers_a_common_factor():
    rng = np.random.default_rng(0)
    common = rng.standard_normal(3000)
    ofi = np.column_stack([common * (1.0 + 0.5 * i) + 0.1 * rng.standard_normal(3000)
                           for i in range(5)])
    integrated, weights = of.integrate_pca(ofi)
    assert weights[0] > 0                                   # sign-aligned to level 0
    assert abs(np.corrcoef(integrated, common)[0, 1]) > 0.98
    assert weights.shape == (5,)


def test_integrate_pca_fitted_on_train_rows_only():
    """The rotation must come from the training rows: if the tail is garbage,
    a component fitted on the head should be unaffected by it."""
    rng = np.random.default_rng(1)
    common = rng.standard_normal(2000)
    clean = np.column_stack([common + 0.05 * rng.standard_normal(2000)
                             for _ in range(4)])
    dirty = clean.copy()
    dirty[1400:] = rng.standard_normal((600, 4)) * 50.0
    _, w_train = of.integrate_pca(dirty, fit_rows=slice(0, 1400))
    _, w_clean = of.integrate_pca(clean[:1400])
    np.testing.assert_allclose(w_train, w_clean, atol=1e-9)


def test_transform_is_odd_and_concave():
    v = np.array([-400.0, -100.0, 0.0, 100.0, 400.0])
    got = of._transform(v, 0.5)
    np.testing.assert_allclose(got, [-20.0, -10.0, 0.0, 10.0, 20.0])


def synthetic_flow_bars(n=3000, seed=0, ofi_weight=0.0):
    """Returns driven by trade flow, optionally with an OFI component too."""
    rng = np.random.default_rng(seed)
    trade = rng.standard_normal(n) * 500.0
    ofi = rng.standard_normal(n) * 3.0
    ret = (1e-6 * of._transform(trade, 0.5)
           + ofi_weight * of._transform(ofi, 0.5)
           + 1e-8 * rng.standard_normal(n))
    return pd.DataFrame({"sec": np.arange(34200, 34200 + n), "signed_vol": trade,
                         "volume": np.abs(trade), "ofi_integrated": ofi,
                         "mid": 100.0 * np.exp(np.cumsum(ret))})


def test_compare_flows_credits_the_flow_that_drives_returns():
    frame = synthetic_flow_bars(ofi_weight=0.0)
    contemp = of.compare_flows(frame, "SYNTH")[0]
    assert contemp.relation == "contemporaneous"
    assert contemp.r2_trade > 0.95
    assert contemp.r2_ofi < 0.05
    assert contemp.incremental_ofi < 0.01          # OFI adds nothing here
    assert contemp.incremental_trade > 0.9         # trade adds everything


def test_compare_flows_splits_the_credit_when_both_matter():
    # weight chosen so the two flows contribute comparable variance to the
    # return; if one dominates the other's incremental R2 is correctly ~0
    frame = synthetic_flow_bars(ofi_weight=1.2e-5)
    contemp = of.compare_flows(frame, "SYNTH")[0]
    assert contemp.r2_both > contemp.r2_trade
    assert contemp.r2_both > contemp.r2_ofi
    assert contemp.incremental_ofi > 0.05
    assert contemp.incremental_trade > 0.05


def test_compare_flows_finds_no_predictive_power_in_contemporaneous_data():
    """Returns depend only on same-second flow, so the lagged regression must
    find nothing. This is the distinction the whole section rests on."""
    frame = synthetic_flow_bars()
    predictive = of.compare_flows(frame, "SYNTH")[1]
    assert predictive.relation == "predictive"
    assert abs(predictive.r2_trade) < 0.02
    assert abs(predictive.r2_ofi) < 0.02


def test_compare_flows_uses_a_fixed_delta_for_both_flows():
    frame = synthetic_flow_bars()
    result = of.compare_flows(frame, "SYNTH", delta=0.25)[0]
    assert result.delta_trade == 0.25
    assert result.delta_ofi == 0.25


def test_delta_sensitivity_ranks_the_true_concavity_first_out_of_sample():
    frame = synthetic_flow_bars()
    table = of.delta_sensitivity(frame, "signed_vol")
    best = table.loc[table.r2_out.idxmax(), "delta"]
    assert best == 0.5
    assert set(table.delta) == set(of.DELTA_GRID)
