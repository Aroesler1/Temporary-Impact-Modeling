"""The panel loader: the joins and the split every other module depends on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import panel


def test_fifteen_sessions_on_three_names():
    meta = panel.meta()
    assert len(meta) == 15
    assert set(meta.symbol) == {"AAPL", "INTC", "MSFT"}
    assert meta.symbol.value_counts().to_dict() == {"AAPL": 5, "INTC": 5, "MSFT": 5}
    assert meta.date.str.startswith("2024").all()


def test_the_two_kept_sessions_are_still_in_the_panel():
    assert "MSFT_2024-06-03" in panel.session_keys()
    assert "INTC_2024-08-02" in panel.session_keys()


def test_bars_are_the_committed_series_untouched():
    """Part 4 needs a join to the OFI grid; nothing else may pay for it, or the
    propagator numbers stop matching the ones the README quotes."""
    bars = panel.bars("MSFT_2024-06-03")
    assert len(bars) == 23_390
    assert list(bars.columns) == ["sec", "signed_vol", "volume", "mid"]
    assert bars.sec.is_monotonic_increasing


def test_bars_with_ofi_is_an_inner_join_and_can_only_shrink():
    plain = panel.bars("MSFT_2024-06-03")
    joined = panel.bars_with_ofi("MSFT_2024-06-03")
    assert len(joined) <= len(plain)
    assert set(joined.sec).issubset(set(plain.sec))
    assert {"ofi_best", "ofi_integrated"}.issubset(joined.columns)
    assert joined[["ofi_best", "ofi_integrated"]].notna().all().all()


def test_unsigned_volume_bounds_signed_volume():
    for key in panel.session_keys():
        bars = panel.bars(key)
        assert (bars.volume >= bars.signed_vol.abs() - 1e-9).all()


def test_session_volume_exceeds_the_displayed_volume_it_contains():
    meta = panel.meta()
    assert (meta.session_volume > meta.displayed_volume).all()
    assert np.allclose(meta.session_volume,
                       meta.displayed_volume + meta.hidden_volume)


def test_metaorder_shares_sum_to_the_displayed_volume():
    for key in panel.session_keys():
        assert panel.metaorders(key).shares.sum() == pytest.approx(
            float(panel.scales(key).displayed_volume))


def test_returns_are_log_returns_with_a_leading_nan():
    frame = pd.DataFrame({"mid": [100.0, 101.0, 99.0]})
    got = panel.returns(frame)
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(np.log(1.01))
    assert got[2] == pytest.approx(np.log(99 / 101))


def test_bookwalk_panel_is_pooled_and_positive():
    frame = panel.bookwalk_panel()
    assert frame.session.nunique() == 15
    assert (frame.u > 0).all()
    assert np.isfinite(frame.y).all()
    assert (frame.w > 0).all()


def test_bookwalk_panel_participation_filter_only_removes_rows():
    everything = panel.bookwalk_panel()
    deep = panel.bookwalk_panel(min_participation=0.9)
    assert 0 < len(deep) < len(everything)
    assert deep.participating.min() >= 0.9


def test_split_index_is_the_same_seventy_percent_everywhere():
    assert panel.split_index(1000) == 700
    assert panel.TRAIN_FRAC == 0.7


def test_unknown_session_raises():
    with pytest.raises(KeyError):
        panel.scales("NVDA_2024-01-02")
