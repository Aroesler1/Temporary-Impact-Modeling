"""Tests for the one-second bar builder.

These pin the conventions that were recovered from the committed MSFT series,
because a silent change to any of them would move the published R^2 without
changing a line of the propagator. The bar grid, the fill sign, the exclusion of
hidden prints and the last-in-second mid are all load-bearing.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_1s_bars import RTH_CLOSE, RTH_OPEN, build  # noqa: E402


def _write(tmp_path, messages, book):
    m = tmp_path / "message.csv"
    b = tmp_path / "book.csv"
    pd.DataFrame(messages).to_csv(m, header=False, index=False)
    pd.DataFrame(book, columns=["timestamp", "bid_px_0", "bid_sz_0",
                                "ask_px_0", "ask_sz_0"]).to_csv(b, index=False)
    return m, b


def _msg(time, event_type, direction, size, order_id=1, price=1_000_000):
    return {"time": time, "event_type": event_type, "order_id": order_id,
            "size": size, "price": price, "direction": direction}


def test_fill_sign_is_the_negation_of_the_resting_side(tmp_path):
    # a fill's `direction` is the side of the RESTING order, so executing
    # against a resting sell (-1) is a BUY and must come out positive
    messages = [_msg(RTH_OPEN + 0.1, 4, -1, 100), _msg(RTH_OPEN + 1.1, 4, 1, 40)]
    book = [{"timestamp": RTH_OPEN + 0.1, "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100, "ask_sz_0": 5},
            {"timestamp": RTH_OPEN + 1.1, "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100, "ask_sz_0": 5}]
    frame = build(*_write(tmp_path, messages, book))
    assert list(frame.signed_vol) == [100.0, -40.0]


def test_hidden_prints_are_excluded_from_signed_volume(tmp_path):
    # type 5 moves no displayed liquidity; including it changes the series
    messages = [_msg(RTH_OPEN + 0.1, 4, -1, 100), _msg(RTH_OPEN + 0.2, 5, -1, 900)]
    book = [{"timestamp": RTH_OPEN + t, "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100, "ask_sz_0": 5} for t in (0.1, 0.2)]
    frame = build(*_write(tmp_path, messages, book))
    assert list(frame.signed_vol) == [100.0]


def test_second_with_messages_but_no_fill_is_kept_as_zero(tmp_path):
    # the grid is every second that carries a message, not every second that
    # carries a trade; dropping quiet seconds would shorten the return series
    messages = [_msg(RTH_OPEN + 0.1, 4, -1, 100), _msg(RTH_OPEN + 1.5, 1, 1, 10)]
    book = [{"timestamp": RTH_OPEN + t, "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100, "ask_sz_0": 5} for t in (0.1, 1.5)]
    frame = build(*_write(tmp_path, messages, book))
    assert list(frame.sec) == [RTH_OPEN, RTH_OPEN + 1]
    assert list(frame.signed_vol) == [100.0, 0.0]


def test_second_with_no_messages_is_absent_not_zero_filled(tmp_path):
    messages = [_msg(RTH_OPEN + 0.1, 4, -1, 100), _msg(RTH_OPEN + 2.5, 4, -1, 10)]
    book = [{"timestamp": RTH_OPEN + t, "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100, "ask_sz_0": 5} for t in (0.1, 2.5)]
    frame = build(*_write(tmp_path, messages, book))
    assert list(frame.sec) == [RTH_OPEN, RTH_OPEN + 2]


def test_mid_is_the_last_quote_in_the_second(tmp_path):
    messages = [_msg(RTH_OPEN + t, 1, 1, 10) for t in (0.1, 0.5, 0.9)]
    book = [{"timestamp": RTH_OPEN + 0.1, "bid_px_0": 100, "bid_sz_0": 5,
             "ask_px_0": 300, "ask_sz_0": 5},
            {"timestamp": RTH_OPEN + 0.5, "bid_px_0": 100, "bid_sz_0": 5,
             "ask_px_0": 500, "ask_sz_0": 5},
            {"timestamp": RTH_OPEN + 0.9, "bid_px_0": 100, "bid_sz_0": 5,
             "ask_px_0": 900, "ask_sz_0": 5}]
    frame = build(*_write(tmp_path, messages, book))
    assert len(frame) == 1
    assert frame.mid.iloc[0] == pytest.approx((100 + 900) / 2 / 10_000)


def test_messages_outside_rth_are_dropped(tmp_path):
    messages = [_msg(RTH_OPEN - 5.0, 4, -1, 100), _msg(RTH_OPEN + 0.1, 4, -1, 7),
                _msg(RTH_CLOSE + 5.0, 4, -1, 100)]
    book = [{"timestamp": t, "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100, "ask_sz_0": 5}
            for t in (RTH_OPEN - 5.0, RTH_OPEN + 0.1, RTH_CLOSE + 5.0)]
    frame = build(*_write(tmp_path, messages, book))
    assert list(frame.sec) == [RTH_OPEN]
    assert list(frame.signed_vol) == [7.0]


def test_mismatched_book_and_message_files_are_rejected(tmp_path):
    # a book built from a different conversion would silently misalign every
    # mid, so this must fail loudly rather than produce a plausible series
    messages = [_msg(RTH_OPEN + 0.1, 4, -1, 100), _msg(RTH_OPEN + 0.2, 4, -1, 100)]
    book = [{"timestamp": RTH_OPEN + 0.1, "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100, "ask_sz_0": 5}]
    with pytest.raises(SystemExit):
        build(*_write(tmp_path, messages, book))


# --------------------------------------------------------------------------
# --bin-ms: finer bars, with the one-second grid bit for bit unchanged
# --------------------------------------------------------------------------

def _fine_fixture(tmp_path):
    """Two fills 300 ms apart inside one second, and one in the next second."""
    messages = [_msg(RTH_OPEN + 0.10, 4, -1, 100),
                _msg(RTH_OPEN + 0.40, 4, -1, 40),
                _msg(RTH_OPEN + 1.25, 4, 1, 70)]
    book = [{"timestamp": m["time"], "bid_px_0": 999_900, "bid_sz_0": 5,
             "ask_px_0": 1_000_100 + 100 * i, "ask_sz_0": 5}
            for i, m in enumerate(messages)]
    return _write(tmp_path, messages, book)


def test_bin_ms_defaults_to_one_second_and_changes_nothing(tmp_path):
    """The default must reproduce the pre-option output exactly, including the
    integer dtype of `sec`: the committed series were written that way and a
    float column would rewrite every row of every file."""
    m, b = _fine_fixture(tmp_path)
    default = build(m, b)
    explicit = build(m, b, bin_ms=1000)
    pd.testing.assert_frame_equal(default, explicit)
    assert default.sec.dtype.kind == "i"
    assert list(default.sec) == [RTH_OPEN, RTH_OPEN + 1]
    # the two fills inside the first second are still summed into one row
    assert list(default.signed_vol) == [140.0, -70.0]
    assert list(default.volume) == [140.0, 70.0]


def test_bin_ms_splits_a_second_into_finer_bars(tmp_path):
    m, b = _fine_fixture(tmp_path)
    fine = build(m, b, bin_ms=100)
    assert list(fine.sec) == [RTH_OPEN + 0.1, RTH_OPEN + 0.4, RTH_OPEN + 1.2]
    assert list(fine.signed_vol) == [100.0, 40.0, -70.0]


def test_bin_ms_conserves_volume_across_widths(tmp_path):
    m, b = _fine_fixture(tmp_path)
    coarse = build(m, b, bin_ms=1000)
    fine = build(m, b, bin_ms=100)
    assert fine.volume.sum() == pytest.approx(coarse.volume.sum())
    assert fine.signed_vol.sum() == pytest.approx(coarse.signed_vol.sum())
    assert len(fine) >= len(coarse)


def test_bin_ms_keeps_the_last_mid_in_each_bin(tmp_path):
    """At one second the two early fills collapse to the LAST mid; at 100 ms
    they keep their own."""
    m, b = _fine_fixture(tmp_path)
    coarse = build(m, b, bin_ms=1000)
    fine = build(m, b, bin_ms=100)
    assert coarse.mid.iloc[0] == pytest.approx(fine.mid.iloc[1])
    assert fine.mid.iloc[0] != fine.mid.iloc[1]


def test_bin_ms_start_is_integral_only_when_the_bin_divides_a_second(tmp_path):
    m, b = _fine_fixture(tmp_path)
    assert build(m, b, bin_ms=2000).sec.dtype.kind == "i"
    assert build(m, b, bin_ms=250).sec.dtype.kind == "f"


def test_bin_ms_rejects_a_non_positive_width(tmp_path):
    m, b = _fine_fixture(tmp_path)
    with pytest.raises(SystemExit):
        build(m, b, bin_ms=0)
