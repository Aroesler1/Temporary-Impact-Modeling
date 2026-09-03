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
