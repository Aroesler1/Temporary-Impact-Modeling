"""Tests for the proxy-metaorder builder.

Two of these exist because the file this replaces got them wrong, and both
errors moved the published exponent while leaving the output looking entirely
reasonable: fills sharing a timestamp were reordered, and the starting mid was
read after the run's first fill instead of before it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_metaorders import build  # noqa: E402


def _write(tmp_path, messages, book):
    m = tmp_path / "message.csv"
    b = tmp_path / "book.csv"
    pd.DataFrame(messages).to_csv(m, header=False, index=False, float_format="%.9f")
    pd.DataFrame(book, columns=["timestamp", "bid_px_0", "ask_px_0"]).to_csv(b, index=False)
    return m, b


def _msg(time, event_type, direction, size, price=1_000_000, order_id=1):
    return {"time": time, "event_type": event_type, "order_id": order_id,
            "size": size, "price": price, "direction": direction}


def _book(time, bid, ask):
    return {"timestamp": time, "bid_px_0": bid, "ask_px_0": ask}


def test_fill_sign_is_the_negation_of_the_resting_side(tmp_path):
    # executing against a resting sell (-1) is a BUY, so sign = +1
    messages = [_msg(1.0, 4, -1, 10), _msg(2.0, 4, 1, 20)]
    book = [_book(1.0, 999_900, 1_000_100), _book(2.0, 999_900, 1_000_100)]
    out = build(*_write(tmp_path, messages, book))
    assert list(out.sign) == [1.0, -1.0]


def test_a_run_is_maximal_and_breaks_only_on_a_sign_change(tmp_path):
    signs = [-1, -1, -1, 1, 1, -1]          # resting sides -> aggressor +,+,+,-,-,+
    messages = [_msg(float(i + 1), 4, d, 10) for i, d in enumerate(signs)]
    book = [_book(float(i + 1), 999_900, 1_000_100) for i in range(len(signs))]
    out = build(*_write(tmp_path, messages, book))
    assert list(out.n_fills) == [3, 2, 1]
    assert list(out.sign) == [1.0, -1.0, 1.0]
    assert list(out.shares) == [30, 20, 10]


def test_non_fill_messages_do_not_break_a_run(tmp_path):
    # an add or cancel between two same-signed fills is not a change of side
    messages = [_msg(1.0, 4, -1, 10), _msg(2.0, 1, 1, 50), _msg(3.0, 2, 1, 50),
                _msg(4.0, 4, -1, 10)]
    book = [_book(float(i + 1), 999_900, 1_000_100) for i in range(4)]
    out = build(*_write(tmp_path, messages, book))
    assert len(out) == 1 and out.n_fills.iloc[0] == 2


def test_mid_start_is_read_before_the_runs_first_fill(tmp_path):
    """The defect that mattered most in the file this replaces.

    Reading the mid AFTER the first fill drops that fill's own impact from every
    metaorder, which flattens small runs hardest and biases the fitted exponent.
    On the real session it moved the exponent from 0.37 to 0.62.
    """
    messages = [_msg(1.0, 1, 1, 10), _msg(2.0, 4, -1, 10), _msg(3.0, 4, -1, 10)]
    book = [_book(1.0, 1_000_000, 1_000_200),     # mid 100.01 -> before the run
            _book(2.0, 1_000_200, 1_000_400),     # mid 100.03 -> after fill 1
            _book(3.0, 1_000_400, 1_000_600)]     # mid 100.05 -> after fill 2
    m, b = _write(tmp_path, messages, book)

    before = build(m, b, mid_convention="before")
    assert before.mid_start.iloc[0] == pytest.approx(100.01)
    assert before.mid_end.iloc[0] == pytest.approx(100.05)

    at = build(m, b, mid_convention="at")
    assert at.mid_start.iloc[0] == pytest.approx(100.03)
    # the 'at' convention understates the run's impact by the first fill's move
    assert (at.mid_end.iloc[0] - at.mid_start.iloc[0]) < (
        before.mid_end.iloc[0] - before.mid_start.iloc[0])


def test_output_is_ordered_in_time(tmp_path):
    """The file this replaces contained a run starting 59 microseconds BEFORE
    the previous one ended, because fills sharing a timestamp were reordered.
    Message order is sequence order and must be preserved."""
    signs = [-1, -1, 1, -1, 1, 1, -1]
    times = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]   # heavy timestamp ties
    messages = [_msg(t, 4, d, 10) for t, d in zip(times, signs)]
    book = [_book(t, 999_900, 1_000_100) for t in times]
    out = build(*_write(tmp_path, messages, book))
    assert (np.diff(out.t_start.to_numpy()) >= 0).all(), "runs must not go backwards"
    assert (out.t_end.to_numpy()[:-1] <= out.t_start.to_numpy()[1:]).all(), \
        "runs must not overlap"


def test_mismatched_book_and_message_files_are_rejected(tmp_path):
    messages = [_msg(1.0, 4, -1, 10), _msg(2.0, 4, -1, 10)]
    book = [_book(1.0, 999_900, 1_000_100)]
    with pytest.raises(SystemExit):
        build(*_write(tmp_path, messages, book))


def test_one_sided_book_carries_the_last_known_mid(tmp_path):
    # dropping these rows instead would bias the sample toward calm periods
    messages = [_msg(1.0, 4, -1, 10), _msg(2.0, 1, 1, 5), _msg(3.0, 4, 1, 10)]
    book = [_book(1.0, 999_900, 1_000_100), _book(2.0, np.nan, 1_000_100),
            _book(3.0, 999_900, 1_000_100)]
    out = build(*_write(tmp_path, messages, book))
    assert np.isfinite(out.mid_start).all() and np.isfinite(out.mid_end).all()
