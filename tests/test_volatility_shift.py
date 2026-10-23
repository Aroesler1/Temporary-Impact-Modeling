"""The held-out volatility shift quoted in the detailed results is a committed table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_volatility_shift import per_session, summarise

REPORTS = Path(__file__).resolve().parent.parent / "reports" / "volatility_shift"


def test_committed_tables_match_a_rebuild():
    frame = per_session()
    pd.testing.assert_frame_equal(pd.read_csv(REPORTS / "per_session.csv"), frame,
                                  check_exact=False, rtol=1e-10)
    pd.testing.assert_frame_equal(pd.read_csv(REPORTS / "summary.csv"), summarise(frame),
                                  check_exact=False, rtol=1e-10)


def test_quoted_values_round_from_the_table():
    summary = pd.read_csv(REPORTS / "summary.csv").iloc[0]
    assert round(summary.median_test_over_train, 2) == 0.63
    assert round(summary.median_sqrt_slope, 2) == 0.52
    assert round(summary.pearson_ratio_slope, 2) == 0.68
    assert round(summary.spearman_ratio_slope, 2) == 0.17
