"""The 100 ms figure plots the corrected level response, not the withdrawn reading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_kernel_100ms import _figure, level_response

ROOT = Path(__file__).resolve().parent.parent
SHAPE = ROOT / "reports" / "kernel_100ms" / "kernel_shape.csv"
AUDIT = ROOT / "reports" / "kernel_audit" / "return_vs_level.csv"


def test_level_response_matches_the_committed_audit():
    shape = pd.read_csv(SHAPE)
    audit = pd.read_csv(AUDIT)
    np.testing.assert_allclose(level_response(shape.mean_G_over_G0),
                               audit.mean_level_response_over_initial, rtol=1e-12)


def test_level_response_of_an_isolated_trade_does_not_decay():
    # return response [1, 0, 0] leaves the price displaced: level [1, 1, 1]
    np.testing.assert_allclose(level_response([1.0, 0.0, 0.0]), [1.0, 1.0, 1.0])


def test_figure_titles_carry_no_withdrawn_interpretation(tmp_path):
    titles = _figure(pd.read_csv(SHAPE), None, 100, tmp_path / "kernel.png")
    assert (tmp_path / "kernel.png").stat().st_size > 0
    assert not any("memory" in title.lower() for title in titles)
    assert "level response" in titles[1].lower()
