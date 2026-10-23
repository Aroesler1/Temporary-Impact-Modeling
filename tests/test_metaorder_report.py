"""The committed metaorder report uses the same session scales as every other script."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import panel
from metaorder_impact import fit_impact_law
from scripts.run_metaorder_impact import session_key, session_scales

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "MSFT_2024-06-03_metaorders.csv"
REPORT = ROOT / "reports" / "metaorder" / "impact_bins.csv"


def test_scales_come_from_the_session_file():
    scales = panel.scales("MSFT_2024-06-03")
    assert session_key(DATA) == "MSFT_2024-06-03"
    assert session_scales(DATA) == (float(scales.session_volume),
                                    float(scales.sigma_daily_20d))


def test_session_key_rejects_other_files():
    with pytest.raises(ValueError):
        session_key(ROOT / "data" / "MSFT_2024-06-03_1s.csv")


def test_committed_report_matches_a_rebuild():
    volume, sigma = session_scales(DATA)
    rebuilt = fit_impact_law(pd.read_csv(DATA), volume, sigma, n_bins=12).bins
    pd.testing.assert_frame_equal(pd.read_csv(REPORT), rebuilt,
                                  check_exact=False, rtol=1e-10)
