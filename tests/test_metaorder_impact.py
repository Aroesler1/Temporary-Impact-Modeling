"""Tests for the metaorder impact-law fit.

The estimator has one failure mode that matters more than any other: selecting
on the dependent variable. Conditioning on positive impact biases the exponent
substantially, and it is an easy mistake to make because it "cleans up" the
data. These tests pin that the fit does not do it, and that a known exponent is
recovered.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metaorder_impact import compute_impact, fit_impact_law  # noqa: E402

SESSION_VOLUME = 5_000_000.0
DAILY_VOL = 0.01


def _synthetic(exponent: float, n: int = 6000, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Metaorders whose mean impact follows a known power law in participation."""
    rng = np.random.default_rng(seed)
    shares = np.exp(rng.uniform(np.log(10), np.log(20000), n))
    participation = shares / SESSION_VOLUME
    mean_impact = 0.02 * DAILY_VOL * participation ** exponent
    impact = mean_impact + rng.normal(0, noise, n)
    mid_start = np.full(n, 400.0)
    sign = rng.choice([-1.0, 1.0], n)
    # invert compute_impact: mid_end is set so the signed impact is `impact`
    mid_end = mid_start * (1.0 + sign * impact)
    return pd.DataFrame({
        "sign": sign, "shares": shares,
        "mid_start": mid_start, "mid_end": mid_end,
    })


def test_recovers_a_known_exponent():
    frame = _synthetic(exponent=0.5)
    fit = fit_impact_law(frame, SESSION_VOLUME, DAILY_VOL)
    assert abs(fit.exponent - 0.5) < 0.05
    assert fit.r_squared > 0.95
    assert fit.is_concave


def test_recovers_a_linear_exponent_too():
    """A linear law must not be reported as concave."""
    fit = fit_impact_law(_synthetic(exponent=1.0), SESSION_VOLUME, DAILY_VOL)
    assert abs(fit.exponent - 1.0) < 0.05
    assert not fit.is_concave


def test_survives_noise_that_flips_individual_signs():
    """With noise large enough to make many single impacts negative, the BIN
    MEAN fit must still recover the exponent. This is the property that makes
    binning the right presentation rather than a cosmetic choice."""
    frame = _synthetic(exponent=0.5, noise=5e-6)
    data = compute_impact(frame, SESSION_VOLUME, DAILY_VOL)
    negative_share = float((data["impact"] < 0).mean())
    assert negative_share > 0.2, "fixture should contain many negative impacts"

    fit = fit_impact_law(frame, SESSION_VOLUME, DAILY_VOL)
    assert abs(fit.exponent - 0.5) < 0.15


def test_conditioning_on_positive_impact_biases_the_exponent():
    """Demonstrates the mistake the module exists to avoid.

    Keeping only positive-impact metaorders selects on the dependent variable
    and pulls the fitted exponent away from truth. If this test ever fails
    because the two agree, the estimator has probably started filtering.
    """
    frame = _synthetic(exponent=0.5, noise=5e-6)
    honest = fit_impact_law(frame, SESSION_VOLUME, DAILY_VOL)

    data = compute_impact(frame, SESSION_VOLUME, DAILY_VOL)
    survivors = data[data["impact"] > 0]
    biased = fit_impact_law(
        survivors.assign(mid_end=survivors["mid_start"] * (1 + survivors["sign"] * survivors["impact"])),
        SESSION_VOLUME, DAILY_VOL,
    )
    assert abs(biased.exponent - 0.5) > abs(honest.exponent - 0.5)


def test_rejects_degenerate_inputs():
    frame = _synthetic(exponent=0.5, n=50)
    with pytest.raises(ValueError):
        fit_impact_law(frame, SESSION_VOLUME, DAILY_VOL)
    with pytest.raises(ValueError):
        fit_impact_law(_synthetic(0.5), 0.0, DAILY_VOL)
