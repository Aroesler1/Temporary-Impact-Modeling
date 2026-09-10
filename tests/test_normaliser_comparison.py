"""Independent synthetic checks for normaliser substitution."""
import numpy as np
import pandas as pd
import pytest

from normaliser_comparison import fit_variants, prepare_variants


def _fixture(n=500):
    rng = np.random.default_rng(4)
    q = np.exp(rng.uniform(np.log(100), np.log(100_000), n))
    venue_volume = 10_000_000.0
    venue_sigma = 0.02
    participation = q / venue_volume
    impact = 0.8 * venue_sigma * np.sqrt(participation)
    dates = pd.bdate_range("2024-04-01", periods=5).astype(str)
    orders = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": np.resize(dates, n),
            "Q": q,
            "participation": participation,
            "impact": impact,
            "sigma_d": venue_sigma,
        }
    )
    consolidated = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": dates,
            "volume_crsp": venue_volume * 2.0,
            "sigma_crsp": venue_sigma * 0.5,
        }
    )
    return orders, consolidated


def test_volume_and_volatility_are_substituted_in_separate_phases():
    orders, consolidated = _fixture()
    variants = prepare_variants(orders, consolidated)
    base = variants["venue_volume_venue_sigma"]
    volume_only = variants["consolidated_volume_venue_sigma"]
    sigma_only = variants["venue_volume_consolidated_sigma"]

    np.testing.assert_allclose(volume_only["fit_sigma"], base["fit_sigma"])
    np.testing.assert_allclose(
        volume_only["participation"], base["participation"] / 2.0
    )
    np.testing.assert_allclose(
        sigma_only["participation"], base["participation"]
    )
    np.testing.assert_allclose(sigma_only["fit_sigma"], base["fit_sigma"] / 2.0)


def test_all_fits_use_the_identical_proxy_order_count():
    orders, consolidated = _fixture()
    fits = fit_variants(prepare_variants(orders, consolidated), n_boot=0)
    assert fits["n_metaorders"].nunique() == 1
    assert int(fits["n_metaorders"].iloc[0]) == len(orders)


def test_missing_consolidated_session_is_fatal():
    orders, consolidated = _fixture()
    consolidated = consolidated.iloc[:-1]
    with pytest.raises(ValueError, match="lack consolidated normalisers"):
        prepare_variants(orders, consolidated)
