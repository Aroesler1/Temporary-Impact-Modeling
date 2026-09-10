"""Pure tests for offline consolidated CRSP normalisers."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import crsp


def names_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "REUSED",
                "permno": 1111,
                "namedt": "1990-01-01",
                "nameendt": "2010-12-31",
                "shrcd": 11,
                "exchcd": 1,
            },
            {
                "ticker": "REUSED",
                "permno": 2222,
                "namedt": "2011-01-01",
                "nameendt": "2100-12-31",
                "shrcd": 11,
                "exchcd": 1,
            },
            {
                "ticker": "TWO",
                "permno": 3333,
                "namedt": "2000-01-01",
                "nameendt": "2100-12-31",
                "shrcd": 12,
                "exchcd": 1,
            },
            {
                "ticker": "TWO",
                "permno": 4444,
                "namedt": "2000-01-01",
                "nameendt": "2100-12-31",
                "shrcd": 11,
                "exchcd": 1,
            },
        ]
    )


def test_permno_resolution_is_date_aware_and_prefers_share_code_11():
    old = crsp.resolve_permnos(names_fixture(), ["REUSED"], "2005-01-01")
    new = crsp.resolve_permnos(names_fixture(), ["REUSED"], "2024-06-28")
    two = crsp.resolve_permnos(names_fixture(), ["TWO"], "2024-06-28")
    assert int(old["permno"].iloc[0]) == 1111
    assert int(new["permno"].iloc[0]) == 2222
    assert int(two["permno"].iloc[0]) == 4444


def test_session_date_does_not_shift_utc_midnight_back_one_day():
    index = pd.to_datetime(["2024-08-01 00:00:00+00:00"])
    assert crsp.session_date(index).tolist() == ["2024-08-01"]


def test_trailing_volatility_excludes_the_current_return():
    rng = np.random.default_rng(0)
    daily = pd.DataFrame(
        {
            "permno": 1,
            "date": pd.bdate_range("2024-01-01", periods=50),
            "ret": rng.normal(0, 0.02, 50),
        }
    )
    base = crsp.trailing_volatility(daily)
    spiked = daily.copy()
    spiked.loc[30, "ret"] = 5.0
    changed = crsp.trailing_volatility(spiked)
    assert changed.loc[30, "sigma_crsp"] == pytest.approx(
        base.loc[30, "sigma_crsp"]
    )
    assert changed.loc[31, "sigma_crsp"] > base.loc[31, "sigma_crsp"]


def _cache_fixture(tmp_path):
    data = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "permno": [10001, 10001],
            "date": ["2024-07-01", "2024-07-02"],
            "volume_crsp": [1_000_000.0, 1_100_000.0],
            "price_crsp": [100.0, 101.0],
            "price_is_midpoint": [False, False],
            "sigma_crsp": [0.02, 0.021],
        }
    )
    data_path = tmp_path / "impact_consolidated_normalisers.csv.gz"
    data.to_csv(data_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "permno": [10001],
            "shrcd": [11],
            "exchcd": [1],
        }
    ).to_csv(tmp_path / "impact_resolved_permnos_verified.csv", index=False)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    summary = {
        "impact": {
            "daily_source": "crsp.dsf_v2",
            "name_source": "crsp.dsenames",
            "normaliser_sha256": digest,
            "median_crsp_over_equs": 1.0,
        }
    }
    (tmp_path / "offline_summary.json").write_text(json.dumps(summary))
    return data


def test_verified_cache_loads_with_units_and_exact_coverage(tmp_path):
    _cache_fixture(tmp_path)
    required = pd.DataFrame(
        {"symbol": ["AAA"], "date": ["2024-07-02"]}
    )
    out = crsp.load_consolidated_cache(
        ["AAA"],
        "2024-07-01",
        "2024-07-02",
        required_pairs=required,
        cache_dir=tmp_path,
    )
    assert len(out) == 2
    assert out.attrs["volume_unit"] == "shares"
    assert out.attrs["equs_validation_ratio"] == 1.0


def test_missing_consolidated_session_never_falls_back(tmp_path):
    _cache_fixture(tmp_path)
    required = pd.DataFrame(
        {"symbol": ["AAA"], "date": ["2024-07-03"]}
    )
    with pytest.raises(ValueError, match="Missing consolidated coverage"):
        crsp.load_consolidated_cache(
            ["AAA"],
            "2024-07-01",
            "2024-07-03",
            required_pairs=required,
            cache_dir=tmp_path,
        )


def test_hash_or_unit_validation_failure_is_fatal(tmp_path):
    _cache_fixture(tmp_path)
    summary_path = tmp_path / "offline_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["impact"]["median_crsp_over_equs"] = 0.1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="units"):
        crsp.load_consolidated_cache(
            ["AAA"], "2024-07-01", "2024-07-02", cache_dir=tmp_path
        )


def test_identifier_mapping_must_match_hash_checked_rows(tmp_path):
    _cache_fixture(tmp_path)
    mapping_path = tmp_path / "impact_resolved_permnos_verified.csv"
    mapping = pd.read_csv(mapping_path)
    mapping["permno"] = 99999
    mapping.to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="does not match consolidated rows"):
        crsp.load_consolidated_cache(
            ["AAA"], "2024-07-01", "2024-07-02", cache_dir=tmp_path
        )
