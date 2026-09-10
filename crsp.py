"""Offline CRSP consolidated normalisers and identifier validation.

Licensed rows stay in the external cache created by the single approved WRDS
session. This module never imports a vendor client and never opens a socket.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CRSP_COMMON_SHARE_CODES = (10, 11, 12, 18, 40, 41, 42, 48, 70, 71, 72)
SHARE_PRIORITY = {
    11: 0,
    10: 1,
    18: 2,
    12: 3,
    41: 4,
    40: 5,
    42: 6,
    48: 7,
    71: 8,
    70: 9,
    72: 10,
}
TRAILING_WINDOW = 20


def normalize_ticker(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def resolve_permnos(names: pd.DataFrame, tickers, as_of: str) -> pd.DataFrame:
    """Resolve one ordinary-share PERMNO per ticker at a point in time."""
    wanted = {normalize_ticker(ticker) for ticker in tickers}
    frame = names.copy()
    end_col = "nameenddt" if "nameenddt" in frame.columns else "nameendt"
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["permno"] = pd.to_numeric(frame["permno"], errors="coerce")
    frame["shrcd"] = pd.to_numeric(frame["shrcd"], errors="coerce")
    frame["exchcd"] = pd.to_numeric(
        frame.get("exchcd", np.nan), errors="coerce"
    ).fillna(99)
    frame["namedt"] = pd.to_datetime(frame["namedt"], errors="coerce").fillna(
        pd.Timestamp("1900-01-01")
    )
    frame[end_col] = pd.to_datetime(frame[end_col], errors="coerce").fillna(
        pd.Timestamp("2100-12-31")
    )
    stamp = pd.Timestamp(as_of)
    frame = frame[
        frame["ticker"].isin(wanted)
        & frame["shrcd"].isin(CRSP_COMMON_SHARE_CODES)
        & (frame["namedt"] <= stamp)
        & (frame[end_col] >= stamp)
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["ticker", "permno", "shrcd", "exchcd", "namedt", end_col]
        )
    frame["share_priority"] = frame["shrcd"].map(SHARE_PRIORITY).fillna(99)
    frame = frame.sort_values(
        ["ticker", "share_priority", "namedt", end_col, "exchcd", "permno"],
        ascending=[True, True, False, False, True, False],
    ).drop_duplicates("ticker")
    frame["permno"] = frame["permno"].astype(int)
    return frame[
        ["ticker", "permno", "shrcd", "exchcd", "namedt", end_col]
    ].reset_index(drop=True)


def session_date(index) -> np.ndarray:
    """Read Databento daily labels in UTC without shifting the session."""
    stamps = pd.to_datetime(index, utc=True)
    return np.asarray(pd.DatetimeIndex(stamps).strftime("%Y-%m-%d"))


def trailing_volatility(
    daily: pd.DataFrame,
    window: int = TRAILING_WINDOW,
) -> pd.DataFrame:
    """Close-to-close log-return volatility ending before the current session."""
    frame = daily.sort_values(["permno", "date"]).copy()
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    logret = np.log1p(frame["ret"])
    frame["sigma_crsp"] = logret.groupby(frame["permno"]).transform(
        lambda values: values.shift(1).rolling(
            window, min_periods=window
        ).std(ddof=1)
    )
    return frame


def consolidated_normalisers(
    daily: pd.DataFrame,
    window: int = TRAILING_WINDOW,
) -> pd.DataFrame:
    """Normalize raw CIZ daily rows, keeping volume in shares."""
    frame = trailing_volatility(daily, window=window)
    frame["volume_crsp"] = pd.to_numeric(
        frame.get("volume", frame.get("vol")), errors="coerce"
    )
    price = pd.to_numeric(
        frame.get("price", frame.get("prc")), errors="coerce"
    )
    frame["price_is_midpoint"] = price < 0
    frame["price_crsp"] = price.abs()
    frame["dollar_volume_crsp"] = frame["volume_crsp"] * frame["price_crsp"]
    return frame[
        [
            "permno",
            "date",
            "volume_crsp",
            "price_crsp",
            "price_is_midpoint",
            "sigma_crsp",
            "dollar_volume_crsp",
        ]
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_consolidated_cache(
    symbols,
    start: str,
    end: str,
    *,
    required_pairs: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load verified external normalisers, failing on missing coverage."""
    root_value = cache_dir or os.environ.get("IMPACT_CRSP_CACHE_DIR")
    if not root_value:
        raise FileNotFoundError(
            "Set IMPACT_CRSP_CACHE_DIR to the completed external WRDS cache."
        )
    root = Path(root_value)
    summary_path = root / "offline_summary.json"
    data_path = root / "impact_consolidated_normalisers.csv.gz"
    mapping_path = root / "impact_resolved_permnos_verified.csv"
    for path in (summary_path, data_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing consolidated cache artifact: {path.name}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = summary["impact"]
    if provenance.get("daily_source") != "crsp.dsf_v2":
        raise ValueError("Unexpected consolidated daily source.")
    if provenance.get("name_source") != "crsp.dsenames":
        raise ValueError("Unexpected identifier source.")
    if provenance.get("normaliser_sha256") != _sha256(data_path):
        raise ValueError("Consolidated normaliser hash mismatch.")
    ratio = provenance.get("median_crsp_over_equs")
    if ratio is None or not 0.9 <= float(ratio) <= 1.1:
        raise ValueError("CRSP volume units are not validated against EQUS.SUMMARY.")

    wanted = {normalize_ticker(symbol) for symbol in symbols}
    mapping = pd.read_csv(mapping_path)
    mapping["ticker"] = mapping["ticker"].map(normalize_ticker)
    mapping["permno"] = pd.to_numeric(
        mapping["permno"], errors="raise"
    ).astype(int)
    if mapping["ticker"].duplicated().any() or mapping["permno"].duplicated().any():
        raise ValueError("Identifier cache is not one-to-one.")
    unresolved = sorted(wanted - set(mapping["ticker"]))
    if unresolved:
        raise ValueError(f"Unresolved consolidated symbols: {unresolved}")

    frame = pd.read_csv(data_path)
    frame["symbol"] = frame["symbol"].map(normalize_ticker)
    frame["permno"] = pd.to_numeric(frame["permno"], errors="raise").astype(int)
    observed_mapping = frame[["symbol", "permno"]].drop_duplicates()
    expected_mapping = mapping.loc[
        mapping["ticker"].isin(wanted), ["ticker", "permno"]
    ].rename(columns={"ticker": "symbol"})
    mapping_check = expected_mapping.merge(
        observed_mapping,
        on=["symbol", "permno"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if (mapping_check["_merge"] != "both").any():
        raise ValueError("Identifier cache does not match consolidated rows.")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame[
        frame["symbol"].isin(wanted)
        & (frame["date"] >= pd.Timestamp(start))
        & (frame["date"] <= pd.Timestamp(end))
    ].copy()
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Consolidated cache has duplicate symbol/date rows.")
    bad_volume = frame["volume_crsp"].isna() | (frame["volume_crsp"] <= 0)
    if bad_volume.any():
        raise ValueError("Consolidated cache contains missing or nonpositive volume.")

    if required_pairs is not None:
        required = required_pairs.loc[:, ["symbol", "date"]].copy()
        required["symbol"] = required["symbol"].map(normalize_ticker)
        required["date"] = pd.to_datetime(required["date"]).dt.normalize()
        required = required.drop_duplicates()
        observed = frame[["symbol", "date"]].drop_duplicates()
        check = required.merge(
            observed,
            on=["symbol", "date"],
            how="left",
            indicator=True,
        )
        missing = check[check["_merge"] != "both"]
        if not missing.empty:
            preview = missing[["symbol", "date"]].head(5).to_dict("records")
            raise ValueError(
                f"Missing consolidated coverage for {len(missing)} required "
                f"symbol-date pairs; first rows: {preview}"
            )

    frame.attrs["daily_source"] = provenance["daily_source"]
    frame.attrs["name_source"] = provenance["name_source"]
    frame.attrs["volume_unit"] = "shares"
    frame.attrs["equs_validation_ratio"] = float(ratio)
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


@dataclass
class VenueComparison:
    per_name: pd.DataFrame
    outliers: pd.DataFrame


def compare_volumes(
    crsp: pd.DataFrame,
    vendor: pd.DataFrame,
    *,
    crsp_col: str = "volume_crsp",
    vendor_col: str = "volume_vendor",
    low: float = 0.9,
    high: float = 1.1,
) -> VenueComparison:
    merged = crsp.merge(vendor, on=["symbol", "date"], how="inner")
    merged = merged[(merged[crsp_col] > 0) & (merged[vendor_col] > 0)].copy()
    merged["ratio"] = merged[crsp_col] / merged[vendor_col]
    per_name = (
        merged.groupby("symbol")["ratio"]
        .agg(
            n_days="size",
            median="median",
            p05=lambda values: float(np.percentile(values, 5)),
            p95=lambda values: float(np.percentile(values, 95)),
        )
        .reset_index()
    )
    outliers = merged[(merged["ratio"] < low) | (merged["ratio"] > high)]
    return VenueComparison(
        per_name.sort_values("symbol").reset_index(drop=True),
        outliers.sort_values(["symbol", "date"]).reset_index(drop=True),
    )
