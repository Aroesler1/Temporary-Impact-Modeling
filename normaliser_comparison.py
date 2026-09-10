"""Paired venue and consolidated normaliser comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crossover import fit_published

VARIANTS = (
    "venue_volume_venue_sigma",
    "consolidated_volume_venue_sigma",
    "venue_volume_consolidated_sigma",
    "consolidated_volume_consolidated_sigma",
)


def prepare_variants(
    orders: pd.DataFrame,
    consolidated: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create paired variants without changing the proxy-order sample."""
    required_orders = {
        "symbol",
        "date",
        "Q",
        "participation",
        "impact",
        "sigma_d",
    }
    missing = required_orders - set(orders.columns)
    if missing:
        raise ValueError(f"orders missing columns: {sorted(missing)}")
    required_crsp = {"symbol", "date", "volume_crsp", "sigma_crsp"}
    missing = required_crsp - set(consolidated.columns)
    if missing:
        raise ValueError(f"consolidated data missing columns: {sorted(missing)}")

    left = orders.copy()
    right = consolidated.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    merged = left.merge(
        right[["symbol", "date", "volume_crsp", "sigma_crsp"]],
        on=["symbol", "date"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_rows = merged[
        (merged["_merge"] != "both")
        | merged["volume_crsp"].isna()
        | merged["sigma_crsp"].isna()
    ]
    if not missing_rows.empty:
        preview = missing_rows[["symbol", "date"]].drop_duplicates().head(5)
        raise ValueError(
            f"{len(missing_rows)} proxy orders lack consolidated normalisers; "
            f"first sessions: {preview.to_dict('records')}"
        )
    merged = merged.drop(columns="_merge")
    venue_volume = pd.to_numeric(merged["Q"], errors="coerce") / pd.to_numeric(
        merged["participation"], errors="coerce"
    )
    if (
        (~np.isfinite(venue_volume))
        | (venue_volume <= 0)
        | (merged["volume_crsp"] <= 0)
        | (merged["sigma_d"] <= 0)
        | (merged["sigma_crsp"] <= 0)
    ).any():
        raise ValueError("normaliser units must be finite and positive")
    merged["volume_venue"] = venue_volume

    specs = {
        "venue_volume_venue_sigma": (
            merged["participation"],
            merged["sigma_d"],
        ),
        "consolidated_volume_venue_sigma": (
            merged["Q"] / merged["volume_crsp"],
            merged["sigma_d"],
        ),
        "venue_volume_consolidated_sigma": (
            merged["participation"],
            merged["sigma_crsp"],
        ),
        "consolidated_volume_consolidated_sigma": (
            merged["Q"] / merged["volume_crsp"],
            merged["sigma_crsp"],
        ),
    }
    result = {}
    for name, (participation, sigma) in specs.items():
        frame = merged.copy()
        frame["participation"] = np.asarray(participation, dtype=float)
        frame["fit_sigma"] = np.asarray(sigma, dtype=float)
        result[name] = frame
    return result


def fit_variants(
    variants: dict[str, pd.DataFrame],
    *,
    n_boot: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit every variant on identical rows, blocking intervals by session."""
    rows = []
    expected_index = None
    for offset, name in enumerate(VARIANTS):
        frame = variants[name]
        if expected_index is None:
            expected_index = frame.index
        elif not frame.index.equals(expected_index):
            raise ValueError("normaliser variants do not use identical proxy orders")
        groups = frame["symbol"].astype(str) + "_" + frame["date"].astype(str)
        fit = fit_published(
            frame,
            frame["fit_sigma"],
            n_boot=n_boot,
            seed=seed + offset,
            groups=groups,
        )
        rows.append(
            {
                "normaliser": name,
                "n_metaorders": fit.n_metaorders,
                "delta": fit.delta,
                "delta_lo": fit.delta_ci[0],
                "delta_hi": fit.delta_ci[1],
                "c_free": fit.c_free,
                "c_half": fit.c_half,
                "c_half_lo": fit.c_half_ci[0],
                "c_half_hi": fit.c_half_ci[1],
            }
        )
    return pd.DataFrame(rows)
