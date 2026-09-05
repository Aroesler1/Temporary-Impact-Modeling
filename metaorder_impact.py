"""Measure impact against participation rate on reconstructed metaorders.

The static book-walk elsewhere in this repository fits a concavity exponent to
the cost of consuming *displayed* liquidity at an instant. As the README states,
that quantity has no time dimension and is indexed by shares rather than by
participation, so it cannot test either square-root law. This module measures
the right quantity.

Measuring the right quantity is not the same as testing the law, and on this
data it is not enough. `crossover.py` locates the linear-to-square-root
crossover at about 2.8e-4 of daily volume, where impact is roughly 8 ticks.
Proxy metaorders built from same-signed fill runs sit four orders of magnitude
BELOW that, at impacts of a tenth of a tick, so the whole range lives inside the
price grid's own resolution. The exponent this module returns on them, 0.209 to
0.487 across fifteen symbol-days, is a measurement of that discreteness floor
rather than of either law.

Method, following the public-data metaorder approach (arXiv 2503.18199):

  1. A proxy metaorder is a maximal run of consecutive same-signed fills. Real
     metaorders are split into child orders that arrive as a burst of one-sided
     pressure, and on anonymous market-by-order data a run is the closest
     observable analogue. The proxy is honest about its limits: a run merges
     concurrent participants trading the same way, and splits one participant
     who pauses.
  2. Impact is the signed mid-price change from immediately before the run to
     immediately after it, normalised by daily volatility.
  3. Participation is the run's volume as a fraction of session volume.
  4. The exponent is fitted on BIN MEANS, not raw observations. Individual
     impacts are dominated by noise, and conditioning on positive impact -- an
     easy mistake -- selects on the dependent variable and biases the exponent.
     Binning across the participation range and averaging within bins is the
     standard presentation and avoids both.

The square-root law predicts an exponent of 0.5 in participation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ImpactLawFit:
    exponent: float
    intercept: float
    r_squared: float
    n_metaorders: int
    n_bins: int
    bins: pd.DataFrame

    # An exponent of 0.999 is linear for any practical purpose; requiring a
    # margin keeps floating-point noise from being reported as concavity.
    CONCAVITY_MARGIN = 0.95

    @property
    def is_concave(self) -> bool:
        return self.exponent < self.CONCAVITY_MARGIN

    def describe(self) -> str:
        # "steeper" was wrong in both directions: an exponent below 0.5 is MORE
        # concave than the square-root law, one above it is LESS. Naming the
        # direction matters when the number is the headline result.
        verdict = (
            "concave, consistent with the square-root law"
            if 0.4 <= self.exponent <= 0.6
            else "concave, and more so than the square-root law"
            if self.exponent < 0.4
            else "concave, but less so than the square-root law"
            if self.is_concave
            else "not concave"
        )
        return (f"exponent {self.exponent:.3f} (R^2 {self.r_squared:.3f}, "
                f"{self.n_metaorders:,} metaorders): {verdict}")


def compute_impact(frame: pd.DataFrame, session_volume: float, daily_vol: float) -> pd.DataFrame:
    """Signed normalised impact and participation for each metaorder."""
    if session_volume <= 0 or daily_vol <= 0:
        raise ValueError("session_volume and daily_vol must be positive")

    out = frame.copy()
    for col in ("sign", "shares", "mid_start", "mid_end"):
        if col not in out.columns:
            raise ValueError(f"missing column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[(out["mid_start"] > 0) & out["shares"].gt(0)].copy()
    out["impact"] = out["sign"] * (out["mid_end"] - out["mid_start"]) / out["mid_start"]
    out["impact_over_sigma"] = out["impact"] / daily_vol
    out["participation"] = out["shares"] / session_volume
    return out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["impact_over_sigma", "participation"])


def fit_impact_law(
    frame: pd.DataFrame,
    session_volume: float,
    daily_vol: float,
    n_bins: int = 12,
) -> ImpactLawFit:
    """Fit impact ~ participation^exponent on bin means."""
    data = compute_impact(frame, session_volume, daily_vol)
    if len(data) < n_bins * 10:
        raise ValueError("not enough metaorders for the requested binning")

    data = data.assign(bin=pd.qcut(data["participation"], n_bins, duplicates="drop"))
    grouped = data.groupby("bin", observed=True).agg(
        participation=("participation", "mean"),
        impact_over_sigma=("impact_over_sigma", "mean"),
        n=("impact_over_sigma", "size"),
    )
    # a bin whose MEAN impact is non-positive carries no information in log space;
    # dropping it is not selection on individual observations
    usable = grouped[grouped["impact_over_sigma"] > 0]
    if len(usable) < 4:
        raise ValueError("too few bins with positive mean impact to fit")

    x = np.log(usable["participation"].to_numpy())
    y = np.log(usable["impact_over_sigma"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    return ImpactLawFit(
        exponent=float(slope),
        intercept=float(intercept),
        r_squared=float(r2),
        n_metaorders=int(len(data)),
        n_bins=int(len(usable)),
        bins=grouped.reset_index(drop=True),
    )


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
