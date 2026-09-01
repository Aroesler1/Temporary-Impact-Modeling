"""Transient-impact propagator calibrated on real order flow.

The piecewise model elsewhere in this repo is MEMORYLESS: the cost of trading at
minute t depends only on size at t. Real impact decays rather than vanishing, so
a trade moves the price and that move relaxes over the following seconds. The
propagator model of Bouchaud and co-authors captures this,

    r_t  =  sum_{l=0..L} G(l) * f(v_{t-l})  +  noise

where v is signed traded volume, f is a concavity transform, and G is the
propagator kernel: G(0) is instantaneous impact and G(l>0) is what survives l
periods later.

This module calibrates G directly from data rather than assuming it, and answers
two questions the snapshot book-walk elsewhere in this repo structurally cannot:

1. DOES HISTORY MATTER? Comparing a memoryless model (L = 0) against one with
   lags, out of sample, tests whether transient impact is real in this data or
   whether the instantaneous model already suffices.

2. IS IMPACT CONCAVE IN THE TIME-SERIES SENSE? The README notes the fitted
   exponent from walking a static book is a cross-sectional depth property and
   cannot speak to the square-root law. Here f(v) = sign(v)*|v|^delta is fitted
   over a grid of delta, so concavity is estimated from the flow itself.

Calibration follows the standard recipe: regress returns on lagged signed
volumes across multiple lags and choose kernel parameters by out-of-sample R^2
on a chronological split. Nothing here is fitted on the evaluation window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def signed_flow(volume: np.ndarray, delta: float) -> np.ndarray:
    """f(v) = sign(v) * |v|^delta, the concavity transform.

    delta = 1 is linear impact; delta = 0.5 is the square-root form. Applying it
    to signed volume keeps the sign and compresses magnitude.
    """
    return np.sign(volume) * np.abs(volume) ** delta


def build_lag_matrix(flow: np.ndarray, n_lags: int) -> np.ndarray:
    """Design matrix whose column l is the flow lagged by l periods."""
    n = len(flow)
    out = np.full((n, n_lags + 1), np.nan)
    for lag in range(n_lags + 1):
        out[lag:, lag] = flow[: n - lag] if lag else flow
    return out


@dataclass
class PropagatorFit:
    delta: float
    n_lags: int
    kernel: np.ndarray            # G(0..L)
    r2_in: float
    r2_out: float
    n_train: int
    n_test: int

    @property
    def instantaneous(self) -> float:
        return float(self.kernel[0])

    @property
    def decay_half_life(self) -> float:
        """Lags until |G| first falls below half of |G(0)|; nan if it never does."""
        if self.kernel.size < 2 or self.kernel[0] == 0:
            return float("nan")
        target = abs(self.kernel[0]) / 2.0
        below = np.flatnonzero(np.abs(self.kernel[1:]) < target)
        return float(below[0] + 1) if below.size else float("nan")


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_propagator(
    returns: np.ndarray,
    volume: np.ndarray,
    n_lags: int,
    delta: float,
    train_frac: float = 0.7,
    drop_contemporaneous: bool = False,
) -> PropagatorFit:
    """OLS kernel on a chronological train split, scored on the held-out tail.

    `drop_contemporaneous` removes the l=0 column, turning an EXPLANATORY
    regression into a strictly PREDICTIVE one. The distinction is the whole
    game: flow during second t mechanically moves the mid during second t, so a
    high contemporaneous R^2 says impact exists, not that it is tradeable.
    """
    flow = signed_flow(volume, delta)
    design = build_lag_matrix(flow, n_lags)
    if drop_contemporaneous:
        if n_lags < 1:
            raise ValueError("predictive fit needs at least one lag")
        design = design[:, 1:]

    valid = np.isfinite(design).all(axis=1) & np.isfinite(returns)
    X, y = design[valid], returns[valid]
    split = int(len(y) * train_frac)
    if split < 50 or len(y) - split < 50:
        raise ValueError("not enough observations either side of the split")

    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]
    kernel, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    return PropagatorFit(
        delta=delta,
        n_lags=n_lags,
        kernel=kernel,
        r2_in=_r2(ytr, Xtr @ kernel),
        r2_out=_r2(yte, Xte @ kernel),
        n_train=len(ytr),
        n_test=len(yte),
    )


@dataclass
class CalibrationReport:
    best: PropagatorFit
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    memoryless: PropagatorFit | None = None

    @property
    def history_gain(self) -> float:
        """Out-of-sample R^2 improvement from lags over the memoryless model."""
        if self.memoryless is None:
            return float("nan")
        return self.best.r2_out - self.memoryless.r2_out


def calibrate(
    frame: pd.DataFrame,
    lag_grid=(0, 1, 2, 5, 10, 20, 60),
    delta_grid=(0.25, 0.5, 0.75, 1.0),
    train_frac: float = 0.7,
    drop_contemporaneous: bool = False,
) -> CalibrationReport:
    """Select (delta, n_lags) by out-of-sample R^2, never in-sample fit."""
    mid = pd.to_numeric(frame["mid"], errors="coerce").to_numpy(dtype=float)
    vol = pd.to_numeric(frame["signed_vol"], errors="coerce").to_numpy(dtype=float)
    # log returns keep the scale comparable across the session
    returns = np.full_like(mid, np.nan)
    returns[1:] = np.log(mid[1:] / mid[:-1])

    rows, fits = [], {}
    for delta in delta_grid:
        for n_lags in lag_grid:
            if drop_contemporaneous and n_lags < 1:
                continue
            try:
                fit = fit_propagator(returns, vol, n_lags, delta, train_frac,
                                     drop_contemporaneous)
            except (ValueError, np.linalg.LinAlgError):
                continue
            fits[(delta, n_lags)] = fit
            rows.append({
                "delta": delta, "n_lags": n_lags,
                "r2_in": fit.r2_in, "r2_out": fit.r2_out,
                "G0": fit.instantaneous, "half_life": fit.decay_half_life,
            })

    if not rows:
        raise RuntimeError("no configuration fitted")
    grid = pd.DataFrame(rows).sort_values("r2_out", ascending=False).reset_index(drop=True)
    best_key = (grid.iloc[0]["delta"], int(grid.iloc[0]["n_lags"]))
    best = fits[best_key]

    # memoryless comparison at the SAME delta, so the contrast isolates lags
    memoryless = fits.get((best_key[0], 0))
    return CalibrationReport(best=best, grid=grid, memoryless=memoryless)


def load(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"mid", "signed_vol"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return frame
