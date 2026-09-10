"""Distributed-lag RETURN response fitted to signed flow.

    r_t = sum_{l=0..L} b(l) * f(v_{t-l}) + noise

The retained public name `kernel` means return coefficients b, not the surviving
price-level displacement. The latter is the cumulative sum of b through the
identified horizon. Near-zero lagged b does not establish fast impact decay.

`fit_propagator` fits a fixed specification on the first window and scores its
tail. `calibrate` chooses the best specification USING that tail, so its winner
has a selected-validation score, not an untouched test score. Nested callers
must reserve their own outer evaluation window, as conditional_impact.py does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kernel_response import return_to_level_response


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
    kernel: np.ndarray            # return response b(0..L), not level impact
    r2_in: float
    r2_out: float
    n_train: int
    n_test: int

    @property
    def instantaneous(self) -> float:
        return float(self.kernel[0])

    @property
    def decay_half_life(self) -> float:
        """Level-response half-life on fitted support; nan if unobserved.

        Historical versions applied this threshold to return coefficients,
        incorrectly interpreting an absence of further returns as reversion.
        """
        # Predictive-only fits omit lag zero, so the initial level is unknown.
        if (self.kernel.size != self.n_lags + 1 or self.kernel.size < 2
                or self.kernel[0] == 0):
            return float("nan")
        level = np.asarray(return_to_level_response(self.kernel))
        target = abs(level[0]) / 2.0
        below = np.flatnonzero(np.abs(level[1:]) <= target)
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
        """Selected-validation R2 gain from lags over same-delta L=0."""
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
    """Select (delta, n_lags) on the scoring tail, which is validation data.

    No untouched test result is produced by this function. Do not label the
    selected winner's r2_out as unbiased OOS performance.
    """
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
