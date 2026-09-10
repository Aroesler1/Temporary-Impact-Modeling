"""Execution algebra for typed price-level responses.

The empirical runner remains withdrawn. The reusable algebra now prevents the
three defects it exposed: raw return coefficients cannot enter a level cost,
the beyond-fit tail policy is explicit, and replay uses the exact same
quadratic objective matrix as optimization.

For a specified level kernel, kernel_matrix and optimal_schedule implement
M[s,t] = G(|t-s|) and the equality-constrained quadratic solution
x = S M^-1 1 / (1' M^-1 1). `replay_cost` uses that same quadratic impact
objective and adds only exogenous mid-price drift. These theoretical helpers do
not establish a valid empirical kernel or a schedule with no price manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kernel_response import LevelResponse
from propagator import build_lag_matrix

SCHEDULE_WITHDRAWAL = (
    "Empirical schedule withdrawn: fitted return coefficients are not a "
    "price-level impact kernel. Specify and validate the supported level "
    "horizon and execution-price convention before replaying. See "
    "docs/kernel_audit.md; reproduce the diagnostic with "
    "python scripts/audit_kernel_response.py --check."
)


def fit_linear_kernel(returns: np.ndarray, volume: np.ndarray, n_lags: int,
                      train_end: int) -> np.ndarray:
    """OLS RETURN coefficients b(0..L), fitted before `train_end`.

    The public function name is retained. Its output cannot be passed directly
    to `kernel_matrix`, which expects a price-level response.
    """
    design = build_lag_matrix(volume.astype(float), n_lags)
    ok = np.isfinite(design).all(axis=1) & np.isfinite(returns)
    ok[train_end:] = False
    kernel, *_ = np.linalg.lstsq(design[ok], returns[ok], rcond=None)
    return kernel


def select_lags(returns: np.ndarray, volume: np.ndarray, train_end: int,
                lag_grid=(1, 2, 5, 10, 20, 60)) -> int:
    """Pick L by out-of-sample R2 INSIDE the training window (nested split)."""
    inner = int(train_end * 0.7)
    best, best_r2 = lag_grid[0], -np.inf
    for n_lags in lag_grid:
        design = build_lag_matrix(volume.astype(float), n_lags)
        ok = np.isfinite(design).all(axis=1) & np.isfinite(returns)
        ok[train_end:] = False
        idx = np.flatnonzero(ok)
        tr, te = idx[idx < inner], idx[idx >= inner]
        if len(tr) < 100 or len(te) < 100:
            continue
        beta, *_ = np.linalg.lstsq(design[tr], returns[tr], rcond=None)
        resid = returns[te] - design[te] @ beta
        sst = float(np.sum((returns[te] - returns[te].mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else -np.inf
        if r2 > best_r2:
            best, best_r2 = n_lags, r2
    return best


@dataclass
class KernelMatrix:
    M: np.ndarray
    min_eigenvalue: float
    projected: bool
    response: LevelResponse


def kernel_matrix(
    response: LevelResponse,
    T: int,
    floor: float = 1e-12,
) -> KernelMatrix:
    """Build Toeplitz M from a typed level response and explicit tail policy."""
    if not isinstance(response, LevelResponse):
        raise TypeError(
            "kernel_matrix requires LevelResponse; convert fitted return "
            "coefficients with level_response_from_returns first"
        )
    kernel = np.asarray(response.for_horizon(T), dtype=float)
    lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    M = kernel[lags]
    M = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(M)
    lo = float(evals.min())
    scale = max(float(np.abs(evals).max()), 1.0)
    if lo < -floor * scale:
        clipped = np.clip(evals, floor * scale, None)
        M = evecs @ np.diag(clipped) @ evecs.T
        return KernelMatrix(M, lo, True, response)
    return KernelMatrix(M, lo, False, response)


def optimal_schedule(km: KernelMatrix, S: float) -> np.ndarray:
    """x* = S M^-1 1 / (1' M^-1 1), the Gatheral-Schied-Slynko solution."""
    ones = np.ones(km.M.shape[0])
    z = np.linalg.solve(km.M, ones)
    return S * z / float(ones @ z)


def twap(T: int, S: float) -> np.ndarray:
    return np.full(T, S / T)


def almgren_chriss(T: int, S: float, kappa: float) -> np.ndarray:
    """Classic AC trajectory with linear temporary impact and inventory risk.

    x_t proportional to sinh(kappa (T - t + 1/2)) - sinh(kappa (T - t - 1/2)).
    kappa = 0 is TWAP; larger kappa front-loads to shed inventory variance. This
    is the closed-form AC schedule rather than the repo's piecewise-model
    allocator, because this comparison is on the propagator's own one-second
    grid where the piecewise model has no depth curve to be allocated against.
    """
    if kappa <= 0:
        return twap(T, S)
    j = np.arange(T)
    trades = np.sinh(kappa * (T - j)) - np.sinh(kappa * (T - j - 1))
    return S * trades / trades.sum()


def model_cost(schedule: np.ndarray, km: KernelMatrix) -> float:
    """The exact quadratic objective minimized by :func:`optimal_schedule`."""
    values = np.asarray(schedule, dtype=float)
    if values.ndim != 1 or len(values) != km.M.shape[0]:
        raise ValueError("schedule length must match the kernel matrix")
    return 0.5 * float(values @ km.M @ values)


def replay_cost(
    schedule: np.ndarray,
    mid: np.ndarray,
    km: KernelMatrix,
) -> dict[str, float]:
    """Replay exogenous drift plus the same impact objective used to optimize.

    The impact term is the quadratic model cost per share, converted to price
    units at the arrival mid. This makes optimizer and replay rankings
    identical. It does not turn an observational fit into a causal model.
    """
    if not isinstance(km, KernelMatrix):
        raise TypeError("replay_cost requires the KernelMatrix used for optimization")
    schedule = np.asarray(schedule, dtype=float)
    mid = np.asarray(mid, dtype=float)
    if len(mid) < len(schedule):
        raise ValueError("mid path is shorter than the schedule")
    size = float(np.sum(schedule))
    if not np.isfinite(size) or size == 0.0:
        raise ValueError("schedule must have finite nonzero total size")
    weights = schedule / size
    drift = float(np.sum(weights * mid[: len(schedule)]) - mid[0])
    objective = model_cost(schedule, km)
    impact = float(mid[0] * objective / size)
    return {
        "total": drift + impact,
        "drift": drift,
        "impact": impact,
        "impact_objective": objective,
    }


@dataclass
class ReplayResult:
    session: str
    start_second: int
    order_fraction: float
    costs: dict[str, dict[str, float]]
    inventory_variance: dict[str, float]

    def saving_vs_twap(self, name: str, component: str = "total") -> float:
        return self.costs["TWAP"][component] - self.costs[name][component]


def replay_session(session: str, bars: pd.DataFrame, session_volume: float,
                   horizon: int = 600, n_starts: int = 40,
                   fractions=(0.005, 0.01, 0.02), train_frac: float = 0.7,
                   kappa: float = 0.005, seed: int = 0) -> list[ReplayResult]:
    """Refuse the withdrawn empirical return-to-level scheduling shortcut."""
    raise RuntimeError(SCHEDULE_WITHDRAWAL)


def bootstrap_saving(results: list[ReplayResult], name: str, n_boot: int = 2000,
                     seed: int = 0, component: str = "total"
                     ) -> tuple[float, float, float]:
    """Mean saving versus TWAP with a 95% band, resampling whole symbol-days."""
    values = np.array([r.saving_vs_twap(name, component) for r in results])
    sessions = np.array([r.session for r in results])
    rng = np.random.default_rng(seed)
    uniq = np.unique(sessions)
    by = {u: np.flatnonzero(sessions == u) for u in uniq}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[u] for u in pick])
        draws.append(float(values[idx].mean()))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)
