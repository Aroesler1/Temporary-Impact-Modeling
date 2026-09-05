"""Optimal execution under the FITTED decay kernel, replayed on held-out bars.

THE SOLUTION
------------
With a linear propagator the expected cost of a schedule x over T steps is the
quadratic form

    C(x) = 1/2 sum_{s,t} G(|t - s|) x_s x_t = 1/2 x' M x,   M[s,t] = G(|t - s|)

and minimising it subject to sum x = S has the closed-form solution of Gatheral,
Schied and Slynko (Mathematical Finance 22, 2012):

    x* = S M^{-1} 1 / (1' M^{-1} 1)

Obizhaeva and Wang (Journal of Financial Markets 16, 2013) is the special case
G(l) = G0 exp(-rho l), whose solution is the familiar bucket: a block at the
start, a constant rate in the middle, a block at the end. Nothing here assumes
that shape; M is built from the kernel actually fitted.

WHY delta IS FIXED AT 1 HERE
----------------------------
The propagator elsewhere in this repo uses f(v) = sign(v)|v|^delta and selects
delta by out-of-sample fit. The GSS solution above needs impact LINEAR in size,
so this module refits the kernel with delta = 1 on the same training window.
Deriving a schedule from a linear theory and then pricing it with a concave
kernel would be an inconsistency dressed up as a result.

NO-MANIPULATION IS CHECKED, NOT ASSUMED
---------------------------------------
The solution is a minimum only when M is positive definite. A fitted kernel need
not be: an empirical G with a sign change produces an indefinite M, which means
the model admits a round trip with negative expected cost -- price manipulation.
`kernel_matrix` reports the smallest eigenvalue, and when it is negative the
matrix is projected onto the positive-semidefinite cone before inversion, with
the projection reported rather than silently applied.

THE CIRCULARITY, STATED
-----------------------
The propagator prices the impact of the schedule it chose. That is unavoidable:
there is no counterfactual price path for an order that was never sent. It is
also why the comparison runs against HELD-OUT bars, and why the reported saving
is a saving under a model fitted on data the evaluation window does not contain.
It is not a claim about money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from propagator import build_lag_matrix


def fit_linear_kernel(returns: np.ndarray, volume: np.ndarray, n_lags: int,
                      train_end: int) -> np.ndarray:
    """OLS kernel G(0..L) with delta = 1, fitted on rows before `train_end`."""
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


def kernel_matrix(kernel: np.ndarray, T: int, floor: float = 1e-12) -> KernelMatrix:
    """Toeplitz M[s,t] = G(|t-s|), projected to PSD if the fit is indefinite."""
    L = len(kernel) - 1
    lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    M = np.where(lags <= L, kernel[np.minimum(lags, L)], 0.0)
    M = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(M)
    lo = float(evals.min())
    if lo <= 0:
        clipped = np.clip(evals, floor * float(evals.max()), None)
        M = evecs @ np.diag(clipped) @ evecs.T
        return KernelMatrix(M, lo, True)
    return KernelMatrix(M, lo, False)


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


def replay_cost(schedule: np.ndarray, mid: np.ndarray, kernel: np.ndarray
                ) -> dict[str, float]:
    """Cost per share of executing `schedule`, split into its two sources.

    The realised bars carry everything the market did without this order. The
    order's OWN displacement is priced by the fitted kernel on top:

        paid_t = mid_t * exp(sum_{s<=t} G(t-s) x_s)

    Splitting matters. `drift` is the realised move of the market between
    arrival and each fill, which every schedule is exposed to and none of them
    controls; over a ten-minute afternoon window it is an order of magnitude
    larger than any impact term and it is noise, not skill. `impact` is the part
    the model actually claims to optimise. A comparison reported only on the
    total is a comparison of which schedule got luckier about the drift.
    """
    T = len(schedule)
    own = np.convolve(schedule, kernel)[:T]
    weights = schedule / np.sum(schedule)
    drift = float(np.sum(weights * mid[:T]) - mid[0])
    impact = float(np.sum(weights * mid[:T] * np.expm1(own)))
    return {"total": drift + impact, "drift": drift, "impact": impact}


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
    """Replay every schedule at many start times inside the held-out tail."""
    mid = pd.to_numeric(bars["mid"], errors="coerce").to_numpy(float)
    vol = pd.to_numeric(bars["signed_vol"], errors="coerce").to_numpy(float)
    sec = bars["sec"].to_numpy(np.int64)
    ret = np.full(len(mid), np.nan)
    ret[1:] = np.log(mid[1:] / mid[:-1])

    train_end = int(len(bars) * train_frac)
    n_lags = select_lags(ret, vol, train_end)
    kernel = fit_linear_kernel(ret, vol, n_lags, train_end)
    km = kernel_matrix(kernel, horizon)

    rng = np.random.default_rng(seed)
    latest = len(bars) - horizon - 1
    if latest <= train_end:
        raise ValueError(f"{session}: held-out window shorter than the horizon")
    starts = rng.choice(np.arange(train_end, latest), size=min(n_starts,
                        latest - train_end), replace=False)

    results = []
    for start in np.sort(starts):
        window = mid[start:start + horizon]
        if not np.all(np.isfinite(window)) or window[0] <= 0:
            continue
        for fraction in fractions:
            S = fraction * session_volume
            schedules = {
                "TWAP": twap(horizon, S),
                "propagator_optimal": optimal_schedule(km, S),
                "almgren_chriss": almgren_chriss(horizon, S, kappa),
            }
            costs = {n: replay_cost(x, window, kernel) for n, x in schedules.items()}
            # remaining inventory variance, the risk side of the tradeoff AC
            # exists to buy; in units of the arrival price squared per second
            inventory = {n: float(np.sum((S - np.cumsum(x)) ** 2))
                         for n, x in schedules.items()}
            results.append(ReplayResult(session, int(sec[start]), fraction, costs,
                                        inventory))
    return results


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
