"""Temporary-impact modeling utilities: vectorized, normalized, tested.

This module replaces the exploratory notebook loops with vectorized
implementations and fixes two methodology issues:

1. Cross-symbol pooling. The notebook pooled (shares, $/share premium)
   pairs across symbols with different price levels, tick sizes, and depth
   scales, which biases the fitted exponent. Here premiums are normalized
   to basis points of the best ask and sizes to multiples of the symbol's
   median top-of-book depth before pooling, making the pooled fit
   scale-free. Per-symbol fits are also reported.
2. Uncertainty. `fit_power_law` reports a symbol-day block bootstrap
   confidence interval for the exponent instead of a bare point estimate.

Scope note (kept honest): premiums from walking displayed book snapshots
measure the VIRTUAL instantaneous cost of consuming visible liquidity.
Realized metaorder impact additionally reflects hidden liquidity, queue
refill, and adverse selection, and empirically scales with participation
of traded volume (the square-root law, impact ~ sigma * sqrt(Q/V)). The
piecewise model calibrated here is a displayed-liquidity lower bound, not
a fitted metaorder impact curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ASK_SZ_COLS = [f"ask_sz_{i:02d}" for i in range(10)]
ASK_PX_COLS = [f"ask_px_{i:02d}" for i in range(10)]


def first_nonzero_ask_depth(df: pd.DataFrame) -> pd.Series:
    """Size at the first non-empty ask level per row (vectorized).

    Replaces the notebook's row-wise ``DataFrame.apply`` loop.
    """
    sizes = df[ASK_SZ_COLS].to_numpy(dtype=float)
    positive = sizes > 0
    first_idx = np.argmax(positive, axis=1)
    has_any = positive.any(axis=1)
    out = sizes[np.arange(len(sizes)), first_idx]
    out[~has_any] = 0.0
    return pd.Series(out, index=df.index, name="depth")


def walk_book_premiums(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative-size vs average-premium pairs from ask-side book walks.

    Vectorized over all rows and levels (the notebook used a per-row,
    per-level Python loop). For each snapshot and each level L>=1:

        size_L    = sum of ask sizes for levels 0..L
        premium_L = VWAP(levels 0..L) - best ask     [$ per share]

    Output columns:
        symbol, size_shares, premium_dollars,
        size_rel (multiples of the symbol's median L1 depth),
        premium_bps (premium / best ask * 1e4)
    """
    sizes = df[ASK_SZ_COLS].to_numpy(dtype=float)
    prices = df[ASK_PX_COLS].to_numpy(dtype=float)
    base = prices[:, [0]]

    cum_sz = np.cumsum(sizes, axis=1)
    cum_cost = np.cumsum(sizes * prices, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        premium = cum_cost / cum_sz - base

    n_rows, n_levels = sizes.shape
    rows = np.repeat(np.arange(n_rows), n_levels - 1)
    lvls = np.tile(np.arange(1, n_levels), n_rows)

    out = pd.DataFrame(
        {
            "symbol": df["symbol"].to_numpy()[rows],
            "size_shares": cum_sz[rows, lvls],
            "premium_dollars": premium[rows, lvls],
            "best_ask": base[rows, 0],
        }
    )
    out = out[(out["premium_dollars"] > 0) & (out["size_shares"] > 0) & (out["best_ask"] > 0)]

    # per-symbol scale normalization
    med_depth = (
        pd.Series(first_nonzero_ask_depth(df).to_numpy(), index=df["symbol"].to_numpy())
        .groupby(level=0)
        .median()
    )
    out["size_rel"] = out["size_shares"] / out["symbol"].map(med_depth).replace(0, np.nan)
    out["premium_bps"] = out["premium_dollars"] / out["best_ask"] * 1e4
    return out.dropna(subset=["size_rel", "premium_bps"]).reset_index(drop=True)


@dataclass
class PowerLawFit:
    exponent: float
    log_intercept: float
    r_squared: float
    ci_low: float = np.nan
    ci_high: float = np.nan
    n_obs: int = 0


def fit_power_law(
    sizes: pd.Series,
    premiums: pd.Series,
    groups: Optional[pd.Series] = None,
    n_boot: int = 500,
    seed: int = 0,
) -> PowerLawFit:
    """OLS fit of log(premium) = log(a) + p * log(size).

    When ``groups`` is given (e.g. symbol or symbol-day labels), a block
    bootstrap over groups gives a 95% CI for the exponent that respects
    within-group dependence of book snapshots.
    """
    x = np.log(pd.to_numeric(sizes, errors="coerce"))
    y = np.log(pd.to_numeric(premiums, errors="coerce"))
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return PowerLawFit(np.nan, np.nan, np.nan, n_obs=int(len(x)))

    p, loga = np.polyfit(x, y, 1)
    resid = y - (loga + p * x)
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    ci_low = ci_high = np.nan
    if groups is not None:
        g = groups[mask].to_numpy()
        uniq = np.unique(g)
        if len(uniq) >= 3:
            rng = np.random.default_rng(seed)
            xv, yv = x.to_numpy(), y.to_numpy()
            idx_by_group = {u: np.flatnonzero(g == u) for u in uniq}
            boots = []
            for _ in range(n_boot):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([idx_by_group[u] for u in pick])
                pb, _ = np.polyfit(xv[idx], yv[idx], 1)
                boots.append(pb)
            ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    return PowerLawFit(float(p), float(loga), float(r2), float(ci_low), float(ci_high), int(len(x)))


def allocate_schedule(
    Dt: np.ndarray,
    c: float,
    p: float,
    S: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """Optimal minute allocation for total cost sum_t x_t * g_t(x_t).

    Piecewise temporary impact, continuous at the depth boundary:
        g_t(x) = c                for 0 <= x <= D_t
        g_t(x) = a_t x^p          for x > D_t, with a_t = c / D_t^p

    Total per-minute cost is convex (marginal cost steps from c up to
    (1+p)c at D_t, then increases), so KKT with a bisection on the shared
    marginal cost lambda is globally optimal. Inside the flat region the
    optimum is degenerate; the D_t-proportional fill is used as the
    tie-break, matching the intraday liquidity profile.
    """
    Dt = np.asarray(Dt, dtype=float)
    if np.any(Dt <= 0):
        raise ValueError("Dt must be strictly positive")
    if S <= 0:
        return np.zeros_like(Dt)

    total_flat = Dt.sum()
    if S <= total_flat:
        return Dt * (S / total_flat)

    a = c / Dt**p

    def alloc(lam: float) -> np.ndarray:
        # marginal cost of the tail is (1+p) a x^p = lam
        xtail = (lam / (a * (1.0 + p))) ** (1.0 / p)
        return np.maximum(Dt, xtail)

    lam_lo = (1.0 + p) * c                     # marginal at the boundary
    lam_hi = (1.0 + p) * float(a.max()) * S**p  # loose upper bound
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if alloc(lam_mid).sum() > S:
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid
        if lam_hi - lam_lo < tol * max(lam_hi, 1.0):
            break

    x = alloc(0.5 * (lam_lo + lam_hi))
    # exact feasibility: scale the tail overflow onto the flat-region names
    return x * (S / x.sum())


def schedule_cost(x: np.ndarray, Dt: np.ndarray, c: float, p: float) -> float:
    """Total cost sum_t x_t * g_t(x_t) for the piecewise impact model."""
    x = np.asarray(x, dtype=float)
    Dt = np.asarray(Dt, dtype=float)
    a = c / Dt**p
    g = np.where(x <= Dt, c, a * np.maximum(x, 1e-300) ** p)
    return float((x * g).sum())
