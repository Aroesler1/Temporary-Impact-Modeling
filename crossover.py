"""Where linear impact becomes square-root impact, and a published comparison.

THE CROSSOVER
-------------
The square-root law is not supposed to hold all the way down. Bucci, Benzaquen,
Lillo and Bouchaud (Physical Review Letters 123, 106401, 2019, arXiv 1901.05332)
argue that below some participation the relation must be LINEAR, because impact
has to be additive for infinitesimal orders, and that the crossover sits where
the metaorder size is comparable to the volume traded in the time it takes the
book to relax. Above it the square root takes over.

So the model fitted here is two regimes joined continuously,

    I(q) = a q                    for q <= q*
    I(q) = a sqrt(q* q)           for q >  q*

with q = Q / V_D. Two free parameters, a and q*, and q* is estimated by profile
likelihood rather than assumed. The linear branch has slope a; the square-root
branch has prefactor a sqrt(q*), so the two regimes share one scale and the
curve has no kink in level, only in slope.

WHY THE ANSWER MATTERS FOR THE 0.370 EXPONENT
---------------------------------------------
A single power law fitted across a range that straddles a real crossover
returns an exponent between the two regimes' slopes, so a genuine crossover
would explain an exponent below 0.5 without anything being wrong. The competing
explanation is a DISCRETENESS FLOOR: a metaorder of one fill still moves the mid
by about half a tick, so the smallest bins cannot fall below that level however
small the order, which lifts the left of the curve and flattens the fitted
slope in exactly the same direction.

The two are distinguishable, and `crossover_in_ticks` is the test. If the
fitted impact AT the crossover is below one tick, then the entire "linear"
regime lies inside the discreteness floor and the crossover is measuring the
floor, not Bucci's mechanism. If it is above one tick, the linear regime is a
real feature of the data.

THE PUBLISHED COMPARISON
------------------------
`bin_metaorders` implements the reconstruction of arXiv 2606.24019, which
confirms the square-root law on AAPL over 178 days: 30-second bins, direction
dominance above 0.3, duration at least 60 seconds, size at least 1e-4 of daily
volume, then I/sigma_D = c (Q/V_D)^delta. Their AAPL result is c_raw 0.69 with
[0.63, 0.77] across reconstruction settings, a bias-corrected c_eff of 0.34, and
delta 0.50 with a confidence interval of 0.32 to 0.66.

Only c_raw is reproducible here. The paper's abstract states the bias-corrected
prefactor but not the correction, so c_eff is quoted as theirs and not
recomputed; inventing a correction to land on 0.34 would be fitting to the
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BIN_SECONDS = 30
DOMINANCE_MIN = 0.3
MIN_DURATION_SECONDS = 60
MIN_SIZE_FRACTION = 1e-4


# --------------------------------------------------------------------------
# two-regime fit
# --------------------------------------------------------------------------

def two_regime(q: np.ndarray, a: float, q_star: float) -> np.ndarray:
    """Linear below the crossover, square-root above, continuous at q*."""
    q = np.asarray(q, float)
    return np.where(q <= q_star, a * q, a * np.sqrt(q_star * q))


@dataclass
class CrossoverFit:
    a: float
    q_star: float
    wsse: float
    r2_weighted: float
    n_bins: int
    profile: pd.DataFrame
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    at_grid_boundary: bool = False

    @property
    def impact_at_crossover(self) -> float:
        """I(q*) in units of daily volatility."""
        return self.a * self.q_star


def fit_two_regime(q: np.ndarray, impact: np.ndarray, weights: np.ndarray,
                   n_grid: int = 200) -> CrossoverFit:
    """Profile the weighted SSE over q*; a is linear given q*, so solved exactly.

    Profiling rather than joint optimisation because the objective in q* is not
    smooth -- q* moves observations between branches -- and a gradient method
    walks into a local minimum at whichever bin edge it starts near.
    """
    q = np.asarray(q, float)
    y = np.asarray(impact, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(q) & np.isfinite(y) & np.isfinite(w) & (q > 0)
    q, y, w = q[ok], y[ok], w[ok]

    grid = np.exp(np.linspace(np.log(q.min()), np.log(q.max()), n_grid))
    rows = []
    for q_star in grid:
        basis = np.where(q <= q_star, q, np.sqrt(q_star * q))
        denom = float(np.sum(w * basis ** 2))
        if denom <= 0:
            continue
        a = float(np.sum(w * basis * y) / denom)
        rows.append({"q_star": float(q_star), "a": a,
                     "wsse": float(np.sum(w * (y - a * basis) ** 2))})
    profile = pd.DataFrame(rows)
    best = profile.loc[profile.wsse.idxmin()]

    ybar = float(np.sum(w * y) / np.sum(w))
    sstot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 1.0 - float(best.wsse) / sstot if sstot > 0 else float("nan")

    # profile interval: n log(SSE / SSE_min) below the chi2(1) 95% point
    profile = profile.assign(
        deviance=len(q) * np.log(profile.wsse / profile.wsse.min()))
    inside = profile[profile.deviance <= 3.841]
    lo = float(inside.q_star.min()) if len(inside) else float("nan")
    hi = float(inside.q_star.max()) if len(inside) else float("nan")

    # q* pinned at an end of the search grid is not an estimate, it is the
    # optimiser saying the crossover is outside the observed range
    boundary = bool(np.isclose(best.q_star, grid[0]) or np.isclose(best.q_star, grid[-1]))
    return CrossoverFit(float(best.a), float(best.q_star), float(best.wsse),
                        r2, int(len(q)), profile, lo, hi, boundary)


def crossover_in_ticks(fit: CrossoverFit, sigma_d: float, mid: float,
                       tick: float = 0.01) -> dict[str, float]:
    """The crossover expressed the way a trader would ask for it.

    `impact_ticks` is the fitted impact AT the crossover measured in ticks. Below
    one tick means the whole linear branch lives inside the price grid's own
    resolution and cannot be read as Bucci's linear regime.
    """
    impact_sigma = fit.impact_at_crossover
    impact_price = impact_sigma * sigma_d * mid
    return {
        "q_star_fraction_of_daily_volume": fit.q_star,
        "q_star_ci_low": fit.ci_low,
        "q_star_ci_high": fit.ci_high,
        "impact_at_crossover_sigma": impact_sigma,
        "impact_at_crossover_dollars": impact_price,
        "impact_at_crossover_ticks": impact_price / tick,
        "half_tick_floor_ticks": 0.5,
        "above_one_tick": bool(impact_price / tick > 1.0),
        "q_star_at_grid_boundary": fit.at_grid_boundary,
    }


# --------------------------------------------------------------------------
# the published recipe
# --------------------------------------------------------------------------

def bin_metaorders(bars: pd.DataFrame, session_volume: float,
                   bin_seconds: int = BIN_SECONDS,
                   dominance_min: float = DOMINANCE_MIN,
                   min_duration: int = MIN_DURATION_SECONDS,
                   min_size_fraction: float = MIN_SIZE_FRACTION) -> pd.DataFrame:
    """Reconstruct metaorders on the arXiv 2606.24019 recipe.

    Trades are aggregated into fixed 30-second bins; a bin is DIRECTIONAL when
    |signed volume| / volume exceeds the dominance threshold; a metaorder is a
    maximal run of consecutive directional bins sharing a sign. Q is the run's
    net signed volume in absolute value, and impact is the signed log mid change
    from the start of the run's first bin to the end of its last.

    This is a different construction from `build_metaorders.py`, which takes a
    maximal same-signed run of individual FILLS. The bin recipe is coarser and
    much more selective: it keeps only sustained, one-sided, large pressure,
    which is why it reaches participation rates where the law is documented and
    the fill-run construction does not.
    """
    frame = bars.copy()
    frame["bin"] = (frame["sec"].to_numpy(np.int64) // bin_seconds) * bin_seconds
    grouped = frame.groupby("bin").agg(
        signed=("signed_vol", "sum"),
        volume=("volume", "sum"),
        mid_first=("mid", "first"),
        mid_last=("mid", "last"),
        n=("sec", "size")).reset_index()

    with np.errstate(divide="ignore", invalid="ignore"):
        grouped["dominance"] = np.abs(grouped.signed) / grouped.volume.replace(0, np.nan)
    grouped["sign"] = np.sign(grouped.signed)
    directional = (grouped.dominance >= dominance_min) & (grouped.sign != 0)

    # maximal runs of consecutive directional bins with the same sign; a gap in
    # the bin sequence also breaks a run, since the pressure was not sustained
    key = np.where(directional, grouped.sign, 0).astype(int)
    contiguous = grouped.bin.diff().fillna(bin_seconds).to_numpy() == bin_seconds
    new_run = (key != np.r_[0, key[:-1]]) | ~contiguous
    run_id = np.cumsum(new_run)
    grouped["run"] = run_id
    runs = grouped[directional].groupby("run").agg(
        sign=("sign", "first"),
        net=("signed", "sum"),
        gross=("volume", "sum"),
        n_bins=("bin", "size"),
        bin_start=("bin", "first"),
        mid_start=("mid_first", "first"),
        mid_end=("mid_last", "last")).reset_index(drop=True)

    runs["duration_seconds"] = runs.n_bins * bin_seconds
    runs["Q"] = runs.net.abs()
    runs["participation"] = runs.Q / session_volume
    runs["impact"] = runs.sign * (np.log(runs.mid_end) - np.log(runs.mid_start))
    keep = ((runs.duration_seconds >= min_duration)
            & (runs.participation >= min_size_fraction)
            & (runs.mid_start > 0) & (runs.mid_end > 0))
    return runs[keep].reset_index(drop=True)


@dataclass
class PublishedFit:
    c_free: float
    delta: float
    c_half: float
    n_metaorders: int
    delta_ci: tuple[float, float]
    c_half_ci: tuple[float, float]


def fit_published(orders: pd.DataFrame, sigma_d: pd.Series | float,
                  n_boot: int = 1000, seed: int = 0,
                  groups: pd.Series | None = None) -> PublishedFit:
    """I/sigma_D = c (Q/V_D)^delta, with delta free and with delta fixed at 1/2.

    Fitted on raw metaorders, not bin means, because the published comparison is
    a prefactor and a prefactor read off bin means depends on the binning. The
    bootstrap resamples whole SESSIONS when `groups` is given, since metaorders
    inside one day share a book and an independent-observation interval would be
    far too tight.
    """
    y = orders["impact"].to_numpy(float) / np.asarray(sigma_d, float)
    x = orders["participation"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    g = (groups.to_numpy()[ok] if groups is not None else np.zeros(len(x)))

    def fit_free(xx, yy):
        res = least_squares(lambda th: yy - th[0] * xx ** th[1],
                            x0=[0.5, 0.5], bounds=([0, 0], [np.inf, 3.0]))
        return float(res.x[0]), float(res.x[1])

    def fit_half(xx, yy):
        b = np.sqrt(xx)
        return float(np.sum(b * yy) / np.sum(b ** 2))

    c_free, delta = fit_free(x, y)
    c_half = fit_half(x, y)

    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    idx_by_group = {u: np.flatnonzero(g == u) for u in uniq}
    deltas, halves = [], []
    for _ in range(n_boot):
        if len(uniq) > 1:
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_group[u] for u in pick])
        else:
            idx = rng.integers(0, len(x), len(x))
        try:
            deltas.append(fit_free(x[idx], y[idx])[1])
            halves.append(fit_half(x[idx], y[idx]))
        except Exception:
            continue
    d_ci = tuple(np.percentile(deltas, [2.5, 97.5])) if deltas else (np.nan, np.nan)
    h_ci = tuple(np.percentile(halves, [2.5, 97.5])) if halves else (np.nan, np.nan)
    return PublishedFit(c_free, delta, c_half, int(len(x)),
                        (float(d_ci[0]), float(d_ci[1])),
                        (float(h_ci[0]), float(h_ci[1])))
