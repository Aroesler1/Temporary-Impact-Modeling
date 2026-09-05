"""Virtual cost of walking the displayed ladder, and the fitters for it.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
For a snapshot of the ten displayed levels, the virtual cost of a marketable
buy of x shares is the volume-weighted price it would pay minus the mid at the
instant before it arrives. It is a property of DISPLAYED liquidity at an
instant: no hidden size, no queue refill, no adverse selection, no time. It is
a LOWER BOUND on realised impact, and it lives in a different section of the
README from the metaorder results on purpose.

x is capped by displayed depth. A snapshot contributes to size x only if its
ten levels hold at least x shares, so the fit is valid strictly inside the
displayed-depth range and `bin_walk` records the participating fraction at
every size so the truncation is visible rather than assumed away.

That truncation is not neutral. At sizes near the top of the grid only the
deepest snapshots survive, and deep books are cheap to walk, so the measured
cost curve flattens and the fitted exponent is biased DOWN. `fit_all` reports
a restricted refit over the sizes where at least 90% of snapshots participate
so the size of that bias is a number rather than a caveat.

FORMS FITTED
------------
All by weighted nonlinear least squares on bin means with bin-count weights and
HC0 sandwich standard errors, which is the right variance estimator here
because bin variance falls with bin count and the plain OLS variance would be
badly wrong.

    linear        y = b u
    power         y = a u^p
    piecewise     y = c for u <= u0, c (u/u0)^p above          (the original)
    athl          y = eta u^beta, u = Q/ADV, y in units of sigma_D
                  (Almgren, Thum, Hauptmann and Li 2005; beta free, and 3/5)
    log           y = k log(1 + u/u0)

The piecewise form is the model `impact_model.py` calibrates. Written in
depth-relative units u = x/D_t the original is exactly this with u0 = 1, so
fitting u0 free measures whether the depth boundary is where the flat region
actually ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PRICE_SCALE = 1e-9  # Databento fixed-point prices are nanodollars


# --------------------------------------------------------------------------
# the walk itself
# --------------------------------------------------------------------------

def walk_vwap(prices: np.ndarray, sizes: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """VWAP of filling each x in `x_grid` against each ladder row.

    prices, sizes: (n_snapshots, n_levels), ordered outward from the touch.
    Returns (n_snapshots, n_x); NaN where x exceeds that row's displayed depth,
    which is the cap the module docstring describes.

    The marginal level is filled PARTIALLY -- x is a share count, not a level
    index, so the cost of 150 shares against levels of 100 and 200 pays 100 at
    the first price and 50 at the second, not 300 shares' worth of both.
    """
    prices = np.asarray(prices, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    cum_sz = np.cumsum(sizes, axis=1)
    cum_cost = np.cumsum(sizes * prices, axis=1)
    # level index whose cumulative size first reaches x
    idx = (cum_sz[:, :, None] < x_grid[None, None, :]).sum(axis=1)
    exhausted = idx >= sizes.shape[1]
    idx_safe = np.minimum(idx, sizes.shape[1] - 1)

    rows = np.arange(len(sizes))[:, None]
    prev_sz = np.where(idx_safe > 0, cum_sz[rows, np.maximum(idx_safe - 1, 0)], 0.0)
    prev_cost = np.where(idx_safe > 0, cum_cost[rows, np.maximum(idx_safe - 1, 0)], 0.0)
    marginal_px = prices[rows, idx_safe]

    total = prev_cost + (x_grid[None, :] - prev_sz) * marginal_px
    vwap = total / x_grid[None, :]
    vwap[exhausted] = np.nan
    return vwap


def walk_costs(book: pd.DataFrame, x_grid: np.ndarray, depth: int = 10) -> dict:
    """Signed virtual cost per share on both sides, in dollars, plus scales.

    `book` carries bid_px_NN / bid_sz_NN / ask_px_NN / ask_sz_NN in dollars and
    shares. Cost is measured from the MID, not from the touch: that is what
    makes the flat region of the piecewise model equal to the half-spread and
    what makes "cost in half-spreads" a meaningful unit. The original notebook
    measured the premium over the best ASK, which tends to zero as x tends to
    zero; `original_recipe` reproduces that quantity for comparison.
    """
    ask_px = book[[f"ask_px_{i:02d}" for i in range(depth)]].to_numpy(float)
    ask_sz = book[[f"ask_sz_{i:02d}" for i in range(depth)]].to_numpy(float)
    bid_px = book[[f"bid_px_{i:02d}" for i in range(depth)]].to_numpy(float)
    bid_sz = book[[f"bid_sz_{i:02d}" for i in range(depth)]].to_numpy(float)

    mid = (bid_px[:, 0] + ask_px[:, 0]) / 2.0
    half_spread = (ask_px[:, 0] - bid_px[:, 0]) / 2.0

    ask_vwap = walk_vwap(ask_px, ask_sz, x_grid)
    bid_vwap = walk_vwap(bid_px, bid_sz, x_grid)
    return {
        "buy_cost": ask_vwap - mid[:, None],       # $/share paid above mid
        "sell_cost": mid[:, None] - bid_vwap,      # $/share given up below mid
        "mid": mid,
        "half_spread": half_spread,
        "depth_l1_ask": ask_sz[:, 0],
        "depth_l1_bid": bid_sz[:, 0],
        "depth_10_ask": ask_sz.sum(axis=1),
        "depth_10_bid": bid_sz.sum(axis=1),
    }


def original_recipe(book: pd.DataFrame, depth: int = 10, cap: float = 1000.0
                    ) -> dict[str, float]:
    """Reproduce the notebook's fit, and the same fit without its filter.

    The notebook built (cumulative size, premium over best ask) pairs at the
    ten level boundaries, averaged them into INTEGER-share buckets, dropped
    every bucket above 1,000 shares, and fitted log premium on log size with
    every surviving bucket weighted equally regardless of how many snapshots
    it held. Three separate choices, so the effect of "filtering" is reported
    here as three numbers, not one.
    """
    ask_px = book[[f"ask_px_{i:02d}" for i in range(depth)]].to_numpy(float)
    ask_sz = book[[f"ask_sz_{i:02d}" for i in range(depth)]].to_numpy(float)
    cum_sz = np.cumsum(ask_sz, axis=1)
    cum_cost = np.cumsum(ask_sz * ask_px, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        premium = cum_cost / cum_sz - ask_px[:, [0]]

    x = cum_sz[:, 1:].ravel()
    g = premium[:, 1:].ravel()
    ok = np.isfinite(x) & np.isfinite(g) & (x > 0) & (g > 0)
    frame = pd.DataFrame({"x_int": x[ok].astype(np.int64), "g": g[ok]})
    buckets = frame.groupby("x_int").agg(g=("g", "mean"), n=("g", "size")).reset_index()

    def slope(sub: pd.DataFrame, weighted: bool) -> float:
        if len(sub) < 3:
            return float("nan")
        lx, ly = np.log(sub.x_int.to_numpy(float)), np.log(sub.g.to_numpy(float))
        w = sub.n.to_numpy(float) if weighted else np.ones(len(sub))
        p = np.polyfit(lx, ly, 1, w=np.sqrt(w))
        return float(p[0])

    kept = buckets[buckets.x_int <= cap]
    return {
        "p_original_filtered": slope(kept, weighted=False),
        "p_unfiltered": slope(buckets, weighted=False),
        "p_unfiltered_count_weighted": slope(buckets, weighted=True),
        "n_buckets_all": int(len(buckets)),
        "n_buckets_kept": int(len(kept)),
        "share_of_snapshot_points_dropped": float(
            1.0 - kept.n.sum() / buckets.n.sum()) if len(buckets) else float("nan"),
    }


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------

def bin_walk(size: np.ndarray, values, n_bins: int = 40) -> pd.DataFrame:
    """Equal-count bins in log size, carrying every value column's mean plus n.

    `values` is a mapping name -> array (a bare array is taken as "cost"). The
    count is the weight the fitters use; `cost_sd` is kept so a reader can see
    that dispersion, not just the mean, is what makes the bin weights right.

    Rows where ANY value column is not finite are dropped together, so every
    column in a bin is averaged over the same observations and the several
    normalisations stay comparable row by row.
    """
    if not isinstance(values, dict):
        values = {"cost": values}
    size = np.asarray(size, float)
    cols = {k: np.asarray(v, float) for k, v in values.items()}

    ok = np.isfinite(size) & (size > 0)
    for v in cols.values():
        ok &= np.isfinite(v)
    size = size[ok]
    cols = {k: v[ok] for k, v in cols.items()}
    if len(size) < n_bins * 2:
        raise ValueError(f"only {len(size)} usable points for {n_bins} bins")

    edges = np.unique(np.quantile(np.log(size), np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, np.log(size), side="right") - 1,
                  0, len(edges) - 2)
    frame = pd.DataFrame({"bin": idx, "size": size, **cols})
    agg = {"size": ("size", "mean"), "n": ("size", "size")}
    for k in cols:
        agg[k] = (k, "mean")
    first = next(iter(cols))
    agg["cost_sd"] = (first, "std")
    return frame.groupby("bin").agg(**agg).reset_index(drop=True)


# --------------------------------------------------------------------------
# candidate forms
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Form:
    """One candidate impact form: prediction, start values, and bounds."""
    name: str
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]
    p0: Callable[[np.ndarray, np.ndarray], np.ndarray]
    bounds: tuple[Sequence[float], Sequence[float]]
    param_names: tuple[str, ...]
    exponent_index: int | None = None  # which parameter is the concavity exponent


def _seed_power(u, y):
    p, loga = np.polyfit(np.log(u), np.log(np.maximum(y, 1e-12)), 1)
    return np.array([np.exp(loga), np.clip(p, 0.05, 1.5)])


LINEAR = Form(
    "linear",
    lambda th, u: th[0] * u,
    lambda u, y: np.array([float(np.mean(y / u))]),
    ([0.0], [np.inf]),
    ("b",),
)

POWER = Form(
    "power",
    lambda th, u: th[0] * u ** th[1],
    _seed_power,
    ([0.0, 0.0], [np.inf, 3.0]),
    ("a", "p"),
    exponent_index=1,
)

PIECEWISE = Form(
    "piecewise_flat_power",
    lambda th, u: np.where(u <= th[1], th[0], th[0] * (u / th[1]) ** th[2]),
    lambda u, y: np.array([max(float(np.min(y)), 1e-9),
                           float(np.exp(np.median(np.log(u)))),
                           float(np.clip(_seed_power(u, y)[1], 0.05, 1.5))]),
    ([0.0, 1e-9, 0.0], [np.inf, np.inf, 3.0]),
    ("c", "u0", "p"),
    exponent_index=2,
)

LOGARITHMIC = Form(
    "logarithmic",
    lambda th, u: th[0] * np.log1p(u / th[1]),
    lambda u, y: np.array([float(np.ptp(y) / max(np.log1p(np.ptp(u)), 1e-9)) or 1.0,
                           float(np.exp(np.median(np.log(u))))]),
    ([0.0, 1e-12], [np.inf, np.inf]),
    ("k", "u0"),
)


def athl(beta_fixed: float | None = None) -> Form:
    """Almgren, Thum, Hauptmann and Li (2005): I/sigma_D = eta (Q/V)^beta.

    Identical algebra to POWER; it is a separate entry because the paper fixes
    the units on BOTH sides (cost in daily volatility, size in average daily
    volume) and because beta = 3/5 is their published estimate, which is a
    hypothesis worth scoring rather than a starting value.
    """
    if beta_fixed is None:
        return Form("athl_beta_free", POWER.predict, POWER.p0, POWER.bounds,
                    ("eta", "beta"), exponent_index=1)
    return Form(
        f"athl_beta_{beta_fixed:g}",
        lambda th, u, b=beta_fixed: th[0] * u ** b,
        lambda u, y, b=beta_fixed: np.array([float(np.mean(y / u ** b))]),
        ([0.0], [np.inf]),
        ("eta",),
    )


ALL_FORMS: tuple[Form, ...] = (LINEAR, POWER, PIECEWISE, athl(None), athl(0.6),
                               LOGARITHMIC)


# --------------------------------------------------------------------------
# weighted nonlinear least squares with robust standard errors
# --------------------------------------------------------------------------

@dataclass
class WNLSFit:
    form: str
    params: np.ndarray
    param_names: tuple[str, ...]
    se_robust: np.ndarray
    wsse: float
    n_bins: int
    n_params: int
    r2_weighted: float
    aic: float
    exponent: float = float("nan")
    exponent_se: float = float("nan")

    def summary(self) -> str:
        bits = ", ".join(f"{n}={v:.4g}+-{s:.2g}"
                         for n, v, s in zip(self.param_names, self.params, self.se_robust))
        return f"{self.form:<22} {bits}   AIC {self.aic:8.2f}   R2w {self.r2_weighted:.4f}"


def _jacobian(form: Form, theta: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian; the piecewise form is not differentiable at
    u0, so an analytic gradient would need a special case for no real gain."""
    base = form.predict(theta, u)
    J = np.zeros((len(u), len(theta)))
    for k in range(len(theta)):
        step = 1e-6 * max(abs(theta[k]), 1e-6)
        up, dn = theta.copy(), theta.copy()
        up[k] += step
        dn[k] -= step
        J[:, k] = (form.predict(up, u) - form.predict(dn, u)) / (2 * step)
    del base
    return J


def fit_wnls(form: Form, u: np.ndarray, y: np.ndarray, w: np.ndarray) -> WNLSFit:
    """Weighted nonlinear least squares with an HC0 sandwich covariance.

    Weights are bin counts. The sandwich matters: bin means have variance
    proportional to 1/n and heteroskedastic dispersion on top of that, so the
    homoskedastic covariance would understate the exponent's error, which is
    exactly the number the identification question turns on.
    """
    u = np.asarray(u, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    sw = np.sqrt(w)

    res = least_squares(lambda th: sw * (y - form.predict(th, u)),
                        form.p0(u, y), bounds=form.bounds, method="trf",
                        xtol=1e-14, ftol=1e-14, max_nfev=20000)
    theta = res.x
    resid = y - form.predict(theta, u)
    wsse = float(np.sum(w * resid ** 2))

    J = _jacobian(form, theta, u)
    A = J.T @ (w[:, None] * J)
    B = J.T @ ((w ** 2 * resid ** 2)[:, None] * J)
    try:
        Ainv = np.linalg.pinv(A)
        cov = Ainv @ B @ Ainv
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(theta), np.nan)

    ybar = float(np.sum(w * y) / np.sum(w))
    sstot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 1.0 - wsse / sstot if sstot > 0 else float("nan")
    n, k = len(u), len(theta)
    # Gaussian AIC on the weighted residual sum of squares, bins as the sample
    aic = n * np.log(wsse / n) + 2 * k if wsse > 0 else -np.inf

    exp_i = form.exponent_index
    return WNLSFit(form.name, theta, form.param_names, se, wsse, n, k, r2, aic,
                   float(theta[exp_i]) if exp_i is not None else float("nan"),
                   float(se[exp_i]) if exp_i is not None else float("nan"))


# --------------------------------------------------------------------------
# model comparison, identification, uncertainty
# --------------------------------------------------------------------------

def loso_cv(form: Form, panel: pd.DataFrame) -> dict[str, float]:
    """Leave-one-symbol-day-out weighted RMSE over the 15 folds.

    A fold is a whole symbol-day, not a random subset of bins: bins inside one
    session share a book, so a random split would leak and every form would
    look better than it is.
    """
    errs, total_w = [], 0.0
    for held in panel["session"].unique():
        tr = panel[panel.session != held]
        te = panel[panel.session == held]
        try:
            fit = fit_wnls(form, tr.u.to_numpy(), tr.y.to_numpy(), tr.w.to_numpy())
        except Exception:
            return {"loso_rmse": float("nan"), "n_folds": 0}
        pred = form.predict(fit.params, te.u.to_numpy())
        errs.append(float(np.sum(te.w.to_numpy() * (te.y.to_numpy() - pred) ** 2)))
        total_w += float(te.w.sum())
    return {"loso_rmse": float(np.sqrt(np.sum(errs) / total_w)),
            "n_folds": int(panel["session"].nunique())}


def block_bootstrap_exponent(form: Form, panel: pd.DataFrame, n_boot: int = 400,
                             seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Resample whole symbol-days with replacement and refit the exponent."""
    if form.exponent_index is None:
        return float("nan"), float("nan"), np.empty(0)
    rng = np.random.default_rng(seed)
    groups = panel["session"].unique()
    by_group = {g: panel[panel.session == g] for g in groups}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([by_group[g] for g in pick], ignore_index=True)
        try:
            fit = fit_wnls(form, boot.u.to_numpy(), boot.y.to_numpy(), boot.w.to_numpy())
        except Exception:
            continue
        draws.append(fit.exponent)
    if not draws:
        return float("nan"), float("nan"), np.empty(0)
    arr = np.asarray(draws)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi), arr


def profile_exponent(form: Form, u: np.ndarray, y: np.ndarray, w: np.ndarray,
                     grid: np.ndarray) -> pd.DataFrame:
    """Weighted SSE with the exponent PINNED at each grid value.

    The curvature of this curve is the identification answer: a flat profile
    means the data cannot separate 0.4 from 0.6 whatever the point estimate
    says, and a bootstrap that is much wider than the profile interval means
    the binned observations are not independent, which they are not.
    """
    if form.exponent_index is None:
        raise ValueError(f"{form.name} has no exponent to profile")
    i = form.exponent_index
    rows = []
    for value in grid:
        lo = list(form.bounds[0])
        hi = list(form.bounds[1])
        lo[i], hi[i] = value - 1e-9, value + 1e-9
        pinned = Form(form.name, form.predict,
                      lambda uu, yy, v=value, f=form, j=i: np.clip(
                          _with(f.p0(uu, yy), j, v), np.array(f.bounds[0]) + 1e-12,
                          np.array(f.bounds[1])),
                      (lo, hi), form.param_names, None)
        try:
            fit = fit_wnls(pinned, u, y, w)
            rows.append({"exponent": float(value), "wsse": fit.wsse})
        except Exception:
            rows.append({"exponent": float(value), "wsse": float("nan")})
    out = pd.DataFrame(rows)
    out["delta_wsse"] = out.wsse - out.wsse.min()
    n = len(u)
    # Gaussian profile deviance: n log(SSE/SSE_min) is asymptotically chi2(1)
    out["deviance"] = n * np.log(out.wsse / out.wsse.min())
    return out


def _with(arr: np.ndarray, index: int, value: float) -> np.ndarray:
    out = np.asarray(arr, float).copy()
    out[index] = value
    return out


def fit_all(panel: pd.DataFrame, forms: Sequence[Form] = ALL_FORMS,
            n_boot: int = 400) -> pd.DataFrame:
    """Fit, cross-validate and bootstrap every candidate form on one panel.

    `panel` has columns session, u (normalised size), y (normalised cost),
    w (bin count).
    """
    u, y, w = panel.u.to_numpy(), panel.y.to_numpy(), panel.w.to_numpy()
    rows = []
    for form in forms:
        fit = fit_wnls(form, u, y, w)
        cv = loso_cv(form, panel)
        lo, hi, _ = block_bootstrap_exponent(form, panel, n_boot=n_boot)
        rows.append({
            "form": fit.form,
            "params": ", ".join(f"{n}={v:.4g}" for n, v in
                                zip(fit.param_names, fit.params)),
            "exponent": fit.exponent,
            "se_robust": fit.exponent_se,
            "boot_lo": lo, "boot_hi": hi,
            "wsse": fit.wsse, "r2_weighted": fit.r2_weighted,
            "aic": fit.aic, "n_params": fit.n_params,
            "loso_rmse": cv["loso_rmse"],
        })
    out = pd.DataFrame(rows).sort_values("loso_rmse").reset_index(drop=True)
    out["delta_aic"] = out.aic - out.aic.min()
    return out
