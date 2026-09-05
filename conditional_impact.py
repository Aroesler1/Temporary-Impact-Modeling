"""Conditional impact accuracy: the number an execution desk actually uses.

THREE DIFFERENT R-SQUAREDS
--------------------------
1. CONTEMPORANEOUS R2 regresses the price change over a bin on flow in the SAME
   bin. It describes. It cannot be traded, because the flow is not known until
   the bin is over.
2. PREDICTIVE R2 regresses the price change on PAST flow. It could be traded.
   In this data it is near zero.
3. CONDITIONAL IMPACT ACCURACY asks the question an execution model exists to
   answer: given an order of this size, executed on this schedule, how far will
   the price move, and how close was the model's answer to what happened. It is
   measured out of sample on orders the model never saw.

A desk cares about (2) for alpha and (3) for cost. Neither is (1), which is the
number impact studies usually lead with.

WHAT THIS MODULE MEASURES
-------------------------
Every reconstructed metaorder that starts inside the held-out last 30% of a
session gets a predicted impact from a propagator whose kernel AND
hyperparameters were fitted strictly inside the first 70%, using only the
order's OWN signed volume:

    I_pred = sum_{t in [t_start, t_end]} sum_l G(l) f(v_own[t - l])

with v_own zero outside the order's own seconds. Realised impact is the signed
log mid change across the order, the same quantity `metaorder_impact` fits.

Two honest limits, both structural rather than fixable:

* f is concave, so applying it to the order's own volume alone under-predicts
  whenever the second also carried someone else's flow. That is the price of
  attributing impact to one participant on anonymous data at all.
* the proxy metaorder is a same-signed run, so "the order's own flow" is the
  run's flow, which may merge two participants.

The square-root benchmark is I = c sigma_D sqrt(Q / V), with c fitted on the
TRAINING window's metaorders and applied unchanged out of sample.

That is not yet an equal comparison, because the square-root model gets a level
fitted on metaorders and the propagator gets a level fitted on one-second bars,
which is a different object. So a THIRD model is scored: the propagator's
prediction multiplied by a single scalar fitted on the training window's
metaorders. It has exactly one free level parameter, like the square-root
model, and differs from it only in SHAPE. Comparing the two isolates whether
the kernel's shape carries information, separately from whether its level
transfers from bar flow to order flow, which it does not.

WHICH VOLATILITY, AND WHETHER THE RATE MATTERS
----------------------------------------------
The square-root benchmark holds sigma at a daily constant while the market it
prices gets quieter through the session, so it over-predicts the afternoon by
about a factor of two. Replacing sigma_D with realised volatility over the half
hour before each order over-corrects the other way. Four more models are scored
against the same held-out orders, all with their parameters fitted on the
training window only:

    sqrt_blend        sigma_D^alpha * sigma_trail^(1-alpha), alpha fitted
    sqrt_tod          sigma_D times a half-hour time-of-day multiplier
    sqrt_rate         c (Q/V)^delta (1 + k log(rate)), the participation-rate
                      term of Zarinelli, Treccani, Farmer and Lillo (2015,
                      arXiv 1412.2152), with rate = Q over the volume traded
                      during the order's own execution window

The time-of-day multiplier cannot come from the scored session's own training
window, because the training window is the first 70% of the day and every
held-out order starts in a half hour the training window never reaches. It is
estimated from OTHER sessions of the SAME symbol instead, and both variants are
reported: `loso` uses the symbol's other four sessions, which is cross-validated
but not strictly causal since some donors are later days, and `prior` uses only
sessions before the scored one, which is strictly causal but leaves the earliest
session of each symbol without a donor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from propagator import build_lag_matrix, fit_propagator, signed_flow

DELTA_GRID = (0.25, 0.5, 0.75, 1.0)
LAG_GRID = (0, 1, 2, 5, 10, 20, 60)


@dataclass
class Calibration:
    """A propagator fitted strictly inside a training window."""
    delta: float
    n_lags: int
    kernel: np.ndarray
    n_train: int
    inner_r2_out: float


def calibrate_on_window(bars: pd.DataFrame, train_end: int,
                        delta_grid=DELTA_GRID, lag_grid=LAG_GRID) -> Calibration:
    """Choose (delta, lags) and fit G, using only rows before `train_end`.

    A NESTED split: hyperparameters are selected on the last 30% of the TRAINING
    window, then the kernel is refitted on the whole training window. The
    existing `propagator.calibrate` selects on the evaluation window, which is
    fine for the descriptive R2 it reports and not fine here, where the
    evaluation window has to stay untouched.
    """
    train = bars.iloc[:train_end]
    mid = pd.to_numeric(train["mid"], errors="coerce").to_numpy(float)
    vol = pd.to_numeric(train["signed_vol"], errors="coerce").to_numpy(float)
    returns = np.full_like(mid, np.nan)
    returns[1:] = np.log(mid[1:] / mid[:-1])

    best, best_r2 = None, -np.inf
    for delta in delta_grid:
        for n_lags in lag_grid:
            try:
                fit = fit_propagator(returns, vol, n_lags, delta, train_frac=0.7)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if fit.r2_out > best_r2:
                best, best_r2 = (delta, n_lags), fit.r2_out
    if best is None:
        raise ValueError("no propagator configuration fitted on the training window")

    delta, n_lags = best
    flow = signed_flow(vol, delta)
    design = build_lag_matrix(flow, n_lags)
    ok = np.isfinite(design).all(axis=1) & np.isfinite(returns)
    kernel, *_ = np.linalg.lstsq(design[ok], returns[ok], rcond=None)
    return Calibration(delta, n_lags, kernel, int(ok.sum()), float(best_r2))


def predict_propagator(orders: pd.DataFrame, cal: Calibration) -> np.ndarray:
    """Impact each order's own flow implies under the fitted kernel.

    Returned IN THE ORDER'S OWN DIRECTION, matching `realised_impact`: a sell
    whose price falls has positive impact, not negative. The raw convolution is
    signed by the flow, so a sell comes out negative and comparing it against a
    direction-normalised realisation would score the model against its own sign
    convention rather than against the data.

    Vectorised over orders. An order that spans D seconds spreads its volume
    evenly over them, which is the only schedule the reconstruction supports:
    a proxy metaorder carries its fills' total size and its start and end, not
    a per-second breakdown.
    """
    delta = cal.delta
    G = cal.kernel
    L = len(G) - 1

    signed_shares = orders["sign"].to_numpy(float) * orders["shares"].to_numpy(float)
    duration = np.maximum(
        np.floor(orders["t_end"].to_numpy(float)).astype(np.int64)
        - np.floor(orders["t_start"].to_numpy(float)).astype(np.int64) + 1, 1)

    out = np.zeros(len(orders))
    for d in np.unique(duration):
        sel = duration == d
        per_second = signed_shares[sel] / d
        # price change accumulated over the order's own d seconds, under a
        # constant execution rate starting at second 0:
        #   sum_{t<d} sum_{lag<=min(L,t)} G(lag)  =  sum_{lag<=min(L,d-1)} (d-lag) G(lag)
        lags = np.arange(min(L, int(d) - 1) + 1)
        coeff = float(np.sum((d - lags) * G[lags]))
        out[sel] = coeff * signed_flow(per_second, delta)
    # back into the order's own direction
    return out * orders["sign"].to_numpy(float)


def realised_impact(orders: pd.DataFrame) -> np.ndarray:
    """Signed log mid change across the order, the quantity the model predicts."""
    mid_start = orders["mid_start"].to_numpy(float)
    mid_end = orders["mid_end"].to_numpy(float)
    return orders["sign"].to_numpy(float) * (np.log(mid_end) - np.log(mid_start))


def trailing_sigma(bars: pd.DataFrame, at_second: np.ndarray,
                   window_seconds: int = 1800) -> np.ndarray:
    """Realised volatility over the `window_seconds` BEFORE each order starts.

    Scaled to a daily figure so it is a drop-in replacement for the 20-day
    close-to-close sigma. Strictly causal: the window ends at the second before
    the order's first fill, so nothing the order itself did is in its own
    normaliser.

    This exists because the constant daily sigma is the wrong scale intraday.
    Volatility falls through the session, so a coefficient fitted on the morning
    is fitted on a more volatile market than the afternoon it is applied to.
    """
    sec = bars["sec"].to_numpy(np.int64)
    ret = np.full(len(bars), np.nan)
    mid = pd.to_numeric(bars["mid"], errors="coerce").to_numpy(float)
    ret[1:] = np.log(mid[1:] / mid[:-1])
    sq = np.nan_to_num(ret ** 2)

    cum = np.concatenate([[0.0], np.cumsum(sq)])
    counts = np.arange(len(sq) + 1, dtype=float)
    end = np.searchsorted(sec, at_second, side="left")            # exclusive
    start = np.searchsorted(sec, at_second - window_seconds, side="left")
    n = np.maximum(counts[end] - counts[start], 1.0)
    var_per_second = (cum[end] - cum[start]) / n
    # a full RTH session is 23,400 seconds; scaling by its length makes this
    # comparable with a close-to-close daily sigma
    daily = np.sqrt(np.maximum(var_per_second, 0.0) * 23_400.0)
    fallback = float(np.nanmedian(daily[np.isfinite(daily) & (daily > 0)])) \
        if np.any(np.isfinite(daily) & (daily > 0)) else np.nan
    daily[~np.isfinite(daily) | (daily <= 0)] = fallback
    return daily


def fit_sqrt_coefficient(orders: pd.DataFrame, session_volume: float,
                         sigma_d: float | np.ndarray) -> float:
    """Least-squares c in I = c sigma_D sqrt(Q / V), fitted in sample.

    Fitted on raw orders rather than bin means, because this coefficient is
    being used as a POINT predictor and has to be the one that minimises point
    error. The binned estimator in `metaorder_impact` answers a different
    question, the shape of the law, and selecting bins there is right.
    """
    x = np.asarray(sigma_d, float) * np.sqrt(
        orders["shares"].to_numpy(float) / session_volume)
    y = realised_impact(orders)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2))


def predict_sqrt(orders: pd.DataFrame, c: float, session_volume: float,
                 sigma_d: float | np.ndarray) -> np.ndarray:
    return c * np.asarray(sigma_d, float) * np.sqrt(
        orders["shares"].to_numpy(float) / session_volume)


HALF_HOUR = 1800
RTH_OPEN_SEC = 34_200


def blend_sigma(sigma_d: float | np.ndarray, sigma_trail: np.ndarray,
                alpha: float) -> np.ndarray:
    """Geometric blend sigma_D^alpha * sigma_trail^(1-alpha).

    Geometric rather than arithmetic because the two inputs differ by a factor
    that is roughly constant in log space: the trailing one-second estimate is
    systematically below the close-to-close one at every hour, so a geometric
    blend moves smoothly between the two levels while an arithmetic one would be
    dominated by whichever is larger.
    """
    return np.asarray(sigma_d, float) ** alpha * np.asarray(sigma_trail, float) ** (
        1.0 - alpha)


def fit_blend_alpha(orders: pd.DataFrame, session_volume: float,
                    sigma_d: float, sigma_trail: np.ndarray,
                    n_grid: int = 101) -> tuple[float, float]:
    """alpha and c minimising the training SSE, alpha on a grid in [0, 1].

    c is linear given alpha, so it is profiled out exactly at each grid point
    and only alpha is searched. A grid rather than a gradient step because the
    objective is cheap and the gradient in alpha is nearly flat near the
    optimum, which is itself part of the result.
    """
    y = realised_impact(orders)
    q = np.sqrt(orders["shares"].to_numpy(float) / session_volume)
    best = (float("nan"), float("nan"), np.inf)
    for alpha in np.linspace(0.0, 1.0, n_grid):
        x = blend_sigma(sigma_d, sigma_trail, alpha) * q
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        denom = float(np.sum(x[ok] ** 2))
        if denom <= 0:
            continue
        c = float(np.sum(x[ok] * y[ok]) / denom)
        sse = float(np.sum((y[ok] - c * x[ok]) ** 2))
        if sse < best[2]:
            best = (float(alpha), c, sse)
    return best[0], best[1]


def halfhour_bucket(at_second: np.ndarray) -> np.ndarray:
    """Which half hour of the regular session a time falls in, 0 at the open."""
    return ((np.asarray(at_second, float) - RTH_OPEN_SEC) // HALF_HOUR).astype(int)


def halfhour_ratios(bars: pd.DataFrame) -> pd.Series:
    """Realised one-second volatility in each half hour, over the whole day's.

    A shape, not a level: the ratio is dimensionless, so it can be carried from
    one session of a symbol to another without carrying that session's level.
    """
    sec = bars["sec"].to_numpy(float)
    mid = pd.to_numeric(bars["mid"], errors="coerce").to_numpy(float)
    ret = np.full(len(mid), np.nan)
    ret[1:] = np.log(mid[1:] / mid[:-1])
    whole = float(np.nanstd(ret))
    if not np.isfinite(whole) or whole <= 0:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({"bucket": halfhour_bucket(sec), "ret": ret}).dropna()
    return frame.groupby("bucket").ret.std() / whole


def profile_sigma(sigma_d: float, profile: pd.Series,
                  at_second: np.ndarray) -> np.ndarray:
    """sigma_D scaled by the time-of-day multiplier for each order's half hour.

    A half hour with no donor coverage falls back to 1.0, which is the same as
    having no profile at all for that order rather than a silent gap.
    """
    buckets = halfhour_bucket(at_second)
    factor = np.array(profile.reindex(buckets).to_numpy(float), copy=True)
    factor[~np.isfinite(factor) | (factor <= 0)] = 1.0
    return sigma_d * factor


def execution_rate(orders: pd.DataFrame, bars: pd.DataFrame) -> np.ndarray:
    """Q over the total volume traded during the order's own execution window.

    The window is the seconds the order spans, and the denominator is the
    unsigned displayed volume the bars record over exactly those seconds. So
    rate is at most 1, reached when the order was the only thing that traded,
    and small when it was a minor part of a busy stretch.
    """
    sec = bars["sec"].to_numpy(np.int64)
    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(float)
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(volume))])
    lo = np.searchsorted(sec, np.floor(orders["t_start"].to_numpy(float)), "left")
    hi = np.searchsorted(sec, np.floor(orders["t_end"].to_numpy(float)), "right")
    window = cum[np.clip(hi, 0, len(cum) - 1)] - cum[np.clip(lo, 0, len(cum) - 1)]
    shares = orders["shares"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(window > 0, shares / window, np.nan)
    # a run can overshoot its own second's tally at the boundary of the bar grid
    return np.clip(rate, 1e-6, 1.0)


@dataclass
class RateFit:
    c: float
    delta: float
    k: float
    n_train: int


def fit_rate_model(orders: pd.DataFrame, session_volume: float, sigma_d: float,
                   rate: np.ndarray) -> RateFit:
    """I/sigma = c (Q/V)^delta (1 + k log rate), fitted by least squares.

    Zarinelli, Treccani, Farmer and Lillo (2015, arXiv 1412.2152) find that
    impact depends on the rate of execution as well as the size, with a
    logarithmic correction to the power law. k is bounded so the correction
    factor stays positive across the training range: outside that bound the
    model predicts impact in the wrong direction for the fastest orders, which
    is not a fit, it is a sign error the optimiser is free to walk into.
    """
    from scipy.optimize import least_squares

    y = realised_impact(orders) / sigma_d
    q = orders["shares"].to_numpy(float) / session_volume
    lr = np.log(np.asarray(rate, float))
    ok = np.isfinite(y) & np.isfinite(q) & np.isfinite(lr) & (q > 0)
    y, q, lr = y[ok], q[ok], lr[ok]

    span = float(np.max(np.abs(lr))) if len(lr) else 1.0
    k_bound = 0.95 / max(span, 1e-9)

    def residual(theta):
        c, delta, k = theta
        return y - c * q ** delta * (1.0 + k * lr)

    res = least_squares(residual, x0=[1.0, 0.5, 0.0],
                        bounds=([0.0, 0.0, -k_bound], [np.inf, 3.0, k_bound]),
                        xtol=1e-14, ftol=1e-14, max_nfev=20000)
    return RateFit(float(res.x[0]), float(res.x[1]), float(res.x[2]), int(len(y)))


def predict_rate_model(orders: pd.DataFrame, fit: RateFit, session_volume: float,
                       sigma_d: float, rate: np.ndarray) -> np.ndarray:
    q = orders["shares"].to_numpy(float) / session_volume
    return sigma_d * fit.c * q ** fit.delta * (
        1.0 + fit.k * np.log(np.asarray(rate, float)))


def fit_scale(predicted: np.ndarray, realised: np.ndarray) -> float:
    """Single least-squares scale k minimising ||realised - k * predicted||.

    One parameter, fitted in sample, applied out of sample: the same budget the
    square-root model gets for its c.
    """
    ok = np.isfinite(predicted) & np.isfinite(realised)
    denom = float(np.sum(predicted[ok] ** 2))
    return float(np.sum(predicted[ok] * realised[ok]) / denom) if denom > 0 else float("nan")


def _scores(realised: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(realised) & np.isfinite(predicted)
    y, p = realised[ok], predicted[ok]
    sst = float(np.sum((y - y.mean()) ** 2))
    r2_raw = 1.0 - float(np.sum((y - p) ** 2)) / sst if sst > 0 else float("nan")
    if p.std() > 0:
        slope, intercept = np.polyfit(p, y, 1)
        resid = y - (intercept + slope * p)
        r2_refit = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else float("nan")
    else:
        slope = intercept = r2_refit = float("nan")
    return {"n": int(len(y)), "r2_no_refit": r2_raw, "r2_refit": r2_refit,
            "slope": float(slope), "intercept": float(intercept),
            "mean_realised": float(y.mean()), "mean_predicted": float(p.mean())}


def calibration_table(realised: np.ndarray, predicted: np.ndarray,
                      n_deciles: int = 10) -> pd.DataFrame:
    """Mean realised against mean predicted, by predicted-impact decile.

    The decile table is where a headline R2 stops hiding things: a model can
    score respectably and still be wrong in a direction that matters, by
    over-predicting exactly the large orders a desk cares about.
    """
    ok = np.isfinite(realised) & np.isfinite(predicted)
    frame = pd.DataFrame({"realised": realised[ok], "predicted": predicted[ok]})
    frame["decile"] = pd.qcut(frame.predicted, n_deciles, labels=False,
                              duplicates="drop")
    table = frame.groupby("decile").agg(
        predicted=("predicted", "mean"),
        realised=("realised", "mean"),
        realised_se=("realised", lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))),
        n=("realised", "size")).reset_index()
    table["ratio"] = table.realised / table.predicted
    return table


MODEL_ORDER = ("propagator", "propagator_scaled", "sqrt", "sqrt_trailing_sigma",
               "sqrt_blend", "sqrt_tod_loso", "sqrt_tod_prior", "sqrt_rate")


@dataclass
class SessionResult:
    """Every model scored on one symbol-day, keyed by name.

    A dict rather than a field per model: there are eight of them now and adding
    the ninth should not mean touching the dataclass, the constructor and the
    runner in three places.
    """
    session: str
    calibration: Calibration
    scores: dict[str, dict] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)
    n_test: int = 0

    def r2(self, model: str) -> float:
        return self.scores.get(model, {}).get("r2_no_refit", float("nan"))


def evaluate_session(session: str, bars: pd.DataFrame, orders: pd.DataFrame,
                     session_volume: float, sigma_d: float,
                     train_frac: float = 0.7,
                     tod_profile_loso: pd.Series | None = None,
                     tod_profile_prior: pd.Series | None = None) -> SessionResult:
    """Conditional impact accuracy for one symbol-day, out of sample.

    Every model's parameters come from `train_orders` and are applied unchanged
    to `test_orders`. The time-of-day profiles are supplied by the caller,
    because they are estimated from OTHER sessions of the same symbol and this
    function only ever sees one.
    """
    train_end = int(len(bars) * train_frac)
    cal = calibrate_on_window(bars, train_end)
    split_second = float(bars["sec"].iloc[train_end])

    orders = orders[(orders.mid_start > 0) & (orders.shares > 0)].copy()
    # bars are RTH only, so only orders inside the bar grid can be priced by a
    # kernel estimated on it
    lo, hi = float(bars["sec"].iloc[0]), float(bars["sec"].iloc[-1])
    in_grid = (orders.t_start >= lo) & (orders.t_end <= hi)
    train_orders = orders[in_grid & (orders.t_start < split_second)]
    test_orders = orders[in_grid & (orders.t_start >= split_second)]
    if len(test_orders) < 50 or len(train_orders) < 50:
        raise ValueError(f"{session}: too few orders either side of the split")

    realised = realised_impact(test_orders)
    train_realised = realised_impact(train_orders)
    train_start = np.floor(train_orders.t_start.to_numpy(float))
    test_start = np.floor(test_orders.t_start.to_numpy(float))

    predictions: dict[str, np.ndarray] = {}
    params: dict[str, float] = {}

    # 1 and 2: the propagator, raw and with one level fitted on training orders
    prop_pred = predict_propagator(test_orders, cal)
    scale = fit_scale(predict_propagator(train_orders, cal), train_realised)
    predictions["propagator"] = prop_pred
    predictions["propagator_scaled"] = scale * prop_pred
    params["propagator_scale"] = scale

    # 3: the square root at a constant daily volatility
    c = fit_sqrt_coefficient(train_orders, session_volume, sigma_d)
    predictions["sqrt"] = predict_sqrt(test_orders, c, session_volume, sigma_d)
    params["sqrt_c"] = c

    # 4: the square root at realised volatility over the preceding half hour
    sig_train = trailing_sigma(bars, train_start)
    sig_test = trailing_sigma(bars, test_start)
    c_local = fit_sqrt_coefficient(train_orders, session_volume, sig_train)
    predictions["sqrt_trailing_sigma"] = predict_sqrt(test_orders, c_local,
                                                      session_volume, sig_test)
    params["sqrt_trailing_c"] = c_local

    # 5: a geometric blend of the two, with the weight fitted in sample
    alpha, c_blend = fit_blend_alpha(train_orders, session_volume, sigma_d,
                                     sig_train)
    predictions["sqrt_blend"] = predict_sqrt(
        test_orders, c_blend, session_volume,
        blend_sigma(sigma_d, sig_test, alpha))
    params["blend_alpha"] = alpha
    params["blend_c"] = c_blend

    # 6 and 7: a time-of-day multiplier on sigma_D, from other sessions
    for name, profile in (("sqrt_tod_loso", tod_profile_loso),
                          ("sqrt_tod_prior", tod_profile_prior)):
        if profile is None or profile.empty:
            continue
        sig_tr = profile_sigma(sigma_d, profile, train_start)
        sig_te = profile_sigma(sigma_d, profile, test_start)
        c_tod = fit_sqrt_coefficient(train_orders, session_volume, sig_tr)
        predictions[name] = predict_sqrt(test_orders, c_tod, session_volume, sig_te)
        params[f"{name}_c"] = c_tod

    # 8: the participation-rate correction, at a constant daily volatility so
    # the contrast against `sqrt` is the rate term and nothing else
    rate_train = execution_rate(train_orders, bars)
    rate_test = execution_rate(test_orders, bars)
    rate_fit = fit_rate_model(train_orders, session_volume, sigma_d, rate_train)
    predictions["sqrt_rate"] = predict_rate_model(test_orders, rate_fit,
                                                  session_volume, sigma_d, rate_test)
    params["rate_c"] = rate_fit.c
    params["rate_delta"] = rate_fit.delta
    params["rate_k"] = rate_fit.k
    params["rate_median_test"] = float(np.median(rate_test))

    return SessionResult(
        session=session, calibration=cal,
        scores={name: _scores(realised, pred) for name, pred in predictions.items()},
        tables={name: calibration_table(realised, pred)
                for name, pred in predictions.items()},
        params=params, n_test=int(len(test_orders)),
    )
