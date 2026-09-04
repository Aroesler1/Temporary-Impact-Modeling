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
    return out


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


@dataclass
class SessionResult:
    session: str
    calibration: Calibration
    propagator: dict = field(default_factory=dict)
    propagator_scaled: dict = field(default_factory=dict)
    sqrt_model: dict = field(default_factory=dict)
    sqrt_local: dict = field(default_factory=dict)
    propagator_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    propagator_scaled_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    sqrt_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    sqrt_local_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    sqrt_c: float = float("nan")
    sqrt_local_c: float = float("nan")
    propagator_scale: float = float("nan")


def evaluate_session(session: str, bars: pd.DataFrame, orders: pd.DataFrame,
                     session_volume: float, sigma_d: float,
                     train_frac: float = 0.7) -> SessionResult:
    """Conditional impact accuracy for one symbol-day, out of sample."""
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
    prop_pred = predict_propagator(test_orders, cal)

    # both level parameters come from the TRAINING orders only
    k = fit_scale(predict_propagator(train_orders, cal), realised_impact(train_orders))
    c = fit_sqrt_coefficient(train_orders, session_volume, sigma_d)
    scaled_pred = k * prop_pred
    sqrt_pred = predict_sqrt(test_orders, c, session_volume, sigma_d)

    # same law, but with volatility measured from the half hour before each
    # order instead of held at the daily constant
    sig_train = trailing_sigma(bars, np.floor(train_orders.t_start.to_numpy(float)))
    sig_test = trailing_sigma(bars, np.floor(test_orders.t_start.to_numpy(float)))
    c_local = fit_sqrt_coefficient(train_orders, session_volume, sig_train)
    local_pred = predict_sqrt(test_orders, c_local, session_volume, sig_test)

    return SessionResult(
        session=session, calibration=cal,
        propagator=_scores(realised, prop_pred),
        propagator_scaled=_scores(realised, scaled_pred),
        sqrt_model=_scores(realised, sqrt_pred),
        sqrt_local=_scores(realised, local_pred),
        propagator_table=calibration_table(realised, prop_pred),
        propagator_scaled_table=calibration_table(realised, scaled_pred),
        sqrt_table=calibration_table(realised, sqrt_pred),
        sqrt_local_table=calibration_table(realised, local_pred),
        sqrt_c=c, sqrt_local_c=c_local, propagator_scale=k,
    )
