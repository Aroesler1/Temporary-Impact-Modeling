#!/usr/bin/env python3
"""Execution scheduling under the validated conditional-impact model.

The old propagator-based schedule (section 4 of the README) is withdrawn:
its return-fitted kernel measured returns, not surviving price level, and
`docs/kernel_audit.md` limits the corrected level response to two seconds.
Nothing here uses a decay kernel, a transient/permanent split, or any horizon
beyond one execution slice: each slice's own cost IS the model's cost for that
slice, which is exactly why the two-second support limit does not apply.

This script reschedules the ONE model that validated out of sample:
`sqrt_tod_prior` in `conditional_impact.py` (median OOS R2 0.206, calibration
slope 0.955, 12 of 15 sessions -- the ones with a same-symbol prior session).
Every fitting step below reuses `conditional_impact.py`'s own functions
(`split_orders_for_evaluation`, `profile_sigma`, `fit_sqrt_coefficient`) so the
calibration is identical to the one already validated; nothing here refits the
propagator, and the same 70/30, strictly-causal discipline applies throughout.

PRE-REGISTERED CHOICES (fixed here, before any saving number was computed)
----------------------------------------------------------------------------
* Parent order: 1% of the session's trailing 20-day ADV (`adv_20d_xnas` in
  `data/session_meta.csv`), the repo's existing ADV convention (section 2's
  normaliser). This is a pre-session constant, not the session's own realised
  volume, so it carries no look-ahead.
* Slice granularity: ONE MINUTE. `impact_model.allocate_schedule` and
  `allocate_schedule_risk_averse`, the KKT/bisection scheduler this script
  reuses, are written in minute units natively (`sigma_per_minute`, "Optimal
  minute allocation"); one-minute slices are the granularity that solver
  already assumes. The held-out window is ~117-119 minutes on all twelve
  sessions (checked below), far above `MIN_SLICES`; the fallback that halves
  the granularity if a session ever came up short is implemented but is not
  exercised by this panel, which the printed report states explicitly.
* Per-slice cost: `q * c_hat * sigma_t * sqrt(q / V_t)`, i.e. a pure power law
  `a_t * x^0.5` with `a_t = c_hat * sigma_t / sqrt(V_t)`, in RAW LOG-RETURN
  units -- the same units as `I = c sigma_D sqrt(Q/V)` in section 1 and
  `conditional_impact.realised_impact`/`predict_sqrt` (not bp, not a
  sigma_D-divided ratio; multiply by 1e4 for bp). `sigma_t` is `sigma_D` times
  the session's `prior` half-hour time-of-day multiplier (the strictly causal
  variant already in `conditional_impact.py`; `loso` is not used here because
  it is not strictly causal). `V_t` is a strictly causal per-minute VOLUME
  PROFILE: the median, across that symbol's sessions strictly BEFORE this one,
  of each donor's own per-minute share of its total volume, scaled by this
  session's trailing 20-day ADV. Nothing from the scored session's own volume
  or from a later session enters `V_t`.
* KKT solver reuse: `impact_model.allocate_schedule` (risk-neutral) and
  `allocate_schedule_risk_averse` (Almgren-Chriss inventory-penalised) are the
  KKT/bisection allocators already in this repo, built for exactly this shape
  of per-slice cost (`g_t(x) = c` for `x <= Dt` else `a_t * x^p`). Their
  interface is a flat cost `c` and a depth array `Dt`, not a direct `a_t`
  array, so `_slice_depths` is a thin, tested adapter: for any shared flat
  cost `c`, `Dt_t = (c / a_t)^(1/p)` makes the solver recover exactly `a_t` in
  its tail branch (`c` cancels algebraically), so picking `c` far below the
  smallest realistic slice cost drives the flat region to a negligible size
  and the reparametrised solve returns the pure power-law optimum to floating
  point precision. No solver math is edited.
* Risk-aversion grid: three multiples, {0.1, 1, 10}, of a session-specific
  reference `lambda_ref` that equalises the impact-cost and inventory-risk
  terms of the Almgren-Chriss objective AT THE TWAP SCHEDULE. This keeps the
  grid meaningfully spanning "near TWAP" to "clearly front-loaded" on every
  session regardless of that session's absolute scale (ADV and sigma differ
  by an order of magnitude between INTC and MSFT/AAPL), without tuning the
  grid to any saving number: it is fixed by the TWAP baseline alone, before
  any schedule is compared.
* Bucket size for the realised, model-free check: half-hour buckets
  (`conditional_impact.halfhour_bucket`'s own 1800-second grid), falling back
  to one hour, then two hours, then the whole held-out window as one bucket,
  the first candidate that leaves at least `MIN_ORDERS_PER_BUCKET` (30)
  reconstructed test-window metaorders in EVERY bucket. All twelve sessions
  use half-hour buckets with the minimum observed bucket holding 86 orders;
  the report states this rather than assuming it.

WHAT THIS DOES NOT CLAIM
-------------------------
Every saving below is either MODEL-IMPLIED (priced by the same calibrated
model that built the schedule -- a consistency check, not independent
evidence) or REALISED-BUCKET-PRICED (priced by coefficients fitted fresh on
the held-out metaorders alone, not the model). No schedule here was executed;
nothing is a realised execution cost. The sample is twelve symbol-days on
three names in 2024, the same panel as the rest of this repository, and
carries no population or regime claim. INTC 2024-08-02 is the post-earnings
event day flagged throughout this README and is flagged again below.

Usage:
    python scripts/run_schedule_conditional.py
    python scripts/run_schedule_conditional.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import conditional_impact as ci  # noqa: E402
import impact_model as im  # noqa: E402
import panel  # noqa: E402
from scripts.run_conditional_impact import build_profiles  # noqa: E402

RTH_OPEN_SEC = ci.RTH_OPEN_SEC
MINUTES_PER_SESSION = 390.0            # 09:30-16:00 RTH, one row per minute-slice
SLICE_SECONDS = 60                     # one-minute slices; see module docstring
MIN_SLICES = 30                        # below this, halve the granularity and say so
ORDER_FRACTION_OF_ADV = 0.01           # 1% of trailing 20-day ADV, the repo's convention
P_EXPONENT = 0.5                       # the square-root law's own exponent on size
FLAT_COST_SCALE = 1e-6                 # adapter: flat-region cost, as a fraction of median a_t
RISK_AVERSION_MULTIPLIERS = {"low": 0.1, "medium": 1.0, "high": 10.0}
# allocate_schedule_risk_averse's SLSQP defaults to 400 iterations, which is
# not always fully converged: checked directly (not part of the test suite,
# since it is a platform-reproducibility property, not a unit test) that on
# every one of the 12 sessions and all three risk-aversion levels, the cost of
# the returned schedule at 5,000 iterations is bit-identical to 20,000, i.e.
# converged to the shared unique optimum of this convex problem (see the
# allocator's own docstring) rather than stopped early along a path that
# happens to differ by platform. 400 iterations left ~0.001-1% platform-
# dependent noise in `reports/schedule_conditional/`'s AC schedules; this does
# not.
AC_MAX_ITER = 5000
BUCKET_CANDIDATES_SECONDS = (1800, 3600, 7200)   # half hour, hour, two hours; then the whole window
MIN_ORDERS_PER_BUCKET = 30
EVENT_DAY_SESSION = "INTC_2024-08-02"
SCHEDULES = ("twap", "vwap", "kkt_risk_neutral", "kkt_ac_low", "kkt_ac_medium", "kkt_ac_high")
KKT_LABEL = "kkt_risk_neutral"          # "the KKT schedule" in the headline counts

ARTIFACTS = (
    "session_summary.csv",
    "bucket_coefficients.csv",
    "schedule_costs.csv",
    "schedule_savings.csv",
    "pooled_summary.csv",
    "methodology.csv",
    "input_manifest.csv",
)
FLOAT_RTOL = 1e-8
FLOAT_ATOL = 1e-10


# --------------------------------------------------------------------------
# causal inputs: which sessions have a prior, and the prior-session volume shape
# --------------------------------------------------------------------------

def sessions_with_prior() -> list[str]:
    """Same-symbol sessions with at least one earlier session: the sqrt_tod_prior set."""
    meta = panel.meta()
    return [row.session for row in meta.itertuples()
            if not meta[(meta.symbol == row.symbol) & (meta.date < row.date)].empty]


def _minute_bucket(at_second: np.ndarray) -> np.ndarray:
    """Which minute-since-open a time falls in, 0 at the RTH open."""
    return ((np.asarray(at_second, float) - RTH_OPEN_SEC) // SLICE_SECONDS).astype(int)


def _bucket_index(at_second: np.ndarray, bucket_seconds: int) -> np.ndarray:
    """Generalises `conditional_impact.halfhour_bucket` to any bucket width."""
    return ((np.asarray(at_second, float) - RTH_OPEN_SEC) // bucket_seconds).astype(int)


def _profile_over_slices(vol_shape: pd.Series, slice_start: np.ndarray,
                         slice_seconds: int) -> np.ndarray:
    """Sum of `vol_shape`'s per-minute values across the minute(s) each slice spans.

    `vol_shape` is indexed by minute-since-open (0..389). Missing minutes are
    filled with the profile's own median share before summing, so a session
    whose held-out window falls back to a coarser slice (`MIN_SLICES`) still
    gets every minute inside a wider slice counted once, not the first
    minute's share repeated.
    """
    if vol_shape.empty:
        return np.zeros(len(slice_start))
    n_minutes_grid = int(vol_shape.index.max()) + 1
    fallback = float(vol_shape[vol_shape > 0].median()) if (vol_shape > 0).any() else 0.0
    full = vol_shape.reindex(range(n_minutes_grid)).fillna(fallback).to_numpy(float)
    cum = np.concatenate([[0.0], np.cumsum(full)])
    span_minutes = max(1, slice_seconds // 60)
    start_minute = ((slice_start - RTH_OPEN_SEC) // 60).astype(int)
    end_minute = start_minute + span_minutes
    start_minute = np.clip(start_minute, 0, n_minutes_grid)
    end_minute = np.clip(end_minute, 0, n_minutes_grid)
    return cum[end_minute] - cum[start_minute]


def build_causal_volume_profiles() -> dict[str, pd.Series]:
    """Per-minute fraction of the day's volume, from strictly PRIOR sessions only.

    For each session, this is the median (across that symbol's sessions
    strictly before this one) of each donor's own per-minute share of its own
    total displayed volume. Scaled by the scored session's trailing 20-day ADV
    at the call site, this gives a causal V_t: nothing from the scored
    session's own volume, and nothing from a later session, enters it. Mirrors
    `scripts.run_conditional_impact.build_profiles`'s "prior" donor rule
    (same-symbol, strictly earlier date), applied to volume shape instead of
    the half-hour volatility ratio.
    """
    meta = panel.meta()
    shares: dict[str, pd.Series] = {}
    for row in meta.itertuples():
        bars = panel.bars(row.session)
        volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(float)
        minute = _minute_bucket(bars["sec"].to_numpy(float))
        total = float(np.nansum(volume))
        shares[row.session] = (
            pd.Series(volume, index=minute).groupby(level=0).sum() / total
            if total > 0 else pd.Series(dtype=float)
        )

    profiles: dict[str, pd.Series] = {}
    for row in meta.itertuples():
        earlier = meta[(meta.symbol == row.symbol) & (meta.date < row.date)]
        donors = [shares[s] for s in earlier.session if not shares[s].empty]
        if not donors:
            profiles[row.session] = pd.Series(dtype=float)
            continue
        combined = pd.concat(donors, axis=1).median(axis=1)
        profiles[row.session] = combined / combined.sum()
    return profiles


# --------------------------------------------------------------------------
# the adapter: a per-slice power-law cost, reusing impact_model's KKT solver
# --------------------------------------------------------------------------

def _slice_depths(a: np.ndarray, p: float, flat_cost: float) -> np.ndarray:
    """Depth array making `impact_model.allocate_schedule` price exactly `a * x**p`.

    See the module docstring's "KKT solver reuse" note: for any shared flat
    cost `flat_cost`, `Dt_t = (flat_cost / a_t)**(1/p)` reproduces `a_t` exactly
    in the solver's tail branch. `flat_cost` is chosen far below the per-slice
    costs being priced so the flat region never binds for a realistic order.
    """
    return (flat_cost / a) ** (1.0 / p)


def build_schedules(a: np.ndarray, p: float, order_size: float, sigma_per_slice: float
                    ) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    """TWAP and the three KKT schedules (risk-neutral, then the AC grid).

    VWAP is built by the caller directly from V_t. Returns the schedules, the
    realised risk-aversion values used (for the record), and lambda_ref.
    """
    n = len(a)
    flat_cost = FLAT_COST_SCALE * float(np.median(a))
    depth = _slice_depths(a, p, flat_cost)
    twap_x = np.full(n, order_size / n)
    x_rn = im.allocate_schedule(depth, flat_cost, p, order_size)

    cost_twap = float(np.sum(a * twap_x ** (1.0 + p)))
    inv_twap_var = float(np.sum(im.inventory_path(twap_x, order_size) ** 2))
    lam_ref = cost_twap / (sigma_per_slice ** 2 * inv_twap_var) if inv_twap_var > 0 else 0.0

    schedules = {"twap": twap_x, "kkt_risk_neutral": x_rn}
    risk_aversions = {}
    for label, mult in RISK_AVERSION_MULTIPLIERS.items():
        lam = mult * lam_ref
        risk_aversions[label] = lam
        schedules[f"kkt_ac_{label}"] = im.allocate_schedule_risk_averse(
            depth, flat_cost, p, order_size, sigma_per_slice, lam, max_iter=AC_MAX_ITER)
    return schedules, risk_aversions, lam_ref


def model_cost_per_share(x: np.ndarray, a: np.ndarray, p: float = P_EXPONENT) -> float:
    """Sum_t a_t x_t^(1+p) / sum(x): the calibrated model's own price for a schedule."""
    x = np.asarray(x, float)
    return float(np.sum(a * x ** (1.0 + p))) / float(np.sum(x))


def realised_cost_per_share(x: np.ndarray, k_per_slice: np.ndarray, sigma_d: float,
                            session_volume: float) -> float:
    """The same functional form, priced by realised per-bucket coefficients."""
    x = np.asarray(x, float)
    cost = k_per_slice * sigma_d * np.sqrt(x / session_volume) * x
    return float(np.sum(cost)) / float(np.sum(x))


# --------------------------------------------------------------------------
# the realised-bucket check: coefficients fit fresh on held-out metaorders only
# --------------------------------------------------------------------------

def choose_bucket_coefficients(
    test_orders: pd.DataFrame, session_volume: float, sigma_d: float,
    split_second: float, last_sec: float,
) -> tuple[int, dict[int, float], dict[int, int]]:
    """The finest bucket size, from `BUCKET_CANDIDATES_SECONDS`, that leaves at
    least `MIN_ORDERS_PER_BUCKET` test-window metaorders in every bucket it uses.

    Falls back to the whole held-out window as a single bucket if even two
    hours is too fine. `conditional_impact.fit_sqrt_coefficient` is reused
    unchanged for each bucket's coefficient: sigma_D is the plain daily
    constant here, not the time-of-day profile, matching the task's own
    "sigma_D sqrt(Q/V)" and keeping this check independent of the fitted
    model's time-of-day shape.
    """
    starts = test_orders["t_start"].to_numpy(float)
    candidates = list(BUCKET_CANDIDATES_SECONDS) + [int(last_sec - split_second) + 1]
    for bucket_seconds in candidates:
        idx = _bucket_index(starts, bucket_seconds)
        counts = pd.Series(idx).value_counts()
        if len(counts) > 0 and (counts >= MIN_ORDERS_PER_BUCKET).all():
            k_bucket = {
                int(b): ci.fit_sqrt_coefficient(test_orders[idx == b], session_volume, sigma_d)
                for b in counts.index
            }
            return bucket_seconds, k_bucket, {int(b): int(n) for b, n in counts.items()}
    raise RuntimeError(
        "even the whole held-out window has fewer than MIN_ORDERS_PER_BUCKET metaorders"
    )


def _check_buckets_covered(slice_bucket: np.ndarray, k_bucket: dict[int, float],
                           session: str, bucket_seconds: int) -> None:
    """Every bucket a schedule slice falls into must have a fitted coefficient.

    `choose_bucket_coefficients` only fits a coefficient for buckets that held
    at least `MIN_ORDERS_PER_BUCKET` held-out test orders. Slices and test
    orders share the same held-out window on this panel, so this has not
    triggered, but indexing `k_bucket` by an uncovered slice bucket would
    otherwise raise an opaque KeyError instead of naming the problem.
    """
    missing = sorted(set(slice_bucket.tolist()) - set(k_bucket))
    if missing:
        raise ValueError(
            f"{session}: {len(missing)} slice bucket(s) {missing} have no fitted "
            f"realised coefficient (bucket_seconds={bucket_seconds}); "
            "choose_bucket_coefficients only covers buckets with "
            ">= MIN_ORDERS_PER_BUCKET held-out test orders"
        )


def model_bucket_coefficient(c_hat: float, prior_profile: pd.Series, bucket_id: int,
                             bucket_seconds: int) -> float:
    """The model's own implied coefficient for a bucket: c_hat times its
    time-of-day multiplier, evaluated at the bucket's midpoint."""
    center = RTH_OPEN_SEC + (bucket_id + 0.5) * bucket_seconds
    multiplier = ci.profile_sigma(1.0, prior_profile, np.array([center]))[0]
    return c_hat * float(multiplier)


# --------------------------------------------------------------------------
# per-session pipeline
# --------------------------------------------------------------------------

@dataclass
class SessionSchedule:
    session: str
    symbol: str
    is_event_day: bool
    order_size: float
    n_slices: int
    slice_seconds: int
    sigma_d: float
    session_volume: float
    adv: float
    c_hat: float
    lam_ref: float
    risk_aversions: dict[str, float]
    bucket_seconds: int
    bucket_counts: dict[int, int]
    spearman_rho: float
    spearman_p: float
    model_costs: dict[str, float] = field(default_factory=dict)
    realised_costs: dict[str, float] = field(default_factory=dict)
    bucket_rows: list[dict] = field(default_factory=list)


def evaluate_schedule_session(
    session: str, prior_vol_profiles: dict[str, pd.Series],
    prior_volume_profiles: dict[str, pd.Series],
) -> SessionSchedule:
    scales = panel.scales(session)
    bars = panel.bars(session)
    orders = panel.metaorders(session)
    orders = orders[(orders.mid_start > 0) & (orders.shares > 0)].copy()
    train_orders, test_orders, split_second = ci.split_orders_for_evaluation(
        bars, orders, train_frac=panel.TRAIN_FRAC)

    sigma_d = float(scales.sigma_daily_20d)
    session_volume = float(scales.session_volume)
    adv = float(scales.adv_20d_xnas)
    prior_profile = prior_vol_profiles[session]
    if prior_profile is None or prior_profile.empty:
        raise ValueError(f"{session}: no prior-session volatility donor")

    train_start = np.floor(train_orders["t_start"].to_numpy(float))
    sig_train = ci.profile_sigma(sigma_d, prior_profile, train_start)
    c_hat = ci.fit_sqrt_coefficient(train_orders, session_volume, sig_train)

    last_sec = float(bars["sec"].iloc[-1])
    slice_seconds = SLICE_SECONDS
    n_slices = int((last_sec - split_second) // slice_seconds)
    while n_slices < MIN_SLICES and slice_seconds < 3600:
        slice_seconds *= 2
        n_slices = int((last_sec - split_second) // slice_seconds)
    if n_slices < MIN_SLICES:
        raise ValueError(f"{session}: held-out window too short even at {slice_seconds}s slices")

    slice_start = split_second + slice_seconds * np.arange(n_slices)
    sigma_t = ci.profile_sigma(sigma_d, prior_profile, slice_start)

    vol_shape = prior_volume_profiles[session]
    v_t = _profile_over_slices(vol_shape, slice_start, slice_seconds) * adv   # shares in one slice
    v_t = np.maximum(v_t, 1e-9)                       # defensive floor; never binds on this panel

    a_t = c_hat * sigma_t / np.sqrt(v_t)
    order_size = ORDER_FRACTION_OF_ADV * adv
    sigma_per_slice = sigma_d / np.sqrt(MINUTES_PER_SESSION * 60.0 / slice_seconds)

    schedules, risk_aversions, lam_ref = build_schedules(a_t, P_EXPONENT, order_size,
                                                         sigma_per_slice)
    schedules["vwap"] = order_size * v_t / v_t.sum()

    bucket_seconds, k_bucket, bucket_counts = choose_bucket_coefficients(
        test_orders, session_volume, sigma_d, split_second, last_sec)
    slice_bucket = _bucket_index(slice_start, bucket_seconds)
    _check_buckets_covered(slice_bucket, k_bucket, session, bucket_seconds)
    k_per_slice = np.array([k_bucket[b] for b in slice_bucket])

    bucket_rows = []
    k_model_list, k_real_list = [], []
    for bucket_id in sorted(k_bucket):
        k_model = model_bucket_coefficient(c_hat, prior_profile, bucket_id, bucket_seconds)
        k_model_list.append(k_model)
        k_real_list.append(k_bucket[bucket_id])
        bucket_rows.append({
            "session": session, "bucket_seconds": bucket_seconds, "bucket_id": bucket_id,
            "bucket_start_sec": RTH_OPEN_SEC + bucket_id * bucket_seconds,
            "n_orders": bucket_counts[bucket_id],
            "k_realised": k_bucket[bucket_id], "k_model": k_model,
        })
    if len(k_real_list) >= 3:
        rho, pval = spearmanr(k_real_list, k_model_list)
    else:
        rho, pval = float("nan"), float("nan")

    model_costs = {name: model_cost_per_share(x, a_t) for name, x in schedules.items()}
    realised_costs = {name: realised_cost_per_share(x, k_per_slice, sigma_d, session_volume)
                      for name, x in schedules.items()}

    return SessionSchedule(
        session=session, symbol=str(scales.symbol), is_event_day=(session == EVENT_DAY_SESSION),
        order_size=order_size, n_slices=n_slices, slice_seconds=slice_seconds,
        sigma_d=sigma_d, session_volume=session_volume, adv=adv, c_hat=c_hat, lam_ref=lam_ref,
        risk_aversions=risk_aversions, bucket_seconds=bucket_seconds, bucket_counts=bucket_counts,
        spearman_rho=float(rho), spearman_p=float(pval),
        model_costs=model_costs, realised_costs=realised_costs, bucket_rows=bucket_rows,
    )


# --------------------------------------------------------------------------
# pooling and bootstrap
# --------------------------------------------------------------------------

def saving_pct(benchmark_cost: float, schedule_cost: float) -> float:
    return (benchmark_cost - schedule_cost) / benchmark_cost * 100.0


def bootstrap_median_by_session(values: np.ndarray, sessions: np.ndarray,
                                n_boot: int = 4000, seed: int = 0
                                ) -> tuple[float, float, float]:
    """95% band for the MEDIAN, resampling whole sessions with replacement.

    The repo's existing bootstraps (`execution.bootstrap_saving`,
    `scripts.run_conditional_impact.band`) resample whole symbol-days for a
    MEAN; this keeps the same by-session resampling and swaps in the median,
    since that is the pooled statistic asked for here.
    """
    values = np.asarray(values, float)
    sessions = np.asarray(sessions)
    ok = np.isfinite(values)
    values, sessions = values[ok], sessions[ok]
    uniq = np.unique(sessions)
    by = {u: np.flatnonzero(sessions == u) for u in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[u] for u in pick])
        draws.append(float(np.median(values[idx])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.median(values)), float(lo), float(hi)


# --------------------------------------------------------------------------
# hashing and artifact verification, matching run_conditional_impact.py
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(sessions: list[str]) -> pd.DataFrame:
    paths = [panel.DATA / "session_meta.csv",
             panel.DATA.parent / "reports" / "conditional_impact_corrected" / "summary.csv"]
    for key in panel.session_keys():
        paths.append(panel.DATA / f"{key}_1s.csv")
    for key in sessions:
        paths.append(panel.DATA / f"{key}_metaorders.csv")
    return pd.DataFrame([
        {"path": p.relative_to(panel.DATA.parent).as_posix(), "bytes": p.stat().st_size,
         "sha256": _sha256(p)}
        for p in paths
    ])


# kkt_ac_* schedules go through allocate_schedule_risk_averse (SLSQP), an
# iterative optimizer, not the closed-form/bisection solvers behind every
# other column. AC_MAX_ITER above already pushes it to a converged, path-
# independent optimum on one machine (checked directly: 5,000 iterations
# reproduces 20,000 bit for bit on all 12 sessions and all three risk-aversion
# levels), but an iterative solver's agreement ACROSS machines is still a
# solver-precision property, not exact arithmetic. Two tolerance tiers for
# kkt_ac_* rows, both checked against CI's Linux runner against this branch's
# macOS-built committed values:
# * AC_FLOAT_RTOL/ATOL for return-unit columns (cost_per_share,
#   saving_per_share, magnitude ~1e-4 to 1e-2): the observed gap there was
#   ~1.5e-6 relative; this keeps two orders of magnitude of margin above it.
# * AC_PCT_RTOL/ATOL for percentage-POINT columns (saving_pct and
#   pooled_summary's derived percentiles/bootstrap band, magnitude ~0.2 to
#   200): relative tolerance alone is not meaningful there because a saving
#   can sit arbitrarily close to zero, so a fixed absolute floor
#   (AC_PCT_ATOL, in percentage points) does the real work. The observed gap
#   was ~3.3e-5 percentage points on a near-zero saving; this keeps roughly
#   thirty times that margin.
# Every other column (TWAP, VWAP, the risk-neutral KKT schedule, and all
# non-schedule tables) keeps the strict, closed-form tolerance, which every
# run so far has matched exactly.
AC_FLOAT_RTOL = 1e-4
AC_FLOAT_ATOL = 1e-7
AC_PCT_RTOL = 1e-4
AC_PCT_ATOL = 1e-3
PCT_COLUMNS = frozenset({
    "saving_pct", "median_saving_pct", "q25_saving_pct", "q75_saving_pct",
    "min_saving_pct", "max_saving_pct", "bootstrap_lo", "bootstrap_hi",
})


def verify_artifacts(expected_dir: Path, rebuilt_dir: Path) -> None:
    for name in ARTIFACTS:
        stored = pd.read_csv(expected_dir / name)
        rebuilt = pd.read_csv(rebuilt_dir / name)
        pd.testing.assert_index_equal(stored.columns, rebuilt.columns)
        if stored.shape != rebuilt.shape:
            raise ValueError(f"{name}: artifact shape differs")

        has_schedule_column = "schedule" in stored.columns
        is_ac_row = (stored["schedule"].astype(str).str.startswith("kkt_ac_")
                    if has_schedule_column else pd.Series(False, index=stored.index))

        for column in stored:
            left, right = stored[column], rebuilt[column]
            floating = (pd.api.types.is_float_dtype(left.dtype)
                        and pd.api.types.is_float_dtype(right.dtype))
            if not floating or not has_schedule_column:
                pd.testing.assert_series_equal(
                    left, right, check_exact=not floating,
                    rtol=FLOAT_RTOL, atol=FLOAT_ATOL, obj=f"{name}/{column}",
                )
                continue
            ac_rtol, ac_atol = ((AC_PCT_RTOL, AC_PCT_ATOL) if column in PCT_COLUMNS
                               else (AC_FLOAT_RTOL, AC_FLOAT_ATOL))
            pd.testing.assert_series_equal(
                left[~is_ac_row], right[~is_ac_row], check_exact=False,
                rtol=FLOAT_RTOL, atol=FLOAT_ATOL, obj=f"{name}/{column} (non-AC rows)",
            )
            pd.testing.assert_series_equal(
                left[is_ac_row], right[is_ac_row], check_exact=False,
                rtol=ac_rtol, atol=ac_atol, obj=f"{name}/{column} (AC rows)",
            )


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------

def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sessions_with_prior()

    _, prior_vol_profiles = build_profiles()
    prior_volume_profiles = build_causal_volume_profiles()

    results: dict[str, SessionSchedule] = {}
    for session in sessions:
        results[session] = evaluate_schedule_session(session, prior_vol_profiles,
                                                      prior_volume_profiles)

    summary_rows, bucket_rows, cost_rows, saving_rows = [], [], [], []
    for session, r in results.items():
        summary_rows.append({
            "session": session, "symbol": r.symbol, "is_event_day": r.is_event_day,
            "order_size_shares": r.order_size, "n_slices": r.n_slices,
            "slice_seconds": r.slice_seconds, "sigma_d": r.sigma_d,
            "session_volume": r.session_volume, "adv_20d": r.adv, "c_hat": r.c_hat,
            "lambda_ref": r.lam_ref,
            "risk_aversion_low": r.risk_aversions["low"],
            "risk_aversion_medium": r.risk_aversions["medium"],
            "risk_aversion_high": r.risk_aversions["high"],
            "bucket_seconds": r.bucket_seconds, "n_buckets": len(r.bucket_counts),
            "min_orders_per_bucket": min(r.bucket_counts.values()),
            "median_orders_per_bucket": float(np.median(list(r.bucket_counts.values()))),
            "spearman_rho": r.spearman_rho, "spearman_p": r.spearman_p,
        })
        bucket_rows.extend(r.bucket_rows)
        for name in SCHEDULES:
            cost_rows.append({"session": session, "schedule": name, "pricing": "model",
                              "cost_per_share": r.model_costs[name]})
            cost_rows.append({"session": session, "schedule": name, "pricing": "realised_bucket",
                              "cost_per_share": r.realised_costs[name]})
        for pricing, costs in (("model", r.model_costs), ("realised_bucket", r.realised_costs)):
            for name in SCHEDULES:
                if name in ("twap", "vwap"):
                    continue
                for benchmark in ("twap", "vwap"):
                    saving_rows.append({
                        "session": session, "schedule": name, "benchmark": benchmark,
                        "pricing": pricing,
                        "saving_per_share": costs[benchmark] - costs[name],
                        "saving_pct": saving_pct(costs[benchmark], costs[name]),
                    })

    summary = pd.DataFrame(summary_rows)
    bucket_table = pd.DataFrame(bucket_rows)
    cost_table = pd.DataFrame(cost_rows)
    saving_table = pd.DataFrame(saving_rows)

    summary.to_csv(out_dir / "session_summary.csv", index=False)
    bucket_table.to_csv(out_dir / "bucket_coefficients.csv", index=False)
    cost_table.to_csv(out_dir / "schedule_costs.csv", index=False)
    saving_table.to_csv(out_dir / "schedule_savings.csv", index=False)

    pooled_rows = []
    for (schedule, benchmark, pricing), group in saving_table.groupby(
        ["schedule", "benchmark", "pricing"]
    ):
        values = group["saving_pct"].to_numpy(float)
        sess = group["session"].to_numpy()
        median, lo, hi = bootstrap_median_by_session(values, sess)
        pooled_rows.append({
            "schedule": schedule, "benchmark": benchmark, "pricing": pricing,
            "median_saving_pct": median,
            "q25_saving_pct": float(np.percentile(values, 25)),
            "q75_saving_pct": float(np.percentile(values, 75)),
            "min_saving_pct": float(values.min()), "max_saving_pct": float(values.max()),
            "bootstrap_lo": lo, "bootstrap_hi": hi,
            "n_sessions": len(values), "n_sessions_beats_benchmark": int((values > 0).sum()),
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(out_dir / "pooled_summary.csv", index=False)

    pd.DataFrame([{
        "report_version": "schedule-conditional-v1",
        "n_sessions": len(sessions),
        "slice_seconds": SLICE_SECONDS,
        "min_slices": MIN_SLICES,
        "order_fraction_of_adv": ORDER_FRACTION_OF_ADV,
        "p_exponent": P_EXPONENT,
        "flat_cost_scale": FLAT_COST_SCALE,
        "risk_aversion_multipliers": ",".join(
            f"{k}={v}" for k, v in RISK_AVERSION_MULTIPLIERS.items()),
        "bucket_candidates_seconds": ",".join(str(b) for b in BUCKET_CANDIDATES_SECONDS),
        "min_orders_per_bucket": MIN_ORDERS_PER_BUCKET,
        "minutes_per_session": MINUTES_PER_SESSION,
        "train_fraction": panel.TRAIN_FRAC,
        "event_day_session": EVENT_DAY_SESSION,
        "any_session_used_coarser_than_one_minute_slices": bool(
            (summary["slice_seconds"] > SLICE_SECONDS).any()),
        "any_session_used_coarser_than_half_hour_buckets": bool(
            (summary["bucket_seconds"] > BUCKET_CANDIDATES_SECONDS[0]).any()),
    }]).to_csv(out_dir / "methodology.csv", index=False)

    input_manifest(sessions).to_csv(out_dir / "input_manifest.csv", index=False)

    # ------------------------------------------------------------- console
    print("SCHEDULING UNDER THE CONDITIONAL MODEL, 12 sessions with a prior session")
    print("Model-implied cost prices a schedule with the same model that built it; "
          "realised-bucket cost prices it with coefficients fit fresh on held-out "
          "metaorders alone.\n")
    show = summary[["session", "c_hat", "spearman_rho", "min_orders_per_bucket"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    print("\nPOOLED SAVING vs TWAP and VWAP, median over 12 sessions with a bootstrap "
          "band by session\n")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))

    for pricing in ("model", "realised_bucket"):
        for benchmark in ("twap", "vwap"):
            row = pooled[(pooled.schedule == KKT_LABEL) & (pooled.benchmark == benchmark)
                        & (pooled.pricing == pricing)].iloc[0]
            print(f"\nKKT risk-neutral beats {benchmark.upper()} under {pricing} pricing on "
                  f"{int(row.n_sessions_beats_benchmark)} of {int(row.n_sessions)} sessions")

    event = summary[summary.is_event_day]
    if len(event):
        print(f"\n{EVENT_DAY_SESSION} (flagged event day) spearman rho: "
              f"{float(event.spearman_rho.iloc[0]):.3f}")

    print(f"\nsaved -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/schedule_conditional"))
    ap.add_argument("--check", action="store_true",
                    help="rebuild in a temporary directory and compare committed artifacts")
    args = ap.parse_args()
    if not args.check:
        build(args.out_dir)
        return 0

    with tempfile.TemporaryDirectory(prefix="impact-schedule-check-") as raw:
        rebuilt = Path(raw)
        build(rebuilt)
        verify_artifacts(args.out_dir, rebuilt)
    print(f"Verified {len(ARTIFACTS)} schedule-conditional artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
