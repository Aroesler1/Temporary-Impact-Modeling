"""Scheduling under the validated conditional model: adapter, schedules, checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conditional_impact as ci
import impact_model as im
import panel
from scripts.run_conditional_impact import build_profiles
from scripts.run_schedule_conditional import (
    AC_FLOAT_ATOL,
    AC_FLOAT_RTOL,
    AC_MAX_ITER,
    ARTIFACTS,
    EVENT_DAY_SESSION,
    FLOAT_ATOL,
    FLOAT_RTOL,
    RISK_AVERSION_MULTIPLIERS,
    _bucket_index,
    _minute_bucket,
    _profile_over_slices,
    _slice_depths,
    bootstrap_median_by_session,
    build_causal_volume_profiles,
    build_schedules,
    choose_bucket_coefficients,
    evaluate_schedule_session,
    model_bucket_coefficient,
    model_cost_per_share,
    realised_cost_per_share,
    saving_pct,
    sessions_with_prior,
    verify_artifacts,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports" / "schedule_conditional"


# --------------------------------------------------------------------------
# the adapter: per-slice power-law cost through impact_model's KKT solver
# --------------------------------------------------------------------------

def test_slice_depths_recovers_the_original_a_exactly():
    """flat_cost cancels: allocate_schedule's internal a = flat_cost/Dt**p must
    equal the a_t passed in, whatever flat_cost is."""
    rng = np.random.default_rng(0)
    a = rng.uniform(1e-6, 1e-3, 40)
    for flat_cost in (1e-15, 1e-9, 1e-3):
        depth = _slice_depths(a, p=0.5, flat_cost=flat_cost)
        np.testing.assert_allclose(flat_cost / depth**0.5, a, rtol=1e-10)


def test_slice_depths_are_negligible_next_to_realistic_slice_sizes():
    """The flat region must never bind: depth << the smallest sane allocation."""
    a = np.full(50, 5e-5)
    depth = _slice_depths(a, p=0.5, flat_cost=1e-6 * np.median(a))
    assert depth.max() < 1e-6            # shares; any real slice trades far more


def test_kkt_risk_neutral_matches_the_closed_form_on_a_synthetic_profile():
    """Risk-neutral optimum for a_t*x^0.5 must be proportional to V_t/sigma_t^2."""
    rng = np.random.default_rng(1)
    n = 60
    sigma_t = rng.uniform(0.005, 0.03, n)
    v_t = rng.uniform(1e3, 1e6, n)
    c_hat = 0.9
    a_t = c_hat * sigma_t / np.sqrt(v_t)
    order_size = 5e4

    schedules, _, _ = build_schedules(a_t, 0.5, order_size, sigma_per_slice=1e-4)
    x = schedules["kkt_risk_neutral"]

    closed_form = v_t / sigma_t**2
    closed_form = order_size * closed_form / closed_form.sum()
    np.testing.assert_allclose(x, closed_form, rtol=1e-6)


def test_schedules_sum_to_the_order_size():
    rng = np.random.default_rng(2)
    a_t = rng.uniform(1e-5, 1e-3, 45)
    order_size = 12345.0
    schedules, _, _ = build_schedules(a_t, 0.5, order_size, sigma_per_slice=2e-4)
    for name, x in schedules.items():
        assert x.sum() == pytest.approx(order_size, rel=1e-8), name
        assert (x >= 0).all()


def test_inventory_penalty_front_loads_monotonically_in_risk_aversion():
    """Half-life falls and impact cost rises as the risk-aversion grid climbs,
    the same monotonicity impact_model's own tests require of the underlying
    allocator, checked here through this script's adapter and grid."""
    rng = np.random.default_rng(3)
    a_t = rng.uniform(2e-5, 8e-5, 90)
    order_size = 8e4
    schedules, risk_aversions, lam_ref = build_schedules(
        a_t, 0.5, order_size, sigma_per_slice=5e-4)
    assert lam_ref > 0
    assert risk_aversions["low"] < risk_aversions["medium"] < risk_aversions["high"]

    half_lives = []
    costs = []
    for label in ("low", "medium", "high"):
        x = schedules[f"kkt_ac_{label}"]
        half_lives.append(int(np.argmax(np.cumsum(x) >= 0.5 * order_size)))
        costs.append(model_cost_per_share(x, a_t) * order_size)
    assert half_lives[0] >= half_lives[1] >= half_lives[2]
    assert half_lives[2] < half_lives[0]
    assert costs[0] < costs[1] < costs[2]


def test_risk_neutral_is_the_cheapest_schedule_under_its_own_pricing():
    rng = np.random.default_rng(4)
    a_t = rng.uniform(1e-5, 5e-4, 70)
    order_size = 3e4
    schedules, _, _ = build_schedules(a_t, 0.5, order_size, sigma_per_slice=3e-4)
    twap_cost = model_cost_per_share(schedules["twap"], a_t)
    rn_cost = model_cost_per_share(schedules["kkt_risk_neutral"], a_t)
    assert rn_cost <= twap_cost + 1e-12


# --------------------------------------------------------------------------
# causal volume profile
# --------------------------------------------------------------------------

def test_minute_bucket_matches_conditional_impact_halfhour_bucket_granularity():
    """_bucket_index at 1800s must agree exactly with conditional_impact's own
    half-hour bucketing, since the realised-bucket check reuses that grid."""
    seconds = np.array([34200.0, 35999.0, 36000.0, 57599.0])
    np.testing.assert_array_equal(_bucket_index(seconds, 1800), ci.halfhour_bucket(seconds))


def test_minute_bucket_starts_at_the_open():
    got = _minute_bucket(np.array([34200.0, 34259.0, 34260.0]))
    np.testing.assert_array_equal(got, [0, 0, 1])


def test_causal_volume_profile_is_empty_for_a_symbols_first_session():
    profiles = build_causal_volume_profiles()
    meta = panel.meta()
    for symbol in meta.symbol.unique():
        rows = meta[meta.symbol == symbol].sort_values("date")
        first_session = rows.iloc[0].session
        assert profiles[first_session].empty


def test_causal_volume_profile_uses_only_strictly_earlier_sessions():
    """A session's profile must reproduce from ONLY its earlier same-symbol
    sessions' bars: swapping in different bars for a LATER sibling must not
    change it."""
    meta = panel.meta()
    row = meta[meta.symbol == "MSFT"].sort_values("date").iloc[1]
    profiles = build_causal_volume_profiles()
    prior_only = meta[(meta.symbol == "MSFT") & (meta.date < row.date)]
    assert len(prior_only) == 1
    donor_bars = panel.bars(prior_only.iloc[0].session)
    volume = pd.to_numeric(donor_bars["volume"], errors="coerce").to_numpy(float)
    minute = _minute_bucket(donor_bars["sec"].to_numpy(float))
    expected = pd.Series(volume, index=minute).groupby(level=0).sum()
    expected = expected / expected.sum()
    pd.testing.assert_series_equal(profiles[row.session].sort_index(),
                                   expected.sort_index(), check_names=False)


def test_profile_over_slices_sums_multiple_minutes_per_slice():
    shape = pd.Series({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
    slice_start = np.array([34200.0, 34320.0])   # minute 0 and minute 2, span 2 minutes
    got = _profile_over_slices(shape, slice_start, slice_seconds=120)
    np.testing.assert_allclose(got, [0.1 + 0.2, 0.3 + 0.4])


def test_profile_over_slices_fills_a_missing_minute_with_the_profile_median():
    shape = pd.Series({0: 0.1, 2: 0.3})   # minute 1 missing
    got = _profile_over_slices(shape, np.array([34260.0]), slice_seconds=60)
    assert got[0] == pytest.approx(shape.median())


def test_sessions_with_prior_excludes_exactly_each_symbols_earliest_session():
    meta = panel.meta()
    got = set(sessions_with_prior())
    earliest = {meta[meta.symbol == s].sort_values("date").iloc[0].session
                for s in meta.symbol.unique()}
    assert got == set(meta.session) - earliest
    assert len(got) == 12


# --------------------------------------------------------------------------
# realised-bucket pricing
# --------------------------------------------------------------------------

def test_choose_bucket_coefficients_falls_back_when_half_hour_is_too_thin():
    rng = np.random.default_rng(5)
    n = 200
    orders = pd.DataFrame({
        "sign": np.ones(n),
        "shares": rng.uniform(10, 1000, n),
        "t_start": np.sort(rng.uniform(34200.0, 34200.0 + 3 * 1800, n)),
        "mid_start": np.full(n, 100.0),
    })
    orders["mid_end"] = orders["mid_start"] * (1 + 0.5 * np.sqrt(orders.shares / 1e7))
    # only 200 orders spread over 3 half-hour buckets: well under 30 in at
    # least one of them by construction of a tiny synthetic sample size
    bucket_seconds, k_bucket, counts = choose_bucket_coefficients(
        orders, session_volume=1e7, sigma_d=1.0, split_second=34200.0,
        last_sec=34200.0 + 3 * 1800)
    assert bucket_seconds in (1800, 3600, 7200, 3 * 1800 + 1)
    assert min(counts.values()) >= 30
    assert set(k_bucket) == set(counts)


def test_choose_bucket_coefficients_uses_half_hour_when_it_is_thick_enough():
    rng = np.random.default_rng(6)
    n = 4000
    orders = pd.DataFrame({
        "sign": np.ones(n),
        "shares": rng.uniform(10, 1000, n),
        "t_start": np.sort(rng.uniform(34200.0, 34200.0 + 2 * 1800, n)),
        "mid_start": np.full(n, 100.0),
    })
    orders["mid_end"] = orders["mid_start"] * (1 + 0.5 * np.sqrt(orders.shares / 1e7))
    bucket_seconds, k_bucket, counts = choose_bucket_coefficients(
        orders, session_volume=1e7, sigma_d=1.0, split_second=34200.0,
        last_sec=34200.0 + 2 * 1800)
    assert bucket_seconds == 1800
    assert len(k_bucket) == 2


def test_check_buckets_covered_passes_when_every_slice_bucket_has_a_coefficient():
    from scripts.run_schedule_conditional import _check_buckets_covered

    _check_buckets_covered(np.array([9, 9, 10, 11]), {9: 1.0, 10: 1.1, 11: 1.2},
                           session="S", bucket_seconds=1800)


def test_check_buckets_covered_names_the_missing_bucket_instead_of_a_keyerror():
    """A slice bucket with no fitted coefficient must fail loudly and
    specifically, not with a bare KeyError from the caller's dict lookup."""
    from scripts.run_schedule_conditional import _check_buckets_covered

    with pytest.raises(ValueError, match=r"12.*no fitted"):
        _check_buckets_covered(np.array([9, 10, 12]), {9: 1.0, 10: 1.1},
                               session="S", bucket_seconds=1800)


def test_realised_cost_per_share_recovers_a_known_coefficient():
    x = np.full(10, 500.0)
    k = np.full(10, 0.8)
    got = realised_cost_per_share(x, k, sigma_d=0.02, session_volume=1e7)
    expected = float(np.mean(0.8 * 0.02 * np.sqrt(x / 1e7)))
    assert got == pytest.approx(expected, rel=1e-10)


def test_model_cost_per_share_matches_direct_formula():
    rng = np.random.default_rng(7)
    a = rng.uniform(1e-5, 1e-3, 30)
    x = rng.uniform(10, 1000, 30)
    got = model_cost_per_share(x, a, p=0.5)
    expected = float(np.sum(a * x**1.5)) / float(x.sum())
    assert got == pytest.approx(expected, rel=1e-12)


def test_model_bucket_coefficient_scales_with_the_time_of_day_multiplier():
    profile = pd.Series({9: 2.0})
    got = model_bucket_coefficient(c_hat=1.5, prior_profile=profile, bucket_id=9,
                                   bucket_seconds=1800)
    assert got == pytest.approx(3.0)


def test_saving_pct_is_positive_when_the_schedule_is_cheaper():
    assert saving_pct(benchmark_cost=1.0, schedule_cost=0.8) == pytest.approx(20.0)
    assert saving_pct(benchmark_cost=1.0, schedule_cost=1.2) == pytest.approx(-20.0)


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_median_by_session_brackets_the_median():
    rng = np.random.default_rng(8)
    sessions = np.repeat([f"S{i}" for i in range(10)], 5)
    values = 0.05 + 0.01 * rng.standard_normal(len(sessions))
    median, lo, hi = bootstrap_median_by_session(values, sessions, n_boot=500)
    assert median == pytest.approx(np.median(values))
    assert lo <= median <= hi


# --------------------------------------------------------------------------
# end to end on one real session
# --------------------------------------------------------------------------

def test_evaluate_schedule_session_reproduces_the_committed_sqrt_tod_prior_c():
    """c_hat here must equal conditional_impact_corrected's own sqrt_tod_prior_c
    for the same session: same split, same profile, same fitter, reused."""
    _, prior = build_profiles()
    volume_profiles = build_causal_volume_profiles()
    result = evaluate_schedule_session("MSFT_2024-06-03", prior, volume_profiles)

    committed = pd.read_csv(REPORTS.parent / "conditional_impact_corrected" / "summary.csv")
    row = committed.set_index("session").loc["MSFT_2024-06-03"]
    assert result.c_hat == pytest.approx(float(row["sqrt_tod_prior_c"]), rel=1e-8)
    assert result.n_slices > 30
    assert set(result.model_costs) == set(result.realised_costs)


def test_event_day_is_flagged():
    _, prior = build_profiles()
    volume_profiles = build_causal_volume_profiles()
    result = evaluate_schedule_session(EVENT_DAY_SESSION, prior, volume_profiles)
    assert result.is_event_day
    other = evaluate_schedule_session("MSFT_2024-06-03", prior, volume_profiles)
    assert not other.is_event_day


# --------------------------------------------------------------------------
# committed reports: self-consistency and the --check mode
# --------------------------------------------------------------------------

def test_pooled_summary_recomputes_from_schedule_savings():
    savings = pd.read_csv(REPORTS / "schedule_savings.csv")
    pooled = pd.read_csv(REPORTS / "pooled_summary.csv")
    for _, row in pooled.iterrows():
        group = savings[(savings.schedule == row.schedule) & (savings.benchmark == row.benchmark)
                        & (savings.pricing == row.pricing)]
        values = group["saving_pct"].to_numpy(float)
        assert int(row.n_sessions) == len(values)
        assert row.median_saving_pct == pytest.approx(float(np.median(values)))
        assert int(row.n_sessions_beats_benchmark) == int((values > 0).sum())


def test_input_manifest_hashes_match_committed_files():
    manifest = pd.read_csv(REPORTS / "input_manifest.csv")
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        assert path.stat().st_size == row.bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256


@pytest.mark.parametrize("damage", ["count", "float", "hash", "missing"])
def test_artifact_check_rejects_material_changes(tmp_path, damage):
    stored, rebuilt = tmp_path / "stored", tmp_path / "rebuilt"
    stored.mkdir()
    rebuilt.mkdir()
    for name in ARTIFACTS:
        frame = pd.DataFrame({"n": [1000], "saving": [0.2], "sha256": ["fixture"]})
        frame.to_csv(stored / name, index=False)
        if name == ARTIFACTS[0]:
            if damage == "count":
                frame.loc[0, "n"] += 1
            if damage == "float":
                frame.loc[0, "saving"] += 1e-4
            if damage == "hash":
                frame.loc[0, "sha256"] = "changed"
            if damage == "missing":
                frame.loc[0, "saving"] = np.nan
        frame.to_csv(rebuilt / name, index=False)
    with pytest.raises(AssertionError):
        verify_artifacts(stored, rebuilt)


def test_artifact_check_accepts_only_negligible_float_roundoff(tmp_path):
    stored, rebuilt = tmp_path / "stored", tmp_path / "rebuilt"
    stored.mkdir()
    rebuilt.mkdir()
    for name in ARTIFACTS:
        pd.DataFrame({"n": [100], "saving": [0.2]}).to_csv(stored / name, index=False)
        pd.DataFrame({"n": [100], "saving": [0.2 + 1e-12]}).to_csv(rebuilt / name, index=False)
    verify_artifacts(stored, rebuilt)


def _write_all_artifacts(base: Path, frame: pd.DataFrame) -> None:
    for name in ARTIFACTS:
        frame.to_csv(base / name, index=False)


def test_ac_rows_get_a_looser_tolerance_than_everything_else(tmp_path):
    """kkt_ac_* rows go through an iterative solver (SLSQP), so their own
    columns tolerate solver-precision-level drift that would fail every other
    schedule's strict tolerance."""
    assert AC_FLOAT_RTOL > FLOAT_RTOL
    assert AC_FLOAT_ATOL > FLOAT_ATOL
    assert AC_MAX_ITER > 400            # allocate_schedule_risk_averse's own default

    stored = pd.DataFrame({
        "schedule": ["twap", "kkt_risk_neutral", "kkt_ac_low", "kkt_ac_high"],
        "cost_per_share": [1.0, 0.9, 0.8, 2.0],
    })
    drift = (AC_FLOAT_RTOL + AC_FLOAT_ATOL) / 2       # inside the AC band, outside the strict one

    ok_dir, expected_dir = tmp_path / "ok", tmp_path / "expected"
    ok_dir.mkdir()
    expected_dir.mkdir()
    _write_all_artifacts(expected_dir, stored)
    drifted_on_ac = stored.copy()
    drifted_on_ac.loc[drifted_on_ac.schedule == "kkt_ac_low", "cost_per_share"] += 0.8 * drift
    _write_all_artifacts(ok_dir, drifted_on_ac)
    verify_artifacts(expected_dir, ok_dir)          # small drift on an AC row must pass

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    drifted_on_twap = stored.copy()
    drifted_on_twap.loc[drifted_on_twap.schedule == "twap", "cost_per_share"] += 0.8 * drift
    _write_all_artifacts(bad_dir, drifted_on_twap)
    with pytest.raises(AssertionError):             # same drift on TWAP must fail
        verify_artifacts(expected_dir, bad_dir)
