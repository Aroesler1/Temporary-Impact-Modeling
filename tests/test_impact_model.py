"""Tests for the vectorized impact model and KKT allocator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from impact_model import (  # noqa: E402
    allocate_schedule,
    first_nonzero_ask_depth,
    fit_power_law,
    schedule_cost,
    walk_book_premiums,
)


def _synthetic_book(n_rows=400, symbol="TEST", px0=20.0, depth_scale=100.0, exponent=0.5, seed=0):
    """MBP-10-style snapshots whose book-walk premium follows a known power law.

    Levels are spaced so that the cumulative premium of walking to size x
    is approximately proportional to x**exponent.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    rows["symbol"] = np.repeat(symbol, n_rows)
    base = px0 * (1 + rng.normal(0, 0.001, n_rows)).cumprod()
    sizes = np.full((n_rows, 10), depth_scale)
    cum = np.cumsum(sizes, axis=1)
    # choose level prices so the average premium ABOVE THE BEST ASK at
    # cumulative size x is k * x^exponent (zero premium at level 0, since
    # walk premiums are measured relative to ask_px_00)
    k = 0.001 * px0 / depth_scale**exponent
    total_prem = k * cum**exponent * cum
    total_prem[:, 0] = 0.0
    lvl_prem = np.diff(np.concatenate([np.zeros((n_rows, 1)), total_prem], axis=1), axis=1)
    for i in range(10):
        rows[f"ask_sz_{i:02d}"] = sizes[:, i]
        rows[f"ask_px_{i:02d}"] = base + lvl_prem[:, i] / sizes[:, i]
    return pd.DataFrame(rows)


def test_first_nonzero_depth_vectorized():
    df = _synthetic_book(50)
    df.loc[df.index[:5], "ask_sz_00"] = 0.0
    depth = first_nonzero_ask_depth(df)
    assert (depth.iloc[:5] == 100.0).all()  # falls through to level 1
    assert (depth.iloc[5:] == 100.0).all()


def test_power_law_fit_recovers_exponent():
    df = _synthetic_book(exponent=0.5)
    walks = walk_book_premiums(df)
    fit = fit_power_law(walks["size_rel"], walks["premium_bps"])
    assert abs(fit.exponent - 0.5) < 0.1
    assert fit.r_squared > 0.9


def test_normalization_makes_pooling_scale_free():
    # same microstructure at 10x the price and 5x the depth must pool cleanly
    df1 = _synthetic_book(symbol="AAA", px0=20.0, depth_scale=100.0, exponent=0.45, seed=1)
    df2 = _synthetic_book(symbol="BBB", px0=200.0, depth_scale=500.0, exponent=0.45, seed=2)
    pooled = pd.concat([df1, df2], ignore_index=True)
    walks = walk_book_premiums(pooled)

    fit_pooled = fit_power_law(walks["size_rel"], walks["premium_bps"])
    assert abs(fit_pooled.exponent - 0.45) < 0.1

    # raw-dollar pooling across the two price scales distorts the exponent
    fit_raw = fit_power_law(walks["size_shares"], walks["premium_dollars"])
    assert abs(fit_raw.exponent - 0.45) > abs(fit_pooled.exponent - 0.45)


def test_bootstrap_ci_brackets_exponent():
    # per-symbol exponents scattered around 0.5 -> non-degenerate CI that
    # must bracket the population value
    exps = [0.42, 0.46, 0.50, 0.50, 0.54, 0.58]
    df = pd.concat(
        [_synthetic_book(symbol=f"S{i}", seed=i, exponent=e) for i, e in enumerate(exps)],
        ignore_index=True,
    )
    walks = walk_book_premiums(df)
    fit = fit_power_law(walks["size_rel"], walks["premium_bps"], groups=walks["symbol"], n_boot=200)
    assert fit.ci_high > fit.ci_low
    assert fit.ci_low < 0.5 < fit.ci_high


def test_allocator_flat_region_proportional():
    Dt = np.array([100.0, 200.0, 300.0])
    x = allocate_schedule(Dt, c=0.01, p=0.45, S=300.0)
    assert np.allclose(x, Dt * 0.5)
    assert np.isclose(x.sum(), 300.0)


def test_allocator_beats_perturbations():
    """KKT solution should not be improvable by moving mass between minutes."""
    rng = np.random.default_rng(4)
    Dt = rng.uniform(50, 400, size=60)
    c, p, S = 0.02, 0.45, 40000.0

    x_opt = allocate_schedule(Dt, c, p, S)
    assert np.isclose(x_opt.sum(), S)
    base_cost = schedule_cost(x_opt, Dt, c, p)

    for _ in range(200):
        i, j = rng.choice(len(Dt), size=2, replace=False)
        eps = min(x_opt[i], rng.uniform(0.1, 25.0))
        x_alt = x_opt.copy()
        x_alt[i] -= eps
        x_alt[j] += eps
        assert schedule_cost(x_alt, Dt, c, p) >= base_cost - 1e-6


def test_allocator_matches_greedy_discretization():
    """Continuous KKT cost ~= cost of a fine greedy marginal-cost allocation."""
    Dt = np.array([120.0, 260.0, 80.0, 500.0, 340.0])
    c, p, S = 0.05, 0.5, 5000.0

    x_opt = allocate_schedule(Dt, c, p, S)
    kkt_cost = schedule_cost(x_opt, Dt, c, p)

    # greedy: repeatedly add small blocks where marginal cost is lowest
    step = 5.0
    a = c / Dt**p
    x = np.zeros_like(Dt)
    for _ in range(int(S / step)):
        marginal = np.where(x < Dt, c, (1 + p) * a * np.maximum(x, 1e-12) ** p)
        j = int(np.argmin(marginal))
        x[j] += step
    greedy_cost = schedule_cost(x, Dt, c, p)

    assert kkt_cost <= greedy_cost * 1.001
