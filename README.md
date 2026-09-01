# Intraday Liquidity & Optimal Execution (Work-Trial Task)

This repo builds an intraday liquidity curve from quote data, calibrates a **piecewise temporary impact** model (flat cost up to depth; concave power-law tail), and solves for the **optimal minute-by-minute execution schedule** for a fixed parent order using KKT conditions with a bisection on the Lagrange multiplier.

![Intraday Average Dt](figs/dt_intraday.png)

---

## What’s in here

- **Modeling note** — estimating the flat cost $c$, intraday depth $D_t$, and tail exponent $p$, with plots and rationale.  
- **Allocation note** — KKT-based algorithm with a bisection on the multiplier to hit the target $S$.  
- **Notebook / code** — end-to-end data loading, P-spline smoothing, power-law fit, and final allocation.  
- **`impact_model.py` + `tests/`** — vectorized, tested implementations (2026-08 revision): book-walk premiums without Python loops, scale-free cross-symbol pooling, bootstrap CIs for the exponent, and the KKT allocator with optimality tests. Runs on a synthetic MBP-10 fixture, no proprietary data needed.

## 2026-08 revision notes

- **Scale-free pooling.** The notebook pooled (shares, \$/share premium) pairs across SOUN/FROG/CRWV directly; different price levels and depth scales bias the pooled exponent. `impact_model.walk_book_premiums` normalizes premiums to basis points of the best ask and sizes to multiples of each symbol's median top-of-book depth before pooling (a unit test shows raw-dollar pooling distorts a known exponent while normalized pooling recovers it).
- **Uncertainty.** `fit_power_law` reports a symbol-block bootstrap 95% CI for $p$, not just a point estimate.
- **Scope, stated honestly.** Walking displayed snapshots measures the *virtual instantaneous* cost of consuming visible liquidity. Realized metaorder impact additionally reflects hidden liquidity, queue refill, and adverse selection. The calibrated curve here is a displayed-liquidity lower bound, not a fitted metaorder impact curve — see the section below on which square-root law this can and cannot speak to.
- **Allocator verified.** The per-minute cost is convex (marginal cost steps from $c$ to $(1+p)c$ at $D_t$), so the KKT/bisection solution is globally optimal; tests confirm no pairwise mass transfer improves it and that it matches a fine greedy marginal-cost discretization. The flat-region allocation is degenerate; proportional-to-$D_t$ is the documented tie-break.

### Which square-root law this estimator can speak to

"The square-root law" is really two distinct statements, and conflating them is the usual error. Durin, Rosenbaum and co-authors ([arXiv 2311.18283](https://arxiv.org/abs/2311.18283)) separate them:

1. **In cumulated volume.** *During* execution of a metaorder, impact grows as the square root of the volume executed so far.
2. **In participation rate.** For a *given* total executed volume $Q$, impact scales as $\sqrt{\gamma}$ in the participation rate $\gamma$, once $\gamma$ is large enough.

The exponent fitted here ($\hat{p} \approx 0.45$) is **neither**. It is the concavity of the cost of walking a *static displayed book* against order size: a cross-sectional depth-profile property measured at an instant. It has no time dimension, so it cannot express law 1, and it is indexed by shares rather than by participation in contemporaneous volume, so it cannot express law 2. That $\hat{p}$ lands near $1/2$ is suggestive of common liquidity structure, but it is not confirmation of either law, and this repo does not claim it is.

Evidence for the laws themselves is now strong: a complete survey of the Tokyo Stock Exchange finds square-root scaling across all liquid stocks over eight years ([arXiv 2411.13965](https://arxiv.org/abs/2411.13965)), and Maitrier and Bouchaud ([arXiv 2506.07711](https://arxiv.org/abs/2506.07711)) give a framework reconciling impact, order imbalance and volatility.

Reaching those laws from here needs metaorders, not snapshots: reconstructing proxy metaorders from public data ([arXiv 2503.18199](https://arxiv.org/abs/2503.18199)) and re-indexing $g_t$ on participation of the minute-volume curve. That is the natural next build, alongside a transient-impact propagator — calibrated by regressing returns on lagged signed volumes across multiple lags and choosing kernel parameters to maximise out-of-sample $R^2$ — to replace the memoryless cost model with one that has resilience.

## Transient-impact propagator, calibrated on real order flow (2026-08)

The piecewise model above is **memoryless**: cost at minute t depends only on
size at t. Real impact decays instead of vanishing. `propagator.py` calibrates
the Bouchaud-style kernel

    r_t  =  sum_{l=0..L} G(l) * sign(v_{t-l}) * |v_{t-l}|^delta  +  noise

directly from data, on one second bars of MSFT 2024-06-03 built from Databento
`XNAS.ITCH` MBO (23,390 regular-session seconds; signed volume from book-affecting
fills, mid from a reconstructed book). Kernel parameters are selected by
out-of-sample R^2 on a chronological 70/30 split — nothing is fitted on the
evaluation window.

| specification | best delta | best lags | out-of-sample R² |
|---|---|---|---|
| **explanatory** (contemporaneous flow included) | 0.25 | 60 | **0.371** |
| memoryless (L = 0, same delta) | 0.25 | 0 | 0.369 |
| **predictive** (lags ≥ 1 only) | 1.0 | 20 | **0.0043** |

Three findings, all reported as they came out:

1. **Signed order flow explains 37% of contemporaneous one-second returns.**
   That is impact, and it is large.
2. **It predicts almost nothing.** Dropping the contemporaneous term collapses
   out-of-sample R² by a factor of **87**, to 0.4%. Contemporaneous explanatory
   power is not tradeable signal, and conflating the two is the standard error
   this table exists to prevent. The pattern matches Cont, Cucuringu and Zhang
   on order flow imbalance: strong contemporaneous relation, weak and
   fast-decaying predictive one.
3. **At one-second resolution the kernel has essentially already decayed.**
   Adding 60 lags of history improves the explanatory model by only +0.0017 R²,
   and G(1) is already about 6% of G(0) with the opposite sign. Transient impact
   is real, but at this sampling frequency the relaxation has mostly happened
   inside the first bucket.

Caveats that bound all three: one instrument, one session, and one-second
buckets. The propagator literature usually works in trade or event time, where
slower decay is visible; a one-second grid may simply be too coarse to resolve
it. The fitted delta of 0.25 is more concave than the square-root form, but this
is aggregated interval flow, not metaorders, so it is not a measurement of the
square-root law either.

Reproduce:

```bash
python scripts/run_propagator.py --data data/MSFT_2024-06-03_1s.csv
```

### Risk-averse extension (Almgren-Chriss)

`impact_model.allocate_schedule_risk_averse` extends the static KKT schedule with the Almgren-Chriss mean-variance objective (impact cost plus $\lambda\sigma^2\sum_t I_t^2$ on remaining inventory), solved as a convex program warm-started from the risk-neutral optimum. `scripts/compare_schedules_demo.py` regenerates `figs/schedule_comparison.png` and a cost/risk table against TWAP and depth-proportional baselines, and exposes a non-obvious property of the piecewise model:

- **Small orders (inside the flat region):** every schedule pays the same flat cost $cS$, so risk aversion is *free*; the AC schedule cuts inventory variance ~10x at zero extra impact cost. The flat-region degeneracy of the risk-neutral problem is exactly what the risk term resolves.
- **Large orders (tail engaged):** a genuine tradeoff appears, e.g. $\lambda=10^{-5}$ cuts inventory variance ~5x for ~61% more impact cost.

Run the tests:

```bash
python -m pytest tests -q
```

Key figures:

![P-spline Smooth of Dt](figs/dt_pspline.png)  
Penalised cubic B-spline fit to minute-level average depth $D_t$.

![Power-law fit to g(x)](figs/gt_powerlaw.png)  
Filtered average premium $g(x)$ with tail $x^{p}$ (log–log).

---

## Method (at a glance)

1. **Depth curve $D_t$**  
   For each trading minute $t\in\{0,\dots,389\}$, compute the average of the first non-zero ask size per quote event for each symbol–day, then average across panels to get $D_t$. Smooth with a penalized cubic B-spline (GCV selects $\lambda$).

2. **Flat cost $c$**  
   Take the median half-spread per symbol during RTH, then the median across symbols → global $c$.

3. **Impact tail exponent $p$**  
   Build empirical $g(x)$ by averaging the premium paid above best-ask as one walks up the ask ladder; fit  
   $\log g(x)=\log a + p\log x$ on filtered buckets → $p\approx 0.45$.

4. **Piecewise temporary impact $g_t(x)$**

```math
g_t(x)=
\begin{cases}
c, & 0\le x\le D_t\\
a_t\,x^{\,p}, & x>D_t
\end{cases}
\qquad
a_t=\dfrac{c}{D_t^{\,p}},\quad 0<p<1.
