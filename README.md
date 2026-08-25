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
- **Scope, stated honestly.** Walking displayed snapshots measures the *virtual instantaneous* cost of consuming visible liquidity. Realized metaorder impact additionally reflects hidden liquidity, queue refill, and adverse selection, and empirically scales with participation of traded volume (the square-root law, impact $\approx \sigma\sqrt{Q/V}$). The calibrated curve here is a displayed-liquidity lower bound. Natural extensions: reformulate $g_t$ in participation of the minute-volume curve, add an Almgren-Chriss risk term, and compare the schedule against TWAP/VWAP/POV baselines.
- **Allocator verified.** The per-minute cost is convex (marginal cost steps from $c$ to $(1+p)c$ at $D_t$), so the KKT/bisection solution is globally optimal; tests confirm no pairwise mass transfer improves it and that it matches a fine greedy marginal-cost discretization. The flat-region allocation is degenerate; proportional-to-$D_t$ is the documented tie-break.

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
