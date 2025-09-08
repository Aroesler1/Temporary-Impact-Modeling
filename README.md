# Intraday Liquidity & Optimal Execution (Work-Trial Task)

This repo builds an intraday liquidity curve from quote data, calibrates a **piecewise temporary impact** model (flat cost up to depth; concave power-law tail), and solves for the **optimal minute-by-minute execution schedule** for a fixed parent order using KKT conditions with a bisection on the Lagrange multiplier.

![Intraday Average Dt](figs/dt_intraday.png)

---

## What’s in here

- **Modeling note** — estimating the flat cost $c$, intraday depth $D_t$, and tail exponent $p$, with plots and rationale.  
- **Allocation note** — KKT-based algorithm with a bisection on the multiplier to hit the target $S$.  
- **Notebook / code** — end-to-end data loading, P-spline smoothing, power-law fit, and final allocation.  
- **Task PDF** — original problem statement for context.

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
