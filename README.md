# Intraday Liquidity & Optimal Execution (Work-Trial Task)

This repo builds an *intraday liquidity curve* from TAQ-style quote data, calibrates a **piecewise temporary impact** model \(flat cost up to depth; concave power-law tail\), and solves for the **optimal minute-by-minute execution schedule** for a fixed parent order using KKT conditions with a bisection on the Lagrange multiplier.

<p align="center">
  <img src="figs/dt_intraday.png" alt="Intraday Average Dt" width="85%">
</p>

---

## What’s in here

- **Modeling note** — how we estimate the flat cost \(c\), intraday depth \(D_t\), and power-law tail exponent \(p\), with plots and rationale. :contentReference[oaicite:0]{index=0}  
- **Allocation note** — the KKT-based algorithm, closed-form inversion, and a robust bisection procedure to hit the share target \(S\). :contentReference[oaicite:1]{index=1}  
- **Notebook / code** — end-to-end data loading, smoothing with a penalized B-spline, power-law fit on the impact curve, and the final allocation. :contentReference[oaicite:2]{index=2}  
- **Task PDF** — original problem statement for context. :contentReference[oaicite:3]{index=3}

Key figures (rendered from the notebook):

<p align="center">
  <img src="figs/dt_pspline.png" alt="P-spline Smooth of Dt" width="85%"><br>
  <em>Penalised cubic B-spline fit to minute-level average depth.</em>
</p>

<p align="center">
  <img src="figs/gt_powerlaw.png" alt="Power-law fit to g(x)" width="55%"><br>
  <em>Filtered average premium \(g_t(x)\) with power-law tail \(x^{\,p}\) (log–log).</em>
</p>

---

## Method (at a glance)

1. **Depth curve \(D_t\)**  
   For each trading minute \(t\in\{0,\dots,389\}\), compute the average of the **first non-zero ask size** per quote event for each symbol–day panel; then average across symbols/days to get an intraday series \(D_t\). Smooth it with a penalized cubic B-spline (GCV to select \(\lambda\)). :contentReference[oaicite:4]{index=4}

2. **Flat cost \(c\)**  
   Median half-spread per symbol during RTH, then the median across symbols → a global \(c\) (≈ \$0.03 in the sample). :contentReference[oaicite:5]{index=5}

3. **Impact tail exponent \(p\)**  
   Build an empirical \(g(x)\) by averaging the premium paid above best-ask as one walks up the ask ladder; fit \(\log g(x)=\log a+p\log x\) on filtered buckets → \(p\approx0.45\). :contentReference[oaicite:6]{index=6}

4. **Piecewise temporary impact \(g_t(x)\)**  
   \[
   g_t(x)=
   \begin{cases}
   c, & 0\le x\le D_t\\[2pt]
   a_t\,x^{\,p}, & x>D_t
   \end{cases}
   \qquad \text{with } a_t=\frac{c}{D_t^{\,p}},\; 0<p<1.
   \]
   Flat marginal cost within displayed depth; concave tail beyond it. :contentReference[oaicite:7]{index=7}

5. **Optimal allocation \(x_t\) (KKT + bisection)**  
   Minimize \(\sum_t x_t\,g_t(x_t)\) s.t. \(\sum_t x_t=S,\; x_t\ge 0\).  
   Stationarity yields \( \frac{d}{dx}\big[x\,g_t(x)\big]=\lambda \). Invert:
   \[
   x_t(\lambda)=
   \begin{cases}
   D_t, & \lambda\le c\\[2pt]
   \max\!\Big\{D_t,\; \big(\tfrac{\lambda}{a_t(p+1)}\big)^{\!1/p}\Big\}, & \lambda>c
   \end{cases}
   \]
   Bisection on \(\lambda\) to enforce \(\sum_t x_t(\lambda)=S\). :contentReference[oaicite:8]{index=8}

---
