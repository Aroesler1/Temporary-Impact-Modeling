# Return response audit, 2026-09-06

Scope: the already published fifteen symbol-days of MSFT, INTC and AAPL in
2024. This is an algebraic audit of committed fitted aggregates, not a new
holdout experiment, a new fit, or a causal impact estimate. The issue and the
first cumulative values were found during the read-only review. They are not
retrospectively described as a preregistered hypothesis.

Before implementing the correction, the required falsifiable checks were fixed:

1. An isolated trade with return response `[1, 0, 0]` must leave level response
   `[1, 1, 1]` on that support. Calling this complete relaxation must fail.
2. Return response `[1, -0.5, -0.25]` must recover a decaying level response
   `[1, 0.5, 0.25]`. Convolving levels must match accumulating predicted returns.
3. A finite fitted horizon must remain finite. No assumed permanent or vanishing
   tail, synthetic confidence interval, or tuned schedule is introduced.

The old regression is `r_t = sum_l b_l f(v_{t-l})`. Its dependent variable is a
log return. The corresponding level displacement at lag k is
`R_k = sum_{l=0}^k b_l`, conditional on the other flow being held fixed. The
existing conditional-impact code already sums returns through each order and
is not affected by this representation error. Its hyperparameter selection is
also nested inside training.

The scheduling path was affected: it supplied b directly to a level-cost
Toeplitz matrix, then priced with an exponential of the same return convolution.
There were two additional mismatches: its quadratic charges half the diagonal
where the replay charges full instantaneous displacement, and it truncates a
600-second level response at the fitted return lag. Cumulative b would not
justify that truncation. The empirical schedule runner now refuses to produce a
result. The archived CSVs remain available as historical evidence, explicitly
withdrawn. Restoring the runner requires an explicit execution-price convention,
a supported level horizon, cost consistency tests, and a fresh evaluation plan.

The 100 ms input is a mean of fifteen per-session normalized return kernels.
Cumulative summation commutes with that mean, so the diagnostic level mean is
exactly recoverable. Per-session kernel vectors and their cross-lag covariance
are not committed. Summing marginal confidence limits would not yield a valid
confidence interval, so none is published. The corrected curve covers only
lags 0 through 20, or 0 through 2 seconds. Selection of delta used the same
last 30% subsequently reported as R2. These are selected-validation diagnostics,
not untouched OOS results. No new specification was selected on that tail.

Reproduce with `python scripts/audit_kernel_response.py --check`. Inputs, hashes,
source revision, the full curve, and corrected crossover denominators are in
`reports/kernel_audit/`. The old interpretation failed; a statistically
identified relaxation horizon and an execution saving remain unestablished.
