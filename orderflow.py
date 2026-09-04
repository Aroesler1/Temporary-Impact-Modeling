"""Order flow imbalance beside trade flow, in the same regressions.

WHERE THIS CODE COMES FROM
--------------------------
`level_ofi` and `integrate_pca` are VENDORED, not reinvented: they are the
transition form and the PCA integration from `scripts/multi_level_ofi.py` in the
sibling `lob-engine-cpp` repository, which in turn implements Cont, Kukanov and
Stoikov (2014) for the per-level increment and Cont, Cucuringu and Zhang
(Quantitative Finance 2023) for the multi-level integration. They are copied
here rather than imported because this repo must run from its own checkout with
no sibling on the path, and copied VERBATIM in substance so that a result
computed here and a result computed there are the same estimator. Two changes,
both noted at the call site: the depth normalisation of Cont, Cucuringu and
Zhang is applied here, and the integration is fitted on TRAIN-ONLY rows so the
out-of-sample split stays honest.

WHY BOTH FLOWS
--------------
Signed trade volume is what the propagator regresses on. It counts only
executions. Order flow imbalance counts the whole displayed book's arrivals,
cancellations and executions, so it moves when a quote is pulled and no trade
happens at all. The two are different measurements of the same pressure and the
interesting question is not which wins but how much each adds GIVEN the other,
which is what `incremental_r2` reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def level_ofi(bid_px: np.ndarray, bid_sz: np.ndarray,
              ask_px: np.ndarray, ask_sz: np.ndarray) -> np.ndarray:
    """Per-event OFI for one level, from consecutive book states.

    Cont, Kukanov and Stoikov (2014):

        e = 1{Pb >= Pb'} qb - 1{Pb <= Pb'} qb'
          - 1{Pa <= Pa'} qa + 1{Pa >= Pa'} qa'

    A bid that improves contributes its whole new size, a bid that is pulled
    contributes minus the old size, and a bid that stays at the same price
    contributes the change. Vendored from lob-engine-cpp/scripts/multi_level_ofi.py.
    """
    e = np.zeros(len(bid_px))
    e[1:] = (
        (bid_px[1:] >= bid_px[:-1]) * bid_sz[1:]
        - (bid_px[1:] <= bid_px[:-1]) * bid_sz[:-1]
        - (ask_px[1:] <= ask_px[:-1]) * ask_sz[1:]
        + (ask_px[1:] >= ask_px[:-1]) * ask_sz[:-1]
    )
    return np.nan_to_num(e)


def integrate_pca(ofi: np.ndarray, fit_rows: slice | np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """First principal component of the per-level OFIs, sign-aligned to level 0.

    Standardising first matters: level sizes differ by an order of magnitude
    across the book, and without it the component is dominated by whichever
    level happens to be deepest rather than by common flow. Vendored from
    lob-engine-cpp/scripts/multi_level_ofi.py, with one change -- the component
    may be fitted on a subset of rows (the training window) and applied to all
    of them, so a predictive score is not contaminated by a rotation chosen with
    the evaluation window in hand.
    """
    fit = ofi if fit_rows is None else ofi[fit_rows]
    mu, sd = fit.mean(axis=0), fit.std(axis=0) + 1e-12
    z_fit = (fit - mu) / sd
    _, _, vt = np.linalg.svd(z_fit - z_fit.mean(axis=0), full_matrices=False)
    w = vt[0]
    if w[0] < 0:
        w = -w
    return ((ofi - mu) / sd) @ w, w


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------

def _ols_r2(y: np.ndarray, X: np.ndarray, split: int) -> tuple[float, float]:
    """In-sample and out-of-sample R2 of y on X (intercept added here)."""
    X = np.column_stack([np.ones(len(X)), X])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    split = int(len(y) * split / 100) if split > 1 else int(len(y) * split)
    if split < 50 or len(y) - split < 50:
        return float("nan"), float("nan")
    beta, *_ = np.linalg.lstsq(X[:split], y[:split], rcond=None)

    def r2(yy, xx):
        resid = yy - xx @ beta
        sst = float(np.sum((yy - yy.mean()) ** 2))
        return 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else float("nan")

    return r2(y[:split], X[:split]), r2(y[split:], X[split:])


@dataclass
class FlowComparison:
    session: str
    relation: str          # contemporaneous | predictive
    r2_trade: float
    r2_ofi: float
    r2_both: float
    incremental_trade: float   # both minus ofi alone
    incremental_ofi: float     # both minus trade alone
    n: int


def compare_flows(frame: pd.DataFrame, session: str, train_frac: float = 0.7,
                  ofi_col: str = "ofi_integrated") -> list[FlowComparison]:
    """Signed trade volume against OFI, alone and together, in both relations.

    Contemporaneous regresses the return over a second on flow in the SAME
    second; predictive regresses it on flow in the PREVIOUS second. Both are
    scored out of sample on the same chronological split the propagator uses, so
    a number here and a number there mean the same thing.
    """
    mid = pd.to_numeric(frame["mid"], errors="coerce").to_numpy(float)
    ret = np.full(len(mid), np.nan)
    ret[1:] = np.log(mid[1:] / mid[:-1])
    trade = pd.to_numeric(frame["signed_vol"], errors="coerce").to_numpy(float)
    ofi = pd.to_numeric(frame[ofi_col], errors="coerce").to_numpy(float)

    out = []
    for relation, lag in (("contemporaneous", 0), ("predictive", 1)):
        t = np.roll(trade, lag).astype(float)
        o = np.roll(ofi, lag).astype(float)
        if lag:
            t[:lag] = np.nan
            o[:lag] = np.nan
        _, r_t = _ols_r2(ret, t[:, None], train_frac)
        _, r_o = _ols_r2(ret, o[:, None], train_frac)
        _, r_b = _ols_r2(ret, np.column_stack([t, o]), train_frac)
        n = int(np.sum(np.isfinite(ret) & np.isfinite(t) & np.isfinite(o)))
        out.append(FlowComparison(session, relation, r_t, r_o, r_b,
                                  r_b - r_o, r_b - r_t, n))
    return out
