"""Stratified sampling and per-stock fitting for the cross-sectional study.

SCOPE, WHICH IS NARROW ON PURPOSE
---------------------------------
S&P 500 members on 2024-06-28, Nasdaq venue only, April to September 2024. That
is large caps, one venue, one half-year, and ordinary days. It is not a
market-wide statement and this module does not make one.

WHY STRATIFY AT ALL
-------------------
The hypothesis under test is that the fitted exponent falls with relative tick
size, because one tick floors the impact of small orders and flattens the fitted
slope. Relative tick size and dollar volume are strongly negatively correlated
in the S&P 500: cheap stocks are usually also heavily traded. A simple random
sample would therefore load the tick-size axis and the liquidity axis together
and leave the regression unable to separate them. Terciles on each, crossed,
force names into the off-diagonal cells where the two disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SAMPLE_SEED = 20240628          # the membership date, written down before drawing
NAMES_PER_CELL = 12
TERCILE_LABELS = ("low", "mid", "high")


@dataclass
class Stratification:
    frame: pd.DataFrame          # one row per candidate, with cell labels
    sample: pd.DataFrame         # the drawn names
    cell_counts: pd.DataFrame    # available and drawn, per cell
    seed: int
    short_cells: list[str]       # cells that could not supply NAMES_PER_CELL


def tercile(values: pd.Series) -> pd.Series:
    """Tercile label by rank, so ties and skew cannot empty a bucket.

    `pd.qcut` on the raw values would put every mega-cap in one bucket when
    dollar volume spans four orders of magnitude; ranking first makes the three
    buckets equal in COUNT, which is what a stratified draw needs.
    """
    ranked = values.rank(method="first", pct=True)
    return pd.cut(ranked, bins=[0.0, 1 / 3, 2 / 3, 1.0],
                  labels=list(TERCILE_LABELS), include_lowest=True)


def stratify(stats: pd.DataFrame, names_per_cell: int = NAMES_PER_CELL,
             seed: int = SAMPLE_SEED) -> Stratification:
    """Nine cells on relative tick size by dollar volume, `names_per_cell` each.

    `stats` needs symbol, relative_tick and dollar_volume. A cell with fewer
    members than requested contributes all of them and is reported in
    `short_cells` rather than being quietly padded from a neighbour.
    """
    for column in ("symbol", "relative_tick", "dollar_volume"):
        if column not in stats.columns:
            raise ValueError(f"missing column: {column}")
    frame = stats.dropna(subset=["relative_tick", "dollar_volume"]).copy()
    frame = frame[(frame.relative_tick > 0) & (frame.dollar_volume > 0)]
    frame["tick_tercile"] = tercile(frame.relative_tick)
    frame["volume_tercile"] = tercile(frame.dollar_volume)
    frame["cell"] = (frame.tick_tercile.astype(str) + "_tick__"
                     + frame.volume_tercile.astype(str) + "_volume")

    rng = np.random.default_rng(seed)
    drawn, counts, short = [], [], []
    for cell, group in frame.groupby("cell", observed=True, sort=True):
        take = min(names_per_cell, len(group))
        # sort before drawing so the seed reproduces the same names whatever
        # order the input arrived in
        ordered = group.sort_values("symbol")
        picks = rng.choice(ordered.symbol.to_numpy(), size=take, replace=False)
        drawn.append(ordered[ordered.symbol.isin(picks)])
        counts.append({"cell": cell, "available": len(group), "drawn": take})
        if take < names_per_cell:
            short.append(cell)

    sample = pd.concat(drawn, ignore_index=True).sort_values("symbol")
    return Stratification(frame.reset_index(drop=True),
                          sample.reset_index(drop=True),
                          pd.DataFrame(counts), seed, short)


# --------------------------------------------------------------------------
# trades to bars
# --------------------------------------------------------------------------

RTH_OPEN_SEC = 34_200        # 09:30:00 exchange time
RTH_CLOSE_SEC = 57_600       # 16:00:00
_NS_PER_S = 1_000_000_000
FIVE_MINUTES = 300
BINS_PER_SESSION = (RTH_CLOSE_SEC - RTH_OPEN_SEC) // FIVE_MINUTES   # 78


def aggressor_sign(side: np.ndarray) -> np.ndarray:
    """+1 for a buy aggressor, -1 for a sell, 0 where the feed gives no side.

    THE TRADES SCHEMA USES THE OPPOSITE CONVENTION FROM MBO, and this is the
    single most dangerous line in the cross-section. On MBO an execution's
    `side` is the side of the RESTING order, which is why
    `databento_to_lobster.py` maps side B to LOBSTER direction +1 and
    `build_1s_bars.py` then negates it to get the aggressor. On the `trades`
    schema `side` is the AGGRESSING side already, so B is a buy and A is a sell
    with no negation.

    Measured, not assumed: signed volume built the resting way correlates
    -0.9999 with the MBO-derived series on all five committed AAPL sessions, and
    +0.9999 built this way. A flipped sign would invert every metaorder in the
    study and still produce a plausible-looking exponent, so
    `scripts/validate_trade_bars.py` runs the check and refuses to pass on
    anything between -0.9 and +0.9.
    """
    side = np.asarray(side)
    return np.where(side == b"B", 1.0, np.where(side == b"A", -1.0, 0.0))


def trade_bars(sec: np.ndarray, price: np.ndarray, size: np.ndarray,
               sign: np.ndarray) -> pd.DataFrame:
    """One-second bars from trades alone: signed volume, volume, last price.

    The price column is named `mid` so `crossover.bin_metaorders` can consume it
    unchanged, but it is the LAST TRADE PRICE in the bin, not a mid. The trades
    schema carries no quotes. Every impact measured from it therefore includes
    whatever bid-ask bounce falls between the first and last print of a run,
    which is a real difference from the three-name study and is stated in the
    README rather than hidden behind the column name.
    """
    frame = pd.DataFrame({"sec": sec.astype(np.int64), "price": price,
                          "size": size, "signed": sign * size})
    grouped = frame.groupby("sec").agg(signed_vol=("signed", "sum"),
                                       volume=("size", "sum"),
                                       mid=("price", "last"))
    return grouped.reset_index()


def realised_vol_5min(sec: np.ndarray, price: np.ndarray) -> float:
    """Daily volatility from five-minute trade prices, scaled to a session.

    Five minutes rather than one second because a one-second trade-price series
    is dominated by bid-ask bounce, which is not volatility and would inflate
    sigma_D by a factor that varies with the tick size. Since tick size is the
    regressor under test, using a bounce-contaminated sigma would plant the
    result being looked for.
    """
    frame = pd.DataFrame({"bin": (np.asarray(sec, np.int64) - RTH_OPEN_SEC)
                          // FIVE_MINUTES, "price": price})
    closes = frame.groupby("bin").price.last().sort_index().to_numpy(float)
    if len(closes) < 10:
        return float("nan")
    returns = np.diff(np.log(closes))
    return float(np.std(returns, ddof=1) * np.sqrt(BINS_PER_SESSION))


HALF_HOUR = 1800
N_HALF_HOURS = (RTH_CLOSE_SEC - RTH_OPEN_SEC) // HALF_HOUR   # 13


def halfhour_vol_profile(sec: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Five-minute realised volatility in each half hour, over the whole day's.

    A shape, not a level, so it can be averaged across a symbol's sessions. Six
    five-minute returns per half hour is noisy for one day and steady across a
    hundred and twenty, which is why the caller takes the median over days.

    Returns NaN for a half hour with too few returns to estimate rather than a
    number that would silently be one observation.
    """
    frame = pd.DataFrame({"bin": (np.asarray(sec, np.int64) - RTH_OPEN_SEC)
                          // FIVE_MINUTES, "price": price})
    closes = frame.groupby("bin").price.last().sort_index()
    if len(closes) < 10:
        return np.full(N_HALF_HOURS, np.nan)
    returns = pd.Series(np.diff(np.log(closes.to_numpy(float))),
                        index=closes.index[1:])
    whole = float(returns.std(ddof=1))
    if not np.isfinite(whole) or whole <= 0:
        return np.full(N_HALF_HOURS, np.nan)
    bucket = (returns.index.to_numpy() * FIVE_MINUTES) // HALF_HOUR
    out = np.full(N_HALF_HOURS, np.nan)
    for b, group in pd.Series(returns.to_numpy()).groupby(bucket):
        if 0 <= b < N_HALF_HOURS and len(group) >= 3:
            out[int(b)] = float(group.std(ddof=1) / whole)
    return out


# --------------------------------------------------------------------------
# cross-sectional regression
# --------------------------------------------------------------------------

@dataclass
class OLSResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    r_squared: float
    n: int

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names, "coef": self.coef,
                             "se_robust": self.se, "t": self.tstat,
                             "significant_5pct": np.abs(self.tstat) > 1.96})


def ols_robust(y: np.ndarray, X: pd.DataFrame) -> OLSResult:
    """OLS with HC1 heteroskedasticity-robust standard errors.

    HC1 rather than HC0 because the cross-section is about a hundred names and
    the small-sample correction n/(n-k) is not negligible there. Written out
    rather than pulled from statsmodels to keep the dependency list at what the
    repository already had.
    """
    y = np.asarray(y, float)
    design = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    ok = np.isfinite(y) & np.isfinite(design).all(axis=1)
    y, design = y[ok], design[ok]
    n, k = design.shape
    if n <= k:
        raise ValueError(f"{n} usable rows for {k} parameters")

    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    meat = design.T @ (resid[:, None] ** 2 * design)
    cov = xtx_inv @ meat @ xtx_inv * (n / (n - k))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    return OLSResult(["intercept"] + list(X.columns), beta, se, tstat, r2, n)


def consolidated_normalisers(symbols, start: str, end: str) -> pd.DataFrame:
    """CRSP consolidated volume and trailing close-to-close volatility.

    NOT IMPLEMENTED, and deliberately a raising stub rather than a silent
    fallback. Every V_D and sigma_D in this study is single-venue Nasdaq, which
    is the same feed arXiv 2606.24019 used, so the comparison with the published
    prefactor is like for like. The consolidated variant is a SECOND normaliser
    that would change the level of c and not the exponent, and it is left as a
    pending row in the README.

    WRDS was refusing logins from this machine when the cross-section was built,
    so nothing here depends on it. When WRDS is reachable this should query CRSP
    daily stock file volume and returns for the same symbols and window.
    """
    raise NotImplementedError(
        "the consolidated CRSP normaliser is a pending row. WRDS was "
        "unreachable when this branch was built and no result in the "
        "cross-section depends on it. See the pending row in README.md.")
