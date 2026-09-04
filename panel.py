"""One place that knows where the panel's derived series live and how to join them.

Every run script in this repo works on the same fifteen symbol-days, so the
loading, the joins and the session-level scale constants are here rather than
repeated with small differences in each script. All of it reads committed
derived data; none of it needs the vendor SDK or the raw extracts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
TRAIN_FRAC = 0.7  # the chronological split used by every out-of-sample number


@lru_cache(maxsize=1)
def meta() -> pd.DataFrame:
    """Session-level scales: volume, volatility, spread, depth."""
    return pd.read_csv(DATA / "session_meta.csv")


def session_keys() -> list[str]:
    return meta().session.tolist()


def scales(session: str) -> pd.Series:
    row = meta().set_index("session")
    if session not in row.index:
        raise KeyError(f"unknown session {session!r}")
    return row.loc[session]


@lru_cache(maxsize=32)
def bars(session: str) -> pd.DataFrame:
    """The committed one-second series, exactly as `build_1s_bars.py` wrote it.

    Deliberately NOT joined to the OFI series: MBO and MBP-10 do not speak in
    exactly the same seconds, so joining drops a handful of rows and would move
    every propagator number by a few units in the fifth decimal against the
    figures the README quotes. Part 4 needs both flows on one grid and asks for
    `bars_with_ofi` explicitly.
    """
    return pd.read_csv(DATA / f"{session}_1s.csv").sort_values("sec").reset_index(
        drop=True)


@lru_cache(maxsize=32)
def bars_with_ofi(session: str) -> pd.DataFrame:
    """One-second bars inner-joined to the one-second OFI series.

    An inner join on `sec`: both builders drop seconds the feed never spoke in,
    and zero-filling the difference would report flow of zero where the truth is
    no observation.
    """
    frame = bars(session)
    ofi_path = DATA / "ofi" / f"{session}_1s_ofi.csv"
    if not ofi_path.exists():
        raise FileNotFoundError(ofi_path)
    return frame.merge(pd.read_csv(ofi_path), on="sec", how="inner").reset_index(
        drop=True)


@lru_cache(maxsize=32)
def metaorders(session: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{session}_metaorders.csv")


@lru_cache(maxsize=32)
def bookwalk_bins(session: str) -> pd.DataFrame:
    return pd.read_csv(DATA / "bookwalk" / f"{session}_bins.csv")


def returns(frame: pd.DataFrame) -> np.ndarray:
    """One-second log returns of the mid, first entry NaN."""
    mid = pd.to_numeric(frame["mid"], errors="coerce").to_numpy(float)
    out = np.full(len(mid), np.nan)
    out[1:] = np.log(mid[1:] / mid[:-1])
    return out


def split_index(n: int, train_frac: float = TRAIN_FRAC) -> int:
    return int(n * train_frac)


def bookwalk_panel(side: str = "buy", normalisation: str = "x_over_depth",
                   cost: str = "cost_half_spreads",
                   min_participation: float = 0.0) -> pd.DataFrame:
    """Pooled bin table in fitter form: session, u, y, w, participating.

    `min_participation` drops bins where fewer than that fraction of snapshots
    had displayed depth to fill the size, which is the truncation-bias control
    described in `bookwalk`.
    """
    rows = []
    for key in session_keys():
        table = bookwalk_bins(key)
        sub = table[(table.side == side)
                    & (table.size_normalisation == normalisation)].copy()
        sub = sub[sub.participating >= min_participation]
        rows.append(pd.DataFrame({
            "session": key,
            "u": sub["size"].to_numpy(float),
            "y": sub[cost].to_numpy(float),
            "w": sub["n"].to_numpy(float),
            "participating": sub["participating"].to_numpy(float),
        }))
    out = pd.concat(rows, ignore_index=True)
    return out[np.isfinite(out.u) & np.isfinite(out.y) & (out.u > 0)].reset_index(drop=True)
