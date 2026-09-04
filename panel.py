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
    """One-second bars joined to the one-second OFI series.

    An inner join on `sec`: the two builders both drop seconds the feed never
    spoke in, but MBO and MBP-10 do not always speak in exactly the same ones,
    and a regression that silently zero-filled the difference would report flow
    of zero where the truth is no observation.
    """
    frame = pd.read_csv(DATA / f"{session}_1s.csv")
    ofi_path = DATA / "ofi" / f"{session}_1s_ofi.csv"
    if ofi_path.exists():
        frame = frame.merge(pd.read_csv(ofi_path), on="sec", how="inner")
    return frame.sort_values("sec").reset_index(drop=True)


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
