"""The fixed session list for this study, and the shared raw-data layout.

The selection rule was fixed BEFORE any result was looked at, which is the
only thing that makes the 15-fold cross-validation in `bookwalk.py` and the
per-session ranges in the README honest:

* keep the two sessions the earlier two-day study used, MSFT 2024-06-03 and
  INTC 2024-08-02;
* add the first trading day of February, April, October and December 2024 for
  MSFT and INTC;
* add the first trading day of February, April, June, August and October 2024
  for AAPL.

AAPL is in the panel so the metaorder result sits beside a published one
(arXiv 2606.24019 reports Apple).

Raw vendor files are NEVER referenced by a committed absolute path. They live
under $DATABENTO_RAW_DIR in the layout

    $DATABENTO_RAW_DIR/<SYMBOL>/<YYYY-MM-DD>.<schema>.dbn.zst

shared with the sibling `lob-engine-cpp` repository so one pull serves both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# first trading day of the named month in 2024 (2024-06-01 and 2024-12-01 are
# weekends, so those roll to the Monday)
_FIRST_TRADING_DAY = {
    "feb": "2024-02-01",
    "apr": "2024-04-01",
    "jun": "2024-06-03",
    "aug": "2024-08-01",
    "oct": "2024-10-01",
    "dec": "2024-12-02",
}

SESSIONS: tuple[tuple[str, str], ...] = tuple(
    [("MSFT", "2024-06-03")]                                   # kept
    + [("MSFT", _FIRST_TRADING_DAY[m]) for m in ("feb", "apr", "oct", "dec")]
    + [("INTC", "2024-08-02")]                                 # kept
    + [("INTC", _FIRST_TRADING_DAY[m]) for m in ("feb", "apr", "oct", "dec")]
    + [("AAPL", _FIRST_TRADING_DAY[m]) for m in ("feb", "apr", "jun", "aug", "oct")]
)

SCHEMAS = ("mbo", "mbp-10")
DATASET = "XNAS.ITCH"


@dataclass(frozen=True)
class Session:
    symbol: str
    date: str

    @property
    def key(self) -> str:
        return f"{self.symbol}_{self.date}"

    def raw_path(self, schema: str, root: Path | None = None) -> Path:
        return (root or raw_root()) / self.symbol / f"{self.date}.{schema}.dbn.zst"


def sessions() -> list[Session]:
    return [Session(sym, date) for sym, date in SESSIONS]


def raw_root() -> Path:
    """Shared Databento raw directory, from the environment only."""
    root = os.environ.get("DATABENTO_RAW_DIR")
    if not root:
        raise SystemExit(
            "DATABENTO_RAW_DIR is not set. It must point at the shared raw "
            "directory, e.g. ~/Data/databento/XNAS.ITCH. Raw vendor paths are "
            "never committed.")
    return Path(root).expanduser()
