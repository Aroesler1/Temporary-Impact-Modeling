# Handoff: Temporary-Impact-Modeling

## Current goal (complete as of 2026-09-03)

Recalibrate the transient-impact propagator on a **second** Databento session so
the R² result is not a single symbol-day, and add the framing citations.

## Verified state

`python -m pytest tests -q` — **29 pass** (22 pre-existing + 7 new in
`tests/test_build_1s_bars.py`). Run with `.venv/bin/python`.

| | MSFT 2024-06-03 | INTC 2024-08-02 |
|---|---:|---:|
| one-second bars | 23,390 | 23,394 |
| explanatory OOS R² | 0.36599 | 0.43229 |
| best delta / lags | 0.25 / 60 | 0.25 / 0 |
| predictive OOS R² | 0.00423 | 0.00455 |
| ratio | 86x | 95x |

Same grid, same chronological 70/30 split on both, so they are comparable by
construction. INTC is large-tick and queue-dominated (one-tick spread 81.1% of
the session, 2,400-3,000 at the touch) against MSFT's small-tick,
spread-dominated book (1.0%, 63-68).

## The MSFT number changed, and why

**Explanatory R² was 0.371, is now 0.36599.** Not a re-fit. The committed
`data/MSFT_2024-06-03_1s.csv` had been built from a message stream that predated
the execution-mirror-cancel fix in `~/Desktop/Quant_Projects/lob-engine-cpp`,
which double-decremented resting orders on every displayed fill. That corrupted
**253 of 23,390 mid values** (max error $0.035). `signed_vol` was unaffected —
it reads only displayed-fill events, which the fix does not touch. Rebuilt from
the corrected book; the CSV in the repo is now the corrected one.

Root cause of the class of problem: the series was committed with **no builder
behind it**, so a stale input could not be detected. Closed by
`scripts/build_1s_bars.py`, which reproduces the corrected MSFT series exactly
and was verified against the committed file before being trusted.

## Open item — the one thing left undone

`data/MSFT_2024-06-03_metaorders.csv` shares that stale lineage and has **not**
been rebuilt. Its `mid_start` / `mid_end` come from the same book, so the 0.788
exponent in the metaorder section is very likely affected the same way.

It was flagged rather than silently corrected because there is no committed
builder for it and the recovered conventions do **not** reproduce it from either
book (mid_start matched 8,354/9,439 against the stale book, 7,788/9,439 against
the corrected one), so its exact construction cannot be verified and re-deriving
it would mean guessing. Flagged in `README.md` and `DATA.md`.

**Next action if resumed:** write `scripts/build_metaorders.py` the same way the
bar builder was written — recover the convention until it reproduces the
committed file from the *stale* book, then re-run against the *corrected* book
and report the movement. Anything less leaves a published exponent resting on a
known-buggy input.

## Bar convention (recovered and pinned by tests)

- RTH only, LOBSTER seconds `[34200, 57600)`.
- One row per second carrying **at least one message**; seconds with none are
  absent, not zero-filled (drops 10 of 23,400 on MSFT).
- `signed_vol` = sum of `-direction * size` over event type **4 only**. Type 5
  hidden prints excluded — including them shifts a second by up to 366,836
  shares on MSFT.
- `mid` = **last** mid in the second, from the engine's reconstructed L1 book.

## Gotchas

- `DATABENTO_API_KEY` is in `~/.zshrc`; non-interactive shells do not source it.
  Use `zsh -ic`, not `zsh -lc`.
- Rebuilding a series needs the sibling `lob-engine-cpp` repo for the LOBSTER
  message stream and the L1 book. `.venv` has no `databento`.
