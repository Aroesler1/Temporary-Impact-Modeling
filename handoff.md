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

## Metaorder file: rebuilt (was the open item, now closed)

`data/MSFT_2024-06-03_metaorders.csv` has been rebuilt by
`scripts/build_metaorders.py`. **The fitted exponent moves 0.788 -> 0.370.**

Recovering the construction found two defects in the original file, neither
reproduced:

1. Fills sharing a timestamp were reordered. The old file is not time-sorted --
   it has a run starting 59 microseconds before the previous one ended, and it
   splits one same-signed run into three. Message order is sequence order.
2. `mid_start` was read AFTER the run's first fill instead of before it, silently
   dropping that fill's impact from every metaorder. This is the change that
   moves the number.

Decomposition (same 12-bin fit):

    committed file, as published                          0.788   R^2 0.948
    rebuilt, pre-fix book,   mid 'at'  first fill         0.622   R^2 0.964
    rebuilt, corrected book, mid 'at'  first fill         0.621   R^2 0.965
    rebuilt, corrected book, mid 'before' the run         0.370   R^2 0.917

**The execution-mirror-cancel fix is irrelevant here** -- 0.001 of exponent. The
earlier provenance caveat blamed the wrong culprit; the mid convention is worth
0.25 on its own. The residual 0.788 -> 0.622 is the reordering and cannot be
decomposed further: no lookup convention reproduces the old file's mid columns
better than 89% (row-index and timestamp-lookup variants both tried), so its
construction is **not recoverable**.

Reading of the new number: 0.370 is below the square-root 0.5, but this is NOT a
refutation. A one-fill metaorder still moves the mid by ~half a tick, so impact
does not fall toward zero as participation does; that discreteness floor lifts
the smallest bins and biases the exponent down. The old `at` convention removed
the floor by accident, which is why it read higher -- by discarding real impact.
Between that floor and a participation range two to four orders of magnitude
below where the law is documented, this session does not test the law cleanly in
either direction. Written up as such.

Also fixed: `ImpactLawFit.describe()` called an exponent below 0.5 "steeper than
the square-root law", which is backwards -- below 0.5 is MORE concave, above is
less. It now names the direction.

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
