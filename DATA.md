# Data provenance

**Two sources, with different standing.**

1. **Databento `XNAS.ITCH` MBO** (Berkeley MFE programme account) — used for the transient-impact propagator. The committed one-second series is a derived aggregate, not raw vendor data.
2. **Work-trial MBP-10 snapshots** (SOUN/FROG/CRWV) — proprietary data supplied for an interview exercise. Not redistributable, and no published result depends on it.

## What is committed

- Source code, tests, notebooks, and figures
- `data/MSFT_2024-06-03_1s.csv` (23,390 rows) and `data/INTC_2024-08-02_1s.csv` (23,394 rows): one-second signed volume and mid price, derived from Databento MBO. Small, aggregated, and sufficient to reproduce the propagator calibration end to end.
- `scripts/build_1s_bars.py`, which regenerates either series from a Databento extract. Until 2026-09 these CSVs were committed with no builder behind them, which made the headline R² unreproducible from source; that gap is now closed, and running the builder on MSFT reproduces the committed file exactly.
- Derived results under `reports/`

## Session selection

Two symbol-days, chosen to differ in kind rather than to repeat a regime.

| session | character | one tick, in bp of mid | share of session at a one-tick spread | median touch, bid / ask |
|---|---|---:|---:|---:|
| MSFT 2024-06-03 | small-tick, spread-dominated | 0.24 bp | 1.0% | 68 / 63 |
| INTC 2024-08-02 | large-tick, queue-dominated | 4.73 bp | 81.1% | 2,415 / 3,000 |

Both extracts cost $0.00 against the programme entitlement, which covers
`XNAS.ITCH` outright.

## Known provenance gap

`data/MSFT_2024-06-03_metaorders.csv` predates the execution-mirror-cancel fix
in the sibling `lob-engine-cpp` repository and has not been rebuilt; there is no
committed builder for it, so its construction cannot be verified. See the
provenance caveat in the metaorder section of `README.md`.

## What is not committed

- `Work_Trial/` (gitignored): ~2.7 GB of proprietary MBP-10 snapshots
- Raw Databento extracts

## Reproducing

```bash
python scripts/run_propagator.py --data data/MSFT_2024-06-03_1s.csv
python scripts/run_propagator.py --data data/INTC_2024-08-02_1s.csv
python -m pytest tests -q
```

Rebuilding a bar series from raw vendor data additionally needs the sibling
`lob-engine-cpp` repo, which produces the LOBSTER message stream and the
reconstructed L1 book that `scripts/build_1s_bars.py` consumes.

The propagator study needs no credentials; the committed derived series is enough.

## Licence and retention

Databento data is accessed under a programme licence; only a small derived aggregate is published. The work-trial data is proprietary to its provider and is neither published nor relied upon for any stated result.
