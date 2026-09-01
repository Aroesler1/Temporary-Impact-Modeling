# Data provenance

**Two sources, with different standing.**

1. **Databento `XNAS.ITCH` MBO** (Berkeley MFE programme account) — used for the transient-impact propagator. The committed one-second series is a derived aggregate, not raw vendor data.
2. **Work-trial MBP-10 snapshots** (SOUN/FROG/CRWV) — proprietary data supplied for an interview exercise. Not redistributable, and no published result depends on it.

## What is committed

- Source code, tests, notebooks, and figures
- `data/MSFT_2024-06-03_1s.csv`: 23,390 rows of one-second signed volume and mid price, derived from Databento MBO. Small, aggregated, and sufficient to reproduce the propagator calibration end to end.
- Derived results under `reports/`

## What is not committed

- `Work_Trial/` (gitignored): ~2.7 GB of proprietary MBP-10 snapshots
- Raw Databento extracts

## Reproducing

```bash
python scripts/run_propagator.py --data data/MSFT_2024-06-03_1s.csv
python -m pytest tests -q
```

The propagator study needs no credentials; the committed derived series is enough.

## Licence and retention

Databento data is accessed under a programme licence; only a small derived aggregate is published. The work-trial data is proprietary to its provider and is neither published nor relied upon for any stated result.
