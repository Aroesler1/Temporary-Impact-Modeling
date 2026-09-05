# Data provenance

Everything published here rests on one vendor source and one entitlement.

**Databento `XNAS.ITCH`**, MBO and MBP-10, accessed under the Berkeley MFE
programme account, which covers the dataset outright. Fifteen symbol-days on
three names. Raw extracts are never committed and never referenced by a path
written into the repository: they live under `$DATABENTO_RAW_DIR` in the layout

    $DATABENTO_RAW_DIR/<SYMBOL>/<YYYY-MM-DD>.<schema>.dbn.zst

shared with the sibling `lob-engine-cpp` checkout so one pull serves both repos.

The work-trial MBP-10 snapshots (SOUN, FROG, CRWV) that the original notebook
used are proprietary to their provider. They are gitignored, no published result
depends on them any longer, and their figures have been removed from the README.

## Session selection

The rule was fixed **before any result was looked at**, which is the only thing
that makes the fifteen-fold cross-validation and the per-session ranges honest:

* keep the two sessions the earlier two-day study used, MSFT 2024-06-03 and
  INTC 2024-08-02;
* add the first trading day of February, April, October and December 2024 for
  MSFT and INTC;
* add the first trading day of February, April, June, August and October 2024
  for AAPL.

2024-06-01 and 2024-12-01 are weekends, so those roll to the Monday. AAPL is in
the panel so the metaorder result sits beside a published one
([arXiv 2606.24019](https://arxiv.org/abs/2606.24019), which studies AAPL).

The sample contains **no market-wide stress day**. INTC 2024-08-02 is a
single-name event day, the post-earnings collapse, and it is an outlier in most
tables; that is said where it matters rather than hidden.

## Every session, and what it cost

`metadata.get_cost` was queried for all 30 requests before a single byte was
downloaded, and `scripts/fetch_sessions.py` **aborts the pull on any non-zero
price** unless explicitly overridden. Every request priced at $0.0000 against
the programme entitlement.

| symbol | date | MBO get_cost | MBP-10 get_cost | 1s bars | metaorders | traded volume | one tick, bp | one-tick spread | median touch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSFT | 2024-06-03 | $0.0000 | $0.0000 | 23,390 | 9,437 | 5,576,188 | 0.24 | 0.4% | 60 |
| MSFT | 2024-02-01 | $0.0000 | $0.0000 | 23,396 | 12,639 | 8,233,801 | 0.25 | 1.2% | 100 |
| MSFT | 2024-04-01 | $0.0000 | $0.0000 | 23,365 | 8,303 | 5,810,226 | 0.24 | 2.4% | 99 |
| MSFT | 2024-10-01 | $0.0000 | $0.0000 | 23,383 | 13,724 | 7,075,746 | 0.24 | 0.3% | 50 |
| MSFT | 2024-12-02 | $0.0000 | $0.0000 | 23,150 | 6,086 | 6,568,878 | 0.23 | 0.6% | 35 |
| INTC | 2024-08-02 | $0.0000 | $0.0000 | 23,394 | 13,801 | 65,284,503 | 4.72 | 98.2% | 4,208 |
| INTC | 2024-02-01 | $0.0000 | $0.0000 | 23,210 | 4,767 | 9,789,978 | 2.32 | 97.9% | 1,125 |
| INTC | 2024-04-01 | $0.0000 | $0.0000 | 22,562 | 4,087 | 10,477,384 | 2.25 | 98.4% | 956 |
| INTC | 2024-10-01 | $0.0000 | $0.0000 | 23,146 | 3,587 | 21,982,123 | 4.40 | 99.2% | 3,499 |
| INTC | 2024-12-02 | $0.0000 | $0.0000 | 23,208 | 8,162 | 40,020,647 | 4.05 | 98.6% | 2,650 |
| AAPL | 2024-02-01 | $0.0000 | $0.0000 | 23,388 | 20,401 | 16,212,848 | 0.54 | 62.1% | 212 |
| AAPL | 2024-04-01 | $0.0000 | $0.0000 | 23,332 | 9,667 | 15,098,956 | 0.59 | 96.0% | 548 |
| AAPL | 2024-06-03 | $0.0000 | $0.0000 | 23,372 | 19,254 | 18,618,509 | 0.52 | 63.4% | 291 |
| AAPL | 2024-08-01 | $0.0000 | $0.0000 | 23,399 | 24,319 | 21,278,097 | 0.46 | 16.4% | 182 |
| AAPL | 2024-10-01 | $0.0000 | $0.0000 | 23,400 | 28,839 | 20,540,253 | 0.44 | 41.1% | 127 |
Total priced: **$0.0000 across 30 requests.** Daily reference bars (`ohlcv-1d`,
three symbols, 2023-10-01 to 2025-01-01) priced at $0.0000 as well.

Traded volume is the deduplicated MBO tally over the 04:00 to 20:00 window, not
the consolidated tape. Databento reports a displayed execution twice, once as
`T` and once as `F` sharing a sequence, and adding both double counts by about
43%. Two independent checks on that tally:

* MSFT 2024-06-03 comes to 5,576,188, exactly the constant
  `run_metaorder_impact.py` used to hardcode.
* every session matches the vendor's own `ohlcv-1d` volume for that date to
  within about a thousand shares.

Volumes and ADV throughout the repository are **Nasdaq only**, roughly a third
of consolidated volume for these names. Every participation rate is therefore a
share of Nasdaq volume, the column names say so, and the ratio is used
consistently on both sides. The level of the prefactor `c` does depend on that
choice; the exponent does not.

## What is committed

Derived aggregates only, about 28 MB in all.

| path | what | rows |
|---|---|---|
| `data/<KEY>_1s.csv` | one-second signed volume, unsigned volume and mid | ~23,300 a session |
| `data/<KEY>_metaorders.csv` | proxy metaorders: sign, shares, fills, start, end, mid before and after | 3,587 to 28,839 a session |
| `data/ofi/<KEY>_1s_ofi.csv` | one-second best-level, summed and PCA-integrated OFI | ~23,300 a session |
| `data/bookwalk/<KEY>_bins.csv` | binned displayed-ladder cost, both sides, three size normalisations | 240 a session |
| `data/bookwalk/original_recipe.csv` | the notebook's fit and the same fit with each filter lifted | 15 |
| `data/daily_reference.csv` | trailing 20-day ADV and close-to-close volatility per session | 15 |
| `data/session_volume.csv` | deduplicated displayed, hidden and total traded volume | 15 |
| `data/session_meta.csv` | every scale constant the run scripts normalise by | 15 |
| `reports/` | the tables in the README, as written by the run scripts | |

Nothing under `data/` is raw vendor data. No file contains a credential.

### Not committed: the 100 ms bars

`data/bars_100ms/` is gitignored. Fifteen sessions at 100 ms come to **57 MB**,
against 28 MB for everything else here combined, which is past what belongs in a
repository. Rebuilding it from the local MBO extracts takes **613 seconds of
wall clock**:

```bash
python scripts/build_all_sessions.py --bin-ms 100 --out-dir data/bars_100ms
```

`--bin-ms` defaults to 1000, and at that default the builder reproduces the
committed one-second series byte for byte, including the integer dtype of the
`sec` column. A test asserts it. The fitted 100 ms kernels ARE committed, in
`reports/kernel_100ms/`, so the result in section 4 of the README can be read
without rebuilding anything.

## Rebuilding from raw

Four steps, all scripted, and the whole panel takes a few minutes:

```bash
pip install -r requirements-extract.txt
export DATABENTO_RAW_DIR=~/Data/databento/XNAS.ITCH
export LOB_ENGINE_REPO=~/path/to/lob-engine-cpp

python scripts/fetch_sessions.py                  # prices the plan, downloads nothing
python scripts/fetch_sessions.py --confirm        # aborts if get_cost is not $0
python scripts/fetch_daily_reference.py --confirm
python scripts/build_all_sessions.py              # MBO -> messages -> L1 book -> bars + metaorders
python scripts/build_all_sessions.py --bin-ms 100 --out-dir data/bars_100ms   # optional, 613 s, 57 MB
python scripts/build_bookwalk.py                  # MBP-10 -> ladder cost bins
python scripts/build_ofi_bars.py                  # MBP-10 -> one-second OFI
python scripts/build_volume_tally.py
python scripts/build_session_meta.py
```

`build_all_sessions.py` rebuilds MSFT 2024-06-03 through the same four steps as
the fourteen new sessions and diffs the result against the committed series. It
prints **IDENTICAL**, which is what establishes that the panel is built on the
convention the earlier published results were built on. If it ever prints
DIFFERS, the run stops.

The two intermediates (a LOBSTER message stream and a reconstructed L1 book,
about 100 MB a session) are deleted after each session unless
`--keep-intermediates` is passed.

## Provenance corrections found along the way

Both were found by building rather than by inspection, which is the argument for
having builders at all.

1. **`ohlcv-1d` stamps each bar at UTC midnight of the session date.** Converting
   that index to exchange time moves every label back to 20:00 the previous
   evening and shifts the whole daily series one trading day. It gave INTC
   2024-08-02 the previous session's close and a wrong trailing volatility.
   Fixed by reading the label off the UTC index. Related: because these bars
   span the whole UTC day, their close is the last Nasdaq print by 20:00
   exchange time and includes the after-hours session, which is why INTC's
   post-earnings collapse on the evening of 2024-08-01 lands in the 2024-08-01
   bar.
2. **`data/MSFT_2024-06-03_metaorders.csv` had no builder behind it** until
   2026-09 and could not be reproduced under any lookup convention tried. It was
   rebuilt by `scripts/build_metaorders.py`, which moved the fitted exponent from
   0.788 to 0.370. Two defects in the original: fills sharing a timestamp were
   reordered, and `mid_start` was read after the run's first fill rather than
   before it. The second is worth 0.25 of exponent on its own. Across the
   fifteen sessions that 0.370 now sits in a range of 0.209 to 0.487.

## Licence and retention

Databento data is accessed under a programme licence; only small derived
aggregates are published. The work-trial data is proprietary to its provider and
is neither published nor relied upon for any stated result.

---

# The cross-sectional sample (TIM-XS)

A separate pull, a separate scope, and the only cross-sectional statement
anywhere in this repository. **S&P 500 members, Nasdaq venue only, April to
September 2024.** Not market-wide, not a regime.

## Universe and stratification

Membership is point in time from the alpha repository's local
`sp500_membership_daily.parquet` on **2024-06-28**: 503 active names, no WRDS
query. Two of them, BF and BRK, do not resolve as raw symbols on `XNAS.ITCH`
(the dotted share classes), and names printing on fewer than 15 Nasdaq sessions
in June are dropped, leaving **501**.

Stratification variables come from `ohlcv-1d` for June 2024, priced at
**$0.0000** for all 503 symbols before the pull:

* **relative tick size** = one cent over the June mean close, spanning
  1.32e-06 to 1.00e-03 across the drawn names;
* **dollar volume** = June mean close times June mean Nasdaq volume, spanning
  $5.6M to $6,559M a day.

Both are cut into terciles BY RANK, so the nine crossed cells are equal in
count rather than dominated by whichever variable is more skewed. Twelve names
are drawn at random from each cell with **seed 20240628**, written down here
before any trade was requested. No cell was short.

| cell (tick tercile x dollar-volume tercile) | available | drawn | names |
|---|---:|---:|---|
| high tick x high volume | 44 | 12 | BKR, C, CMCSA, CSX, EBAY, ETSY, FCX, FITB, HBAN, KHC, UAL, UBER |
| high tick x low volume | 72 | 12 | AMCR, BALL, CFG, CMS, DOC, FE, MAS, PPL, RF, SRE, SYF, SYY |
| high tick x mid volume | 51 | 12 | BSX, DVN, EQT, HAL, HAS, HOLX, HSIC, NDAQ, NEM, REG, SO, VTRS |
| low tick x high volume | 78 | 12 | AAPL, ANET, BKNG, FDX, FSLR, GS, LIN, MPWR, NOW, PANW, POOL, SNPS |
| low tick x low volume | 33 | 12 | AMP, AVB, AVY, BIO, EFX, EPAM, IT, NVR, RL, RSG, UHS, VMC |
| low tick x mid volume | 56 | 12 | AJG, BDX, BLK, CI, COR, HUBB, ITW, MKTX, MMC, MTD, STZ, VRSN |
| mid tick x high volume | 45 | 12 | ABBV, AEP, EA, ENPH, GOOGL, PAYX, ROST, SBUX, TER, TTWO, UPS, VST |
| mid tick x low volume | 62 | 12 | ALLE, ATO, CBRE, DLR, DVA, EXR, GPC, MTB, OKE, OTIS, PAYC, WAB |
| mid tick x mid volume | 60 | 12 | APH, CINF, DHI, EL, HWM, JBHT, MS, NUE, TROW, WYNN, YUM, ZBH |

**Two comparison names, MSFT and INTC, are appended and flagged
`role=comparison` in `sample.csv`.** They were not drawn by the sampler, they
exist so the cross-section can be put beside the three-name study, and they are
excluded from every cross-sectional statistic. AAPL was drawn, in the low tick
by high volume cell.

## The trades pull

| | |
|---|---|
| schema | `trades` |
| symbols | 110 (108 stratified, 2 comparison) |
| window | 2024-04-01 to 2024-10-01 |
| `metadata.get_billable_size` | 8,219,204,592 bytes (8.22 GB uncompressed) |
| `metadata.get_cost` | **$0.0000** |

`fetch_cross_section_trades.py` aborts on any non-zero price. The trades schema
carries price, size and aggressor side and nothing else, which is exactly what
the square-root law needs and a fraction of the size of MBO or MBP-10.

**Layout deviation, deliberate.** The rest of this repository stores raw
extracts as `<SYMBOL>/<YYYY-MM-DD>.<schema>.dbn.zst`, one file per symbol-day.
For 110 symbols over 126 trading days that would be nearly 14,000 files and
14,000 requests, so trades are stored one file per symbol over the whole range,
`<SYMBOL>/2024-04-01_2024-10-01.trades.dbn.zst`.

## Two things measured about the trades schema, not assumed

Both are in `reports/cross_section/trades_validation.csv`, checked against the
five AAPL sessions whose MBO-derived bars are committed.

1. **The aggressor side convention is the OPPOSITE of MBO's.** On MBO an
   execution's `side` is the side of the RESTING order, which is why
   `databento_to_lobster.py` negates it. On the `trades` schema `side` is the
   AGGRESSING side already. Built the MBO way, signed volume correlates
   **-0.9999** with the MBO-derived series; built the trades way, **+0.9998 to
   +0.9999**. A flipped sign would have inverted every metaorder in the study
   and still produced a plausible exponent.
2. **No deduplication is needed.** Sequence numbers are unique WITHIN a session;
   a six-month file contains collisions across days because sequences reset
   daily, and those are not duplicates. The T/F double count that
   `build_volume_tally.py` handles on MBO does not arise here, because the
   trades schema carries the print once.

Between 17% and 22% of AAPL prints carry no side at all, about 32% of RTH
volume. Those are the hidden prints. They count toward `volume` and toward V_D,
because they are traded volume, and not toward `signed_vol`, because their
direction is unknown. The direction-dominance filter therefore has them in its
denominator, which is the conservative direction.

## Normalisers

Both from the Nasdaq data itself, which is the same single-venue feed
arXiv 2606.24019 used, so the comparison with its prefactor is like for like.

* **V_D**, the day's total RTH Nasdaq volume, sided and unsided prints together.
* **sigma_D**, realised volatility from FIVE-MINUTE trade prices scaled to a
  session. Five minutes rather than one second because a one-second trade-price
  series is dominated by bid-ask bounce, and bounce scales with tick size, which
  is the regressor under test. A bounce-contaminated sigma would plant the
  result being looked for.

Venue volume is roughly a third of consolidated volume for these names, so every
participation rate here is a share of NASDAQ volume and the prefactor's level
depends on that choice. The exponent does not.

**Pending: the consolidated normaliser.** CRSP daily volume and trailing 20-day
close-to-close volatility would be a second normaliser.
`cross_section.consolidated_normalisers` is a stub that raises rather than
silently falling back, because a fallback would let a consolidated claim be made
from single-venue data. WRDS was refusing logins from this machine when this
branch was built and nothing in the cross-section depends on it.

## What is committed

Per-stock fitted parameters, the sample list with its seed and cell
assignments, and the aggregated (binned) metaorder tables, all under
`reports/cross_section/` and `data/cross_section/`. **Never the trades**, and
never the raw per-metaorder file: `data/cross_section/metaorders/` is
gitignored.
