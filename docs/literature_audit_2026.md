# Primary-source literature audit, checked 2026-09-06

This is a targeted audit through the retrieval date, not a claim that every
2026 preprint has been reviewed. Empirical results below describe each paper's
own sample. They do not enlarge this repository's fifteen-session panel or
turn its 108-name Nasdaq cross-section into institutional parent orders.

| Primary source | Sample and horizon | Verified finding and consequence here |
|---|---|---|
| [Naviglio, Bormetti, Campigli, Rodikov and Lillo, 2025, arXiv 2501.17096](https://arxiv.org/html/2501.17096v1) | AMZN and MSFT, 22 trading days in June 2021; simulated metaorders lasting 1,000 trades in the linear analysis | Equations 8 to 10 distinguish price-level propagators from return coefficients. Fitting public-flow dependencies can yield unrealistic impact paths when an inserted order is allowed to trigger other flow. The cumulative response here is an algebraic conditional response, not an identified causal intervention. |
| [Gatheral, Schied and Slynko, Mathematical Finance 22, 2012](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9965.2011.00478.x) | Mathematical transient linear-impact model, general trading horizons; no empirical sample | Optimization and no-manipulation results require a specified price-level cost kernel. They cannot be applied directly to coefficients of a return regression. Positive definiteness of the wrong matrix does not validate an empirical execution model. |
| [Obizhaeva and Wang, JFM 16, 2013, author copy](https://web.mit.edu/wangj/www/pap/ObizhaevaWang13.pdf) | Theoretical resilient order book over a chosen liquidation horizon; no calibration sample claimed here | Resilience shapes optimal execution. Block trades combined with continuous trading follow from the specified liquidity dynamics, not from near-zero lagged return coefficients. |
| [Bucci, Benzaquen, Lillo and Bouchaud, PRL 122, 108302, 2019, author copy](https://www.off-ladhyx.polytechnique.fr/people/benzaquen/publications/Bucci2018crossover.pdf) | About 8 million institutional US-equity trades; impact over order execution | Finds a linear-to-square-root crossover. This corrects the old README's article number and link: arXiv 1901.05332 is the different paper *Slow decay of impact in equity markets*. Boundary estimates in this repository are not identified crossovers. |
| [Zarinelli, Treccani, Farmer and Lillo, 2015, arXiv 1412.2152](https://arxiv.org/abs/1412.2152) | Institutional US-equity metaorders; peak impact over execution, with participation and duration conditioning | Logarithmic dependence and participation effects challenge a universal single power fit. The repository's multiplicative log-rate term is a simplified local specification, not an exact reproduction of the paper's full surface. |
| [Maitrier, Loeper and Bouchaud, 2025, arXiv 2503.18199](https://arxiv.org/abs/2503.18199) | Public trade-data construction of synthetic metaorders; execution and post-execution horizons | Proposes a reconstruction algorithm reproducing concavity and decay. A maximal same-sign fill run is only this repository's simpler proxy; citation does not establish equivalence to their algorithm or reveal real parent identity. |
| [Vasaikar, 2026, arXiv 2606.24019](https://arxiv.org/abs/2606.24019) | AAPL Nasdaq MBO, 178 trading days, 2024-12-02 to 2025-08-19; proxy-metaorder execution impact | Reports raw prefactor 0.69 and robust reconstruction checks. This repository uses an earlier sample and its 108-name extension uses last trade prices rather than mids. The comparison is same venue, not an exact replication. The bias-corrected prefactor is not reproduced. |
| [Cont, Cucuringu and Zhang, QF 2023, arXiv 2112.13213v4](https://arxiv.org/html/2112.13213v4) | Top 100 S&P 500 constituents chosen at 2019 year-end, 2017 to 2019; short intraday windows and one-minute forecasts | Integrated multi-level OFI improves contemporaneous fit; lagged cross-asset OFI can add forecast power. This repository's three-name single-asset result cannot refute that cross-asset finding or claim its names were absent from the paper. |
| [Maitrier, Loeper, Kanazawa and Bouchaud, 2025, arXiv 2502.16246](https://arxiv.org/abs/2502.16246) | Tokyo Stock Exchange trader identifiers, 2012 to 2018; child orders through metaorder execution and relaxation | Finds square-root child impact with inverse-square-root time decay, including scrambled synthetic parent identities. Separating trade identity, aggregation and response horizon is essential before interpreting proxy exponents structurally. |
| [Noble, Rosenbaum and Souilmi, 2026, arXiv 2603.24137](https://arxiv.org/html/2603.24137v1) | Large-tick US stocks including PFE, INTC, VZ and T; Databento December 2023 to December 2025, 10:00 to 15:30 | Separates event timing, latency-race fills and persistent impact feedback. The latest implementation argument is to validate execution and response jointly. An observational fit alone does not establish simulated P&L fidelity. |

## Scope correction from primary calendars

The six-month cross-section is not free of macro announcements. The
[Federal Reserve's 2024 calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
includes decisions on May 1, June 12, July 31 and September 18, inside its date
window. The [BLS September 11, 2024 CPI release](https://www.bls.gov/news.release/archives/cpi_09112024.pdf)
also falls inside it. No event exclusion is implemented in the cross-section
builder, so the README now states that events were not separately stratified.

## Ranked next research

1. Correct the response estimand and stop publishing invalid schedule inference.
   This requires no data or retuning, and is the change implemented here.
2. Obtain committed per-session coefficient vectors from the already cached
   runs, preserving original selection status, then form simultaneous cumulative
   response bands. Do not add marginal confidence limits together.
3. Design an execution-cost experiment with a common level-price convention,
   a supported tail, nested model selection and an evaluation window that has
   not been used to choose among these repairs. That work is deferred, not
   described as successful because the corrected point curve looks plausible.
