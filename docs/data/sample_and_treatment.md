# ADR-001: Currency sample and data treatment

**Status:** Selected primary research design; implementation deferred.
**Date:** 13 October 2026.
**Decision authority:** The user delegated sample and treatment choices for this project.

The primary study universe is **39 distinct non-USD currencies: 16 advanced and
23 emerging market and developing economies (EMDE)**. Keep the panel unbalanced,
preserve the original BIS values, and apply the eligibility, availability and
missing-data rules below when production processing is implemented.

[The machine-readable specification](../../config/research_sample.json) is the
canonical roster, series mapping, interval schedule and treatment specification.
This is a project-selected roster, not a recovery of an undisclosed list in the
original brief. The earlier EDA's 42-currency comparison remains descriptive and
has not been recomputed on this design. No production panel, feature script,
benchmark pipeline or additional HTML report has been built in this step.

## Context and selection

The 8 October BIS snapshot contains 192 FX reference areas, rather than 192
distinct currencies. The prior EDA identifies 42 non-USD currency representatives
with broad REER coverage after using `XM` for EUR and omitting national EUR
backcasts. Select these representatives and remove `AE/AED`, `HK/HKD` and `SA/SAR`
because their USD pegs make them poor subjects for the intended non-peg USD
forecasting comparison. This produces 39 without ranking countries by forecast
correlations, returns or later performance.

The exclusions are supported by [CBUAE's intervention framework](https://centralbank.ae/en/our-operations/monetary-policy-and-domestic-markets/domestic-market-operations/)
and [historical US government description of AED's dollar relationship](https://1997-2001.state.gov/issues/economic/trade_reports/neareast96/uae96.html),
[HKMA's linked exchange-rate description](https://www.hkma.gov.hk/eng/key-functions/money/linked-exchange-rate-system/),
and [SAMA's fixed-parity commitment](https://sama.gov.sa/en-US/MediaCenter/News/Pages/news-557.aspx).

Freeze reporting groups to the [April 2026 IMF WEO, Table B](https://www.imf.org/-/media/files/publications/weo/2026/april/english/statsappendix.pdf),
available before the source download. EMDE includes developing economies; it is
not equivalent to membership in an investable emerging-market index. Groups are
static reporting labels and must not be used as historical classification
evidence or predictive features. EUR represents the euro area.

## Primary roster

| Area | Currency | Economy | Reporting group |
| --- | --- | --- | --- |
| DZ | DZD | Algeria | EMDE |
| AR | ARS | Argentina | EMDE |
| AU | AUD | Australia | Advanced |
| BA | BAM | Bosnia and Herzegovina | EMDE |
| BR | BRL | Brazil | EMDE |
| CA | CAD | Canada | Advanced |
| CL | CLP | Chile | EMDE |
| CN | CNY | China | EMDE |
| TW | TWD | Chinese Taipei | Advanced |
| CO | COP | Colombia | EMDE |
| CZ | CZK | Czechia | Advanced |
| DK | DKK | Denmark | Advanced |
| XM | EUR | Euro area | Advanced |
| HU | HUF | Hungary | EMDE |
| IS | ISK | Iceland | Advanced |
| IN | INR | India | EMDE |
| ID | IDR | Indonesia | EMDE |
| IL | ILS | Israel | Advanced |
| JP | JPY | Japan | Advanced |
| KR | KRW | Korea | Advanced |
| MY | MYR | Malaysia | EMDE |
| MX | MXN | Mexico | EMDE |
| MA | MAD | Morocco | EMDE |
| NZ | NZD | New Zealand | Advanced |
| MK | MKD | North Macedonia | EMDE |
| NO | NOK | Norway | Advanced |
| PE | PEN | Peru | EMDE |
| PH | PHP | Philippines | EMDE |
| PL | PLN | Poland | EMDE |
| RO | RON | Romania | EMDE |
| RU | RUB | Russia | EMDE |
| RS | RSD | Serbia | EMDE |
| SG | SGD | Singapore | Advanced |
| ZA | ZAR | South Africa | EMDE |
| SE | SEK | Sweden | Advanced |
| CH | CHF | Switzerland | Advanced |
| TH | THB | Thailand | EMDE |
| TR | TRY | Türkiye | EMDE |
| GB | GBP | United Kingdom | Advanced |

## Currency histories and peg intervals

Use January 1994–August 2026 as the outer monthly calendar. A roster of 39 does
not mean 39 usable observations at every month, horizon or specification.

| Rule | Primary choice | Evidence and rationale |
| --- | --- | --- |
| EUR representation | `XM` only, from January 1999 | [ECB](https://www.ecb.europa.eu/press/key/date/1999/html/sp990521.sl.html): euro monetary union began 1 January 1999. Omit earlier synthetic EUR values and national EUR histories. |
| BRL history | From July 1994 | [Banco Central do Brasil](https://www.bcb.gov.br/controleinflacao/planoreal?modalAberto=30anosreal_noticia4): the real entered circulation on 1 July 1994. Do not describe predecessor observations as traded BRL history. |
| BAM history | From July 1998 | [CBBH](https://www.cbbh.ba/press/ShowNews/1521?lang=en): first banknotes entered circulation on 22 June 1998. July is the first full post-launch month; this is a conservative sample choice. |
| ARS exclusion | January 1994–January 2002 inclusive | [IMF IEO](https://www.imf.org/External/NP/ieo/2004/arg/eng/): convertibility ended in January 2002. Include that transition month in the exclusion. |
| CNY exclusion 1 | January 1994–July 2006 inclusive | Exclude pre-reform history and use the de facto cutoff: [IMF WP/19/50, footnote 6](https://www.imf.org/-/media/files/publications/wp/2019/wpiea2019050.pdf) retains the conventional fixed-peg classification until 1 August 2006 despite formal reform in July 2005. |
| CNY exclusion 2 | June 2008–June 2010 inclusive | The same IMF paper dates the stabilized arrangement from 1 June 2008 to 21 June 2010. Exclude the entire transition month. |
| MYR exclusion | September 1998–July 2005 inclusive | [BNM milestones](https://www.bnm.gov.my/web/guest/significant-milestones-in-the-malaysian-foreign-exchange-market): peg introduced September 1998 and removed 21 July 2005. |

The CNY primary rule uses de facto stability rather than only the announcement
date. A required sensitivity uses the formal-policy end date, July 2005, for the
first exclusion. Crawling/managed arrangements outside these specified intervals
remain eligible; this is not an exhaustive classification of every exchange-rate
regime. Retain EUR-linked currencies because their dollar rates can move—see
[Danmarks Nationalbank](https://www.nationalbanken.dk/en/frequently-asked-questions/questions-regarding-fixed-exchange-rate-policy).
Run a sensitivity omitting EUR-linked currencies to assess common EUR exposure.

For an h-month target, require positive, observed FX at the origin and every
calendar month through its endpoint, with no excluded month in that interval.
Reset FX lag and rolling-return histories across currency starts, peg exclusions
and missing FX; do not calculate returns across those gaps. Retain original
values in the source layer and record exclusion reasons separately.

This is a retrospective non-peg sample definition. Future regime transitions
were not necessarily known at the original forecast origin. Therefore later
evaluation is conditional on eligible historical intervals and cannot establish
an executable real-time forecast selection rule.

## Missing observations: preserve data, flag assumptions

The selected roster has **35 policy sources and 37 CPI sources**. Policy is absent
for BAM, DZD, SGD and TWD; CPI is absent for BAM and TWD. REER and NEER exist for all
39. These are source matches, not guarantees of uninterrupted histories.

| Input | Selected treatment |
| --- | --- |
| FX and targets | Never fill, interpolate or clip. Missing/nonpositive FX invalidate dependent targets and return histories. |
| Policy rates | Allow a causal carry for at most one missing reference month, flagged separately. Preserve zero/negative rates. Never bridge a long run: only its first missing month can receive that one-month carry. |
| Japan policy gaps | Keep 116 source missing months explicit, including the 17-, 59- and 40-month internal runs. The general one-month rule permits a flagged first-gap-month carry, not years of assumed zero rates. |
| National euro rates | No stitching or post-adoption carry across national histories. Only the `XM` series supplies the euro-area policy rate from 1999. |
| Inflation | Calculate observed 12-month CPI log inflation first. Allow its previous available reading to carry for one month, with a flag. Do not fabricate CPI index levels or use later months to interpolate. |
| US October 2025 CPI | Use September 2025 annual log inflation as a flagged one-month October fallback. Under the two-month lag, it supplies the December 2025 feature. It is an assumption, not an observed October inflation value. |
| Long/structural macro gaps | Leave values null in the persisted panel. Retain source-absent flags and imputation provenance. Do not fetch or splice incomparable substitute measures silently. |
| REER, NEER and credit | No filling. Require their correctly lagged observations for specifications that need them. |

[BLS documents the October CPI collection gap](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm).
The selected fallback uses information already available, rather than a two-sided
reconstruction using November. A required sensitivity excludes all carried rows,
including the affected December 2025 origins.

At model fitting time, optional policy/inflation differences may use a **pooled
training-window median plus explicit missing indicators**. Keep those replacements
out of the source/persisted economic measures. This retains all 39 currencies
without disguising absent national macro series as observed values. Every model
comparison must use identical origins, flags and treatment. If a predictor is
entirely absent in a training window, drop it identically for the compared models
or fail explicitly; never derive a replacement from evaluation observations.

FX lag/rolling inputs, REER and USD NEER remain mandatory observed inputs. Credit
is additionally mandatory for credit-versus-no-credit comparisons, using a
matched sample. Also report a strict complete-case macro sensitivity and a
specification omitting optional macro variables. These checks will show how much
results depend on the imputation choice.

## Timing and vintage convention

Forecast origins are month ends. Adopt the following conservative reference-date
lags before constructing features; these do not claim to reproduce every
country's historical release calendar.

| Input | Selected reference date available at origin t |
| --- | --- |
| Bilateral FX | End of month t |
| Policy rates | t−1 month, after any allowed one-month carry |
| CPI-derived inflation | t−2 months, after any allowed one-month inflation carry |
| REER | t−2 months |
| Broad US NEER | t−1 month |
| Outside-US USD credit | Quarter containing t minus two quarters |

For July and August 2026 origins, credit uses 2026Q1. Hold that quarter's value
within the origin quarter, with no interpolation or additional carry. Preserve
its native millions-of-USD units in source tables. The missing 2026Q2 stock does
not enter this study window under this lookup rule.

Apply lags before constructing rolling macro gaps and interactions. Fit scaling,
imputation and any learned transformations inside each training window; training
targets must have realised by the forecast origin. Preserve reference periods,
assumed availability periods, source IDs and carry/missing flags in later outputs.

The BIS snapshot is revised history. The primary experiment is **pseudo out of
sample with reference-period availability assumptions**, not a real-time vintage
study. The April 2026 economy groups are used only for reporting. Obtaining
historical release calendars and vintages would be a separate extension.

## Extreme changes and source breaks

Retain observed large moves and use logs only for positive level measures.
Do not set an exclusion threshold using the full sample or remove crisis months
because they hurt benchmark performance. Exclude a unit/source break only when
independently documented, recording the evidence and interval. Use the same
break decisions for every model. Native policy-rate definition changes should
be treated as comparability limits, not corrected by arbitrary offsets.

## Options considered and consequences

| Option | Decision and trade-off |
| --- | --- |
| Keep all 42 EDA currencies | Rejected as the primary sample: includes three persistent USD pegs. The earlier EDA remains an audit reference. |
| Drop every country missing any macro history | Rejected as primary: removes economies for data availability and defeats the 39-currency question. Retain a complete-case sensitivity. |
| Fill long gaps or interpolate using later values | Rejected: invents economic history and can leak later information. |
| Never carry any short gap | Kept as sensitivity; primary permits a flagged one-month carry to avoid discarding a global forecast origin for a short reference gap. |
| Formal CNY reform dates only | Kept as sensitivity; primary uses the IMF de facto fixed/stabilized cutoff. |

The chosen design keeps currency coverage broad but requires explicit masks and
missing indicators. It sacrifices observations around launches and peg episodes,
and introduces warm-up periods when FX histories restart. The missingness and
regime rules must be applied before future model comparisons; the existing EDA
metrics do not represent the chosen panel's benchmark performance.

The next implementation step is to consume this specification in the production
panel/feature scripts and test boundary dates, carry limits, training-window
imputation and common forecast samples. That implementation is intentionally
deferred under the user's revised scope.
