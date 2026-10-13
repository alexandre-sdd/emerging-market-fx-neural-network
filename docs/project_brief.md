# Executive project brief

**Source:** Instructions supplied by the project executives, reproduced from the
user's message on 13 October 2026. Counts and bibliography below are the brief's
statements; they are not assertions about the current downloaded snapshot.

## Topic and objective

Emerging-market dollar exchange rates in one global neural network. Predict
39 currencies against the US dollar over 1-, 3- and 12-month horizons using
monthly BIS data for January 1994–August 2026. Compare one shared network, a
shared network with learned currency embeddings, and separate currency networks.
Determine whether they beat the random walk, especially for emerging currencies
and longer horizons, and whether the stock of dollar credit outside the US helps.

## Data specified by the executives

| Dataset | Brief's scope |
| --- | --- |
| WS_XRU | 39 currencies; end-of-month USD exchange rates; January 1994–August 2026 |
| WS_EER | 63 economies; monthly from 1994; includes the US dollar index |
| WS_CBPOL | Policy rates for 34 of the 39 currencies |
| WS_LONG_CPI | 62 economies; monthly |
| WS_GLI | USD credit to non-banks outside the US and in 12 emerging economies; quarterly from 2000 |

## Required work

1. Build a monthly panel of 39 currencies, excluding dollar pegs and renminbi/
   ringgit peg periods. Targets are changes over 1, 3 and 12 months.
2. Build past-change, real exchange-rate gap, interest-rate and inflation
   differential, dollar-index and dollar-credit inputs; lag credit two quarters.
3. Fit random walk, AR(1), Taylor-rule and pooled elastic-net benchmarks.
4. Fit shared, currency-embedding and separate networks on identical inputs.
5. Use expanding windows from January 2010. Report error relative to random walk,
   Clark–West tests and directional accuracy; split emerging/advanced currencies
   and compare with/without dollar credit.
6. Write a 15–20-page report. A random walk winning at every horizon is a valid
   finding.

## Starting literature and access as supplied

- Filippou, Rapach, Taylor and Zhou (2021), "Exchange rate prediction with machine
  learning and a smart carry trade portfolio," forthcoming in the Journal of
  Financial and Quantitative Analysis.
- Cheung, Chinn, Garcia Pascual and Zhang (2019), Journal of International Money
  and Finance 95.
- Clark and West (2007), Journal of Econometrics 138.

The brief provides the CSV pattern
`https://stats.bis.org/api/v1/data/DATAFLOW/all/all?format=csv`, replacing DATAFLOW
with the dataset code, and the [BIS data portal](https://data.bis.org).
The repository uses official BIS bulk exports and records their exact source
URLs and hashes. Access routes and frequency/series selections must be recorded
when comparing counts.

## What the brief does not specify

It does not enumerate the 39 currencies, define economy groups or exact peg
intervals, prescribe missing-data treatment or release timing, or specify the
functional form of the dollar-credit stock. Those are initial design choices.
The current project-selected choices are in
[ADR-001](data/sample_and_treatment.md); their correspondence with the executives'
original intended roster cannot be established from the counts alone.
