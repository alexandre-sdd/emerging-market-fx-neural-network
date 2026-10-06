# Data

No datasets are included. Put original BIS exports in `raw/` and future derived
panels in `processed/`. Both folders' contents are ignored by Git, except their
empty `.gitkeep` files. Keep source exports unchanged.

## Planned sources

The coverage below comes from the project brief and has not yet been verified
against downloaded files. The intended monthly sample is January 1994–August 2026.

| BIS dataset | Intended use | Planned coverage |
| --- | --- | --- |
| WS_XRU | End-of-month exchange rates against USD; forecast targets | 39 currencies, monthly |
| WS_EER | Effective exchange rates; real exchange-rate gap and dollar index | 63 economies, monthly |
| WS_CBPOL | Policy-rate differentials with the US | 34 of the 39 currencies |
| WS_LONG_CPI | Inflation differentials with the US | 62 economies, monthly |
| WS_GLI | USD credit to non-banks outside the US and in 12 emerging economies | Quarterly, from 2000 |

When collecting data, record the source, download date, selected series
identifiers, frequency, units, quotation convention, and available date range
here. File formats and loaders will be chosen after inspecting the actual exports.

Before building the panel, document the currency list and classification,
peg exclusions, missing-data treatment, and quarterly-to-monthly alignment.
Dollar credit should enter with a two-quarter lag. Check publication timing for
all predictors and document any limitations from using revised historical data.
