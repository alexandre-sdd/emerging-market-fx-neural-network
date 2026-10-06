# Emerging-market dollar exchange rates in one global neural network

A research project on forecasting currency movements against the US dollar at
1-, 3-, and 12-month horizons. The planned panel includes 39 emerging-market
and advanced-economy currencies, with particular attention to emerging markets.

This repository is an initial scaffold. Data collection, panel construction,
features, models, and evaluation are not implemented yet.

## Planned study

Use BIS data for January 1994–August 2026 to compare three neural-network designs
on identical inputs: a shared network, a shared network with currency embeddings,
and a separate network for each currency.

Planned inputs are past exchange-rate changes, a real exchange-rate gap,
interest-rate and inflation differentials with the US, the dollar index, and
dollar credit lagged two quarters. Exclude dollar pegs and the peg periods of
the renminbi and ringgit.

Benchmarks are the random walk, AR(1), a Taylor-rule model, and pooled elastic
net. Evaluate with an expanding window from January 2010, reporting error
relative to the random walk, the Clark–West test, and direction accuracy.
Compare emerging and advanced currencies separately, and compare models with
and without dollar credit on a common evaluation sample.

The final deliverable is a 15–20 page report. A random walk that wins at every
horizon is a valid result.

## Repository

```text
README.md                 Project scope and setup
requirements.txt          Initial exploration dependencies
data/README.md           Planned datasets and data-handling notes
data/raw/                Local source files (ignored by Git)
data/processed/          Local derived data (ignored by Git)
notebooks/01_eda.ipynb     Exploration outline, with no analysis code
```

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

On Windows, create the environment with `py -3 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell.

Dependencies are limited to initial exploration. Choose the neural-network
framework and add modeling dependencies when implementation begins.

## First steps

1. Obtain the BIS exports and record their download dates and series identifiers.
2. Inspect their actual formats, coverage, units, and missing values.
3. Agree on the currency list, emerging/advanced classification, and peg exclusions.
4. Define the exchange-rate quotation, target changes, and feature construction.
5. Define time-based training and evaluation rules before fitting models.

Use only information available at each forecast date, including publication
lags. Training targets must have been observed by that date; the final forecast
origins depend on the horizon and available outcomes. Decide how to handle
missing policy rates and the shorter dollar-credit history before comparing models.

## Starting literature

References supplied in the project brief; confirm bibliographic details before
writing the report:

- Filippou, Rapach, Taylor and Zhou (2021), “Exchange rate prediction with machine
  learning and a smart carry trade portfolio”, *Journal of Financial and
  Quantitative Analysis*.
- Cheung, Chinn, Garcia Pascual and Zhang (2019), *Journal of International Money
  and Finance*, 95.
- Clark and West (2007), *Journal of Econometrics*, 138.
