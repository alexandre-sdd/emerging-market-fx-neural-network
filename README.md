# Emerging-Market Dollar Exchange Rates in One Global Neural Network

## Project overview

This project studies whether a global neural network can predict exchange-rate changes for a panel of 39 currencies against the U.S. dollar over 1-, 3-, and 12-month horizons. The analysis uses monthly BIS data from January 1994 through August 2026 and compares several neural-network architectures against familiar macro-finance baselines.

The project tests three model designs on the same input set:
- a single shared network for all currencies;
- a shared network with learned currency embeddings;
- a separate network for each currency.

The main empirical question is whether any of these models beats the random walk benchmark, especially for emerging-market currencies and at longer horizons, and whether the stock of dollar credit outside the United States adds predictive power.

This repository is a starting scaffold. The exploration notebook is an outline;
feature engineering, models, and evaluation are not implemented.

## Data sources

The project uses BIS monthly or quarterly data series:
- WS_XRU: US dollar exchange rates for 39 currencies (monthly, end-of-month)
- WS_EER: effective exchange rates for 63 economies
- WS_CBPOL: central bank policy rates for 34 of the 39 currencies
- WS_LONG_CPI: consumer price indices for 62 economies
- WS_GLI: global liquidity indicators, including non-US dollar credit to non-banks, quarterly from 2000

## Research design

The planned project workflow is:
1. Build a monthly panel of 39 currencies.
2. Exclude dollar pegs and renminbi/ringgit peg periods.
3. Construct 1-, 3-, and 12-month target changes.
4. Create features based on:
   - lagged changes in exchange rates;
   - real exchange-rate gaps;
   - interest-rate and inflation differentials relative to the U.S.;
   - the dollar index;
   - dollar credit outside the United States lagged two quarters.
5. Estimate benchmark models:
   - random walk;
   - AR(1);
   - Taylor-rule style model;
   - pooled elastic net.
6. Fit the three neural-network architectures on identical inputs.
7. Evaluate the models using an expanding-window design starting in January 2010.
8. Compare performance relative to the random walk, using Clark-West tests and directional accuracy.
9. Report results separately for emerging and advanced economies, with and without dollar-credit variables.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── scripts/
│   └── download_data.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    └── fx_forecasting/
        ├── __init__.py
        ├── config.py
        ├── data/
        │   ├── __init__.py
        │   ├── download.py
        │   ├── loaders.py
        │   └── panel.py
        ├── features/
        │   ├── __init__.py
        │   ├── targets.py
        │   └── predictors.py
        ├── models/
        │   ├── __init__.py
        │   ├── benchmarks.py
        │   └── networks.py
        └── evaluation/
            ├── __init__.py
            ├── backtest.py
            ├── metrics.py
            └── plots.py
```

## Module responsibilities

The package follows the planned research flow:

```text
BIS exports → data → features → models → evaluation → report
```

- `config.py` defines paths relative to the checkout, independent of the working directory.
- `data/` owns acquisition, parsing, alignment, and currency exclusions.
- `features/` owns targets and predictors, with one common definition for all models.
- `models/` owns benchmark and neural-network fitting and prediction.
- `evaluation/` owns time splits, training-window preprocessing, forecast comparisons, and plots.
- `scripts/` contains thin command entry points; notebooks call package modules for exploration.

The existing download helper lives in `data/download.py`. Other research modules
contain responsibility docstrings only; their functions and model interfaces will
be defined during implementation. Imports do not download files or run experiments.

Keep data and feature modules independent of model code. Fit scaling, imputation,
and other learned transformations within each training window. Share forecast
origins, observed targets, and inputs across model comparisons. Document currency
groups and compare credit specifications on common samples.

One small package keeps the team workflow simple. Separate network architectures
or experiment configuration into additional modules when their implementations
become large enough to warrant it.

## Setup

### 1) Clone the repository

```bash
git clone https://github.com/alexandre-sdd/emerging-market-fx-neural-network.git
cd emerging-market-fx-neural-network
```

### 2) Create a virtual environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
python -m pip install -e .
```

The editable installation makes `fx_forecasting` importable in scripts and
notebooks. Dependencies remain in `requirements.txt`, which supplies the package
metadata. Select the virtual environment when running notebooks.

### 4) Download the dataset if it is missing

```bash
python scripts/download_data.py
```

The script checks whether the required raw datasets are already present under `data/raw/`. If they are not, it attempts to download them from the configured BIS sources. If the network or registry is unavailable, it exits gracefully and tells you exactly which files are missing.

## Data download script

The repository includes a download helper that:
- checks whether data files already exist;
- downloads missing files into `data/raw/`;
- normalizes filenames and creates the required local folder structure;
- keeps raw and processed datasets out of Git via `.gitignore`, while retaining data documentation.

## Notes on the BIS series

These BIS series are not always made available through a single static archive. Depending on the exact download source you use, the script may need the specific BIS file names or a temporary source list. The included helper is designed to be easy to adapt to your course environment or institution's data source.

## References

Starting references from the project brief; verify bibliographic details before writing the report.

- Filippou, Rapach, Taylor and Zhou (2021), "Exchange rate prediction with machine learning and a smart carry trade portfolio", Journal of Financial and Quantitative Analysis.
- Cheung, Chinn, Garcia Pascual and Zhang (2019), Journal of International Money and Finance, 95.
- Clark and West (2007), "Approximately normal tests for equal predictive accuracy in nested models", Journal of Econometrics.

## Team instructions

- Work in feature branches when making large changes.
- Keep scripts reproducible and documented.
- Store results and raw datasets outside version control.
- Keep notebooks focused and well commented.
- Raise issues for missing dataset files or model reproduction steps.
