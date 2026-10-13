# Emerging-Market Dollar Exchange Rates in One Global Neural Network

## Project overview

This project studies whether a global neural network can predict exchange-rate changes for a panel of 39 currencies against the U.S. dollar over 1-, 3-, and 12-month horizons. The analysis uses monthly BIS data from January 1994 through August 2026 and compares several neural-network architectures against familiar macro-finance baselines.

The project tests three model designs on the same input set:
- a single shared network for all currencies;
- a shared network with learned currency embeddings;
- a separate network for each currency.

The main empirical question is whether any of these models beats the random walk benchmark, especially for emerging-market currencies and at longer horizons, and whether the stock of dollar credit outside the United States adds predictive power.

Data acquisition, ingestion, and the initial exploration notebook are implemented. The
basic EDA reviews missing values, duplicates, coverage, ranges, and time paths.
Production feature engineering, models, and evaluation are not implemented.
An additional forecast-oriented EDA report builds explicitly exploratory targets,
lagged predictors, correlations, and simple benchmark diagnostics.

The primary 39-currency roster and data-treatment rules are now selected in
[the sample decision record](docs/data/sample_and_treatment.md) and
[the machine-readable specification](config/research_sample.json). The initial
review covers the wider downloaded BIS universe; the forecasting EDA retains its
42-currency exploratory comparison. The chosen design has not yet been applied
to a production panel. Current exports contain 64 broad EER economies and 63
CPI economies; the source counts below are the original project brief.

## Data sources

The project uses BIS monthly or quarterly data series:
- WS_XRU: US dollar exchange rates for 39 currencies (monthly, end-of-month)
- WS_EER: effective exchange rates for 63 economies
- WS_CBPOL: central bank policy rates for 34 of the 39 currencies
- WS_LONG_CPI: consumer price indices for 62 economies
- WS_GLI: global liquidity indicators, including non-US dollar credit to non-banks, quarterly from 2000

## Selected sample and treatment

The primary design uses 39 distinct currencies, with 16 advanced and 23 IMF
emerging market and developing economies for reporting. AED, HKD and SAR are
excluded as persistent USD pegs. The documented schedule excludes ARS, CNY and
MYR peg/transition periods and masks pre-launch EUR, BRL and BAM history. The
roster and series IDs are fixed without selecting on forecast performance.

Keep the monthly panel unbalanced. Raw data remain unchanged. The selected
future processing rules allow a flagged one-month causal policy-rate or derived
inflation carry, leave long/structural gaps explicit, and use training-window-only
median imputation with missing indicators for optional macro features. Mandatory
FX/EER inputs and forecast targets are not imputed. Conservative reference-period
lags define a revised-history pseudo-out-of-sample experiment, not a real-time
vintage study.

See [the decision record](docs/data/sample_and_treatment.md) for the full roster,
peg dates and sources, US CPI/Japan treatment, required sensitivity checks and
implementation boundaries. Production feature and benchmark steps are deferred.

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
│   ├── download_data.py
│   ├── process_data.py
│   └── run_eda.py
├── data/
│   ├── raw/
│   ├── ingested/
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
        │   ├── ingest.py
        │   ├── eda.py
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
BIS exports → ingestion → features → models → evaluation → report
```

- `config.py` defines paths relative to the checkout, independent of the working directory.
- `data/` owns acquisition, parsing, alignment, and currency exclusions.
- `features/` owns targets and predictors, with one common definition for all models.
- `models/` owns benchmark and neural-network fitting and prediction.
- `evaluation/` owns time splits, training-window preprocessing, forecast comparisons, and plots.
- `scripts/` contains thin command entry points; notebooks call package modules for exploration.

The download helper lives in `data/download.py`; `data/loaders.py` parses the
native exports, `data/ingest.py` creates typed observation and series tables,
and `data/eda.py` generates the separate initial review. Ingestion does not call
EDA or feature modules. Other research modules contain responsibility docstrings
only. Imports do not download files or run experiments.

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

The script checks for existing nonempty files under `data/raw/` and downloads
missing datasets from the official BIS bulk exports. It retains source ZIPs and
unchanged CSVs, and records URLs, UTC timestamps, SHA-256 hashes, and HTTP metadata
in `data/raw/download_manifest.json`. Failed downloads do not leave partial CSVs;
the script exits with a nonzero status if a dataset remains unavailable.

### 5) Ingest the downloaded dataset

```bash
python scripts/process_data.py
```

This is a separate local-only step. It selects the documented BIS series and
normalizes the wide exports into `data/ingested/observations.parquet` and
`data/ingested/series.parquet`, with a provenance and validation manifest.
Observations have one row per dataset, series, and native period. Missing values
remain null, quarterly credit stays quarterly, and values retain their BIS units.
The monthly calendar defaults to January 1994–August 2026; credit starts in 2000
and includes only completed quarters within that window. No predictors, forecast
targets, imputations, merged currency panel, or statistical tests are created.

Use `python scripts/download_data.py --force` to refresh the source snapshot.
Both scripts accept `--raw-dir`; processing also accepts `--output-dir`,
`--start YYYY-MM`, and `--end YYYY-MM`. See [data documentation](data/README.md)
for the ingestion schema and selections.

### 6) Optional: run the initial EDA separately

```bash
python scripts/run_eda.py
```

Open `notebooks/01_eda.ipynb` for the executed tables and charts, or read
[the initial findings](reports/eda/README.md). The script regenerates that report,
quality tables in `reports/eda/`, figures in `reports/figures/`, and selected
descriptive matrices with metadata in `data/processed/`. Source files are not
modified. The analysis uses January 1994–August 2026 for monthly data and
2000Q1–2026Q2 for expected completed quarterly credit observations. The downloaded
credit snapshot ends at 2026Q1, so Q2 is recorded as missing.

## Interactive forecast-oriented EDA

```bash
python scripts/run_interactive_eda.py
```

Three HTML reports live side by side:

- [Data quality](reports/eda/data_quality.html): the earlier review, restyled from
  its saved evidence, with interactive charts for
  coverage, gaps, duplicate/parsing checks, currency mappings and source ranges.
  A visually distinct closing discussion compares the executive brief with the
  observed export and selected roster, identifies the first design questions,
  and separates completed analysis from documented choices and pending work.
- [Data EDA](reports/eda/data_eda.html): the forecast-oriented Plotly analysis,
  with short quality notes and direct links to the relevant quality sections.
- [Research choices](reports/eda/research_choices.html): the selected sample and
  treatment, why each choice was made, alternatives and unresolved distinctions.
  Expandable explanations and interactive FX-only eligibility counts show the
  costs of regime restrictions, horizon lengths and credit availability. This is
  a report audit, not a production feature panel or model evaluation.

All three pages share one visual theme and Plotly runtime, lead with the most relevant
findings, and link evidence to forecasting decisions. Quality tables are copied
to `reports/eda/quality/`; EDA tables live in `reports/eda/interactive/`;
decision-audit tables and specification copies live in `reports/eda/decision_assets/`.
Regenerate only the third report with `python scripts/run_decision_report.py`.

The EDA report embeds Plotly and works offline, with currency, horizon, period and correlation
selectors. Accompanying CSVs expose coverage, sample mapping, tail statistics,
per-currency predictive correlations, stress-conditioned relationships, feature
availability, and random-walk / expanding historical-mean forecast metrics.

The report follows the questions raised by the July 2, 2021 working-paper version
of Filippou, Rapach, Taylor and Zhou. It covers 42 distinct non-USD currencies with
broad REER data, uses the euro-area EUR representative from 1999, and omits
national EUR backcasts. This exploratory universe is not the final 39-currency
roster. Pegs are flagged but retained, and no emerging/advanced grouping is assumed.
Feature definitions and conservative timing assumptions are documented in the
report; revised historical exports do not establish real-time availability.

The generator is separate from the initial EDA and the production feature/model
modules. It does not train neural networks or claim statistical significance.
Use `--output-dir PATH` to write all three HTML reports elsewhere, with quality tables
in `quality/`, EDA tables and provenance in `interactive/`, and decision evidence
in `decision_assets/`. Regenerating the
pages reads the saved earlier quality evidence; it does not rerun that analysis.
Generated results remain outside version control under `reports/eda/` by default.

All report outputs are grouped under `reports/`: EDA reports and tables in
`eda/`, exported charts in `figures/`, and weekly PDF reports in `progress/`.
The [8 October progress report](reports/progress/weekly_progress_report_2026-10-08.pdf)
is stored alongside the other report outputs. Generated progress PDFs are also
excluded from version control.

## Data download script

The repository includes a download helper that:
- checks whether data files already exist;
- downloads missing files into `data/raw/`;
- normalizes filenames and creates the required local folder structure;
- keeps raw, ingested, and processed datasets out of Git via `.gitignore`, while retaining data documentation.

## Notes on the BIS series

Default source archives are listed on [BIS bulk downloads](https://data.bis.org/bulkdownload).
Override a source with `BIS_WS_XRU_URL`, `BIS_WS_EER_URL`, `BIS_WS_CBPOL_URL`,
`BIS_WS_LONG_CPI_URL`, or `BIS_WS_GLI_URL` if needed. Ingestion and EDA expect the
official wide CSV layout with `FREQ` and `Series` metadata. Flat SDMX API exports
use a different schema and would need a separate parser.

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
