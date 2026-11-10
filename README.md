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
Causal FX/macro features and joint sequence exports are implemented for the
clustering handoff below. Neural forecasting models have not been trained.
An additional forecast-oriented EDA report builds explicitly exploratory targets,
lagged predictors, correlations, and simple benchmark diagnostics.

The primary 39-currency roster and data-treatment rules are now selected in
[the sample decision record](docs/data/sample_and_treatment.md) and
[the machine-readable specification](config/research_sample.json). The initial
review covers the wider downloaded BIS universe; the forecasting EDA retains its
42-currency exploratory comparison. The joint handoff now applies the selected
design to a causal FX/macro panel. Current exports contain 64 broad EER economies and 63
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
implementation boundaries. The handoff applies these data rules; production
forecasting model and benchmark comparisons remain separate.

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
EDA or feature modules. The canonical panel and joint dataset builder now serve
the clustering handoff; forecasting model modules remain scaffolds. Imports do not download
files or run experiments.

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

Two HTML reports live side by side:

- [Data quality](reports/eda/data_quality.html): the earlier review, restyled from
  its saved evidence, with interactive charts for
  coverage, gaps, duplicate/parsing checks, currency mappings and source ranges.
- [Data EDA](reports/eda/data_eda.html): the forecast-oriented Plotly analysis,
  with short quality notes and direct links to the relevant quality sections.

Both pages share one visual theme and Plotly runtime, lead with the most relevant
findings, and link evidence to forecasting decisions. Quality tables are copied
to `reports/eda/quality/`; EDA tables live in `reports/eda/interactive/`.

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
Use `--output-dir PATH` to write both HTML reports elsewhere, with quality tables
in `quality/` and EDA tables and provenance in `interactive/`. Regenerating the
pages reads the saved earlier quality evidence; it does not rerun that analysis.
Generated results remain outside version control under `reports/eda/` by default.

All report outputs are grouped under `reports/`: EDA reports and tables in
`eda/`, exported charts in `figures/`, and weekly PDF reports in `progress/`.
The [8 October progress report](reports/progress/weekly_progress_report_2026-10-08.pdf)
is stored alongside the other report outputs. Generated progress PDFs are also
excluded from version control.

## Currency clustering for pooled training

Open [the currency-clustering notebook](notebooks/02_currency_clustering.ipynb)
to group currencies by constrained DTW similarity and assemble candidate training
batches. Install its optional dependency with `python -m pip install -e ".[clustering]"`.
It runs from either the repository root or `notebooks/` and reads the existing
ingested snapshot without downloading data or depending on another notebook's state.

The default fit window is January 2011–December 2019. Exact configured source IDs,
currency starts and peg exclusions apply before returns are constructed. Only
currencies with complete, nonconstant returns throughout that shared calendar
enter clustering; exclusions are reported. Standardized monthly log returns
are clustered using constrained DTW. Silhouette and inertia plots guide
an explicit choice of K explained in Markdown. Short sensitivity calculations
compare seeds, historical windows, Euclidean distance and retained return magnitude.
The notebook keeps the calculations in short cells and the checks and interpretation
in Markdown, without separate clustering helper or test modules.

Exports under `reports/clustering/` include membership and coverage tables,
model-selection diagnostics, run settings, and `pooled_training.parquet`.
The notebook also plots a two-dimensional MDS map of the DTW distances and saves
PNG/SVG maps, an offline interactive HTML map, and cluster return-history plots.
The pooled training table keeps separate currency-month observations, a common FX-only predictor
schema, and original 1-, 3- and 12-month targets realised by the fit cutoff.
It demonstrates how to batch data; it does not replace the production macro panel.
These group labels can support forecasts after December 2019 and must not be
backfilled into the planned January 2010 evaluation. Later forecast comparisons
must refit grouping within the corresponding historical information set.

For a concise presentation page, run `python scripts/run_cluster_summary.py`.
Open `reports/clustering/cluster_summary.html` for the K comparison, interactive
cluster map and combined return histories, with brief interpretations. The page
uses the notebook's saved results and works offline as a single HTML file.

[The correlation-clustering notebook](notebooks/03_currency_correlation_clustering.ipynb)
compares contemporaneous return correlation with normalized FX-level correlation
on the same dates and currencies. It uses average-linkage hierarchies, silhouette
and dendrograms, with the interpretation in Markdown. After executing it, run
`python scripts/run_correlation_summary.py` to create the standalone comparison
at `reports/clustering/correlation/correlation_comparison.html`.

## Joint forecasting: fixed clusters and teammate handoff

Start with [the main comparison notebook](notebooks/05_r2kmeans_methodology_audit.ipynb).
For ready-to-try memberships, training counts and loader settings, use the
generated [teammate quick start](reports/clustering/handoff/teammate_quickstart.md).
The TA-requested priority is chronological evolution: continuous R2K monthly
states, observed pair co-clustering in sliding 60-month windows, and an
independent NumPy audit of Algorithm 2. R2K's 12-month feature lookback,
centroid memory W=1/6/12 and five-year reporting window are distinct. Correlation
references refit trailing 60-month return histories. All K=2–6 remain available.
The interactive report now leads with this explorer; the earlier fixed-group
comparison is collapsed as a handoff reference. Post-2019 temporal results are
descriptive and do not establish neural forecast performance.

It compares synchronous-return correlation, common-factor-adjusted returns,
constrained DTW, level paths and frozen R2K moment groups. Static methods use the
same 39 currencies and January 2011–December 2019 calendar; memberships freeze
at that cutoff for later joint forecasts.

```bash
python -m pip install -e ".[clustering,reports,dev]"
python scripts/build_clustering_handoff.py
python -m pytest -q
```

One command exports the common numeric dataset, memberships, diagnostics,
source/reference provenance, sequence counts and the
[interactive HTML](reports/clustering/handoff/clustering_report.html),
[Markdown](reports/clustering/handoff/clustering_report.md), and
[PDF](reports/clustering/handoff/clustering_report.pdf) reports under
`reports/clustering/handoff/`. The notebook verifies saved source/code/output
hashes before reusing results, or invokes the same workflow. Generated data and
reports remain local and outside version control. The HTML embeds Plotly once and
works offline as a 3–5 minute findings brief: currency memberships, numerical
correlation gaps, silhouettes/Jaccard scores, fixed-scale heatmaps and K profiles.
Robustness and methodology are collapsed; the PCA/factor-reference analysis and
full reproducibility detail stay in the PDF/Markdown reference. Its appendices
include metric definitions, every method/K result and alternative memberships.
Temporal exports include `evolution_labels.npz` (fixed currency axis; −1 for
unavailable labels), monthly metrics, pair/currency summaries and initial-seed /
annual five-year-restart sensitivity. Pair frequencies condition on joint
eligibility. Summaries begin when all currencies can support 60-month returns;
monthly coverage and lag-comparison denominators remain explicit.
Conditional factor-reference ranges are exploratory, not significance tests or
neural forecasting results.

The reproducible default is adjusted-return average linkage, K=2 (25/14 currencies).
Every compared K=2–6 is available, together with the all-39 K=1 control. K=3
adjusted-average is the first grouped forecasting challenger against K=1;
K=2 remains the stability control, with K=4 adjusted-Ward an additional candidate.
K=2 is not an inferred true or forecast-optimal cluster count. Cluster selection does not remove the original
FX changes or targets from the neural inputs. Group sizes, stability and data
availability matter alongside separation. A stable complement can be internally
heterogeneous: the 25-currency group has within-return correlation 0.448 versus
0.442 outside, compared with 0.750 within the Europe-heavy group. Silhouettes
across different
representations are not a shared forecast-performance scale.

The framework-neutral loader selects stable currency channels from one bundle:

```python
from pathlib import Path
from fx_forecasting.data.neural_dataset import load_cluster_dataset

batch = load_cluster_dataset(
    Path("reports/clustering/handoff"), method="adjusted_average", k=2,
    cluster=0, split="train", horizon=3, feature_view="macro",
)
print(batch.X_local.shape, batch.X_global.shape, batch.Y.shape)
```

Inputs have shapes `X_local[N,12,C,7]` and `X_global[N,12,3]`, with availability
and carry masks; targets have `Y[N,C,H]` and a target mask. Currency IDs, input
months, forecast origins and endpoints are explicit. Local features cover FX
changes/volatility, REER gap and policy/inflation differentials. Common globals
cover the lagged USD index and quarterly credit. Optional macro medians and
feature scales fit only through December 2019; mandatory missing values remain
masked placeholders. Targets extending beyond each split cutoff are masked and
numerically zeroed.

Defaults use validation 2020–2022 and test 2023–August 2026. Longer historical
data are retained where sequences are eligible. For the recommended 25-currency
group, only 79 complete training origins support the 12-month target; this is
not thousands of independent neural examples. `sequence_counts.csv` reports
each group, feature view, horizon and split. Credit/no-credit model comparisons
must use intersected samples. Teammates choose and train their neural models.

## Robust Rolling K-Means (optional paper reference)

The earlier [scalar-routing audit](docs/r2kmeans_audit.md) is an optional historical
appendix. Its Ridge models do not test synchronized multi-currency inputs or
vector outputs, and do not restrict the joint-cluster forecasting goal.

The [R2K-Means notebook](notebooks/04_robust_rolling_kmeans.ipynb) implements
Algorithm 2 from Holmes, Hirsa, Klinkert and Malhotra (6 March 2024,
[SSRN 4729393](https://ssrn.com/abstract=4729393)) and adapts it to the selected
39-currency BIS roster. It includes synthetic examples, causal trailing FX
features, monthly updates with persistent cluster IDs, rolling-centroid decision
regions, independent/aligned K-Means baselines, and pre-2010 K/seed diagnostics.
The core algorithm and experiment live in reusable package modules; the notebook
explains their choices and displays the results.

```bash
python -m pip install -e ".[dev]"
python scripts/run_r2kmeans.py --tune-k
python scripts/run_r2kmeans_audit.py
python -m pytest -q
```

Select that environment as the notebook kernel. The experiment reads the local
ingested snapshot without downloads and saves labels, centroids, coverage,
normalization, diagnostics and a provenance manifest under
`reports/clustering/r2kmeans/`. The notebook adds figures and seed checks.
This is an implementation of the grouping method and an FX adaptation, rather
than a reproduction of the paper's mutual-fund results or a forecast-performance
claim. See [implementation and timing notes](docs/r2kmeans.md) for the paper-to-code
mapping, output schemas, limitations and incremental API.

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
