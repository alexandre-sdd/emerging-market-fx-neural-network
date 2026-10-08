# Data

All five official BIS bulk exports were downloaded on 8 October 2026. Original
ZIPs and unchanged CSVs live in `raw/`; the selected descriptive matrices and
their EDA metadata live in `processed/`. The canonical ingested dataset lives
in `ingested/`. Data files are ignored by Git.

Run `python scripts/download_data.py` to acquire missing files. Source URLs,
UTC download timestamps, HTTP metadata, CSV sizes, and SHA-256 hashes are recorded
in `raw/download_manifest.json`. Existing files are reused; rerunning the script
does not automatically refresh the snapshot.

## Download and ingestion commands

```bash
python scripts/download_data.py
python scripts/process_data.py
```

Downloading only saves source exports and their provenance. Processing only
reads local exports, selects the source definitions below, parses dates and
numeric values, and writes tidy Parquet tables. It does not run EDA, construct
features or forecast targets, impute values, align credit to months, or perform
statistical tests.

`download_data.py --force` refreshes the five exports. Both commands accept
`--raw-dir PATH`. Processing also accepts `--output-dir PATH`, `--start YYYY-MM`,
and `--end YYYY-MM`; defaults use the repository paths and January 1994–August 2026.

## Ingested dataset

`ingested/observations.parquet` contains one row per `(dataset, series_id, period)`:

| Column | Meaning |
| --- | --- |
| dataset | Original BIS dataflow, such as `WS_XRU` |
| series_id | Unchanged BIS series identifier |
| frequency | Native monthly `M` or quarterly `Q` frequency |
| period | `YYYY-MM` or `YYYY-Qn` reference period |
| period_start | First calendar date of the reference period |
| period_end | Last calendar date of the reference period |
| value | Nullable numeric source value, retaining its original scale |

`ingested/series.parquet` describes each `(dataset, series_id)` with the component
(`fx`, `reer`, `neer`, `policy`, `cpi`, or `usd_credit`), frequency, reference-area
code/name, source currency code where supplied, units, multiplier, index base year,
collection convention, source title, and every original metadata field in
`source_metadata_json`. Currency codes are not inferred for sources that omit them.

Credit values are in millions of USD: `unit_code=USD`, `unit_multiplier=6`.
No unit conversion occurs. CPI and effective-rate index base years are retained.
Period dates describe the observation, not its release or publication date.

`ingested/manifest.json` records sample limits, source URLs and hashes, source
validation checks, row/missing counts, output hashes, and schema version. The
current snapshot produces 170,722 rows (including 6,468 null values) and 445 series.
It covers all selected source series because the 39-currency roster is unspecified.

```python
import pandas as pd

observations = pd.read_parquet("data/ingested/observations.parquet")
series = pd.read_parquet("data/ingested/series.parquet")
data = observations.merge(
    series, on=["dataset", "series_id", "frequency"], validate="many_to_one"
)
```

Missing cells remain rows with null values, including expected periods absent
from an export. Credit begins no earlier than 2000Q1 and includes only quarters
completed by the end sample month. Duplicate keys, unexpected nonnumeric tokens,
infinite values, or source-hash mismatches stop ingestion. Existing ingested files
are replaced only after parsing, validation, and staging finish successfully.

## Sources and selection

The [official BIS bulk listing](https://data.bis.org/bulkdownload) supplies wide
CSV exports: one series per row, with metadata and native time periods in columns.
The entire original export is retained, including frequencies outside the EDA.

| Dataset | Selected definition | Available selected universe | Units |
| --- | --- | --- | --- |
| WS_XRU | Monthly, end of period (`M`, `COLLECTION=E`) | 192 reference areas, 147 currency codes | Local currency units per USD |
| WS_EER | Monthly broad real and nominal (`M`, `EER_BASKET=B`, types `R` / `N`) | 64 economies for each type | Index, 2020 = 100 |
| WS_CBPOL | Native monthly policy rate (`M`) | 49 histories, including discontinued national euro-area rates | Percent per annum, end of month |
| WS_LONG_CPI | Monthly index levels (`M`, `UNIT_MEASURE=628`) | 63 economies | Index, 2010 = 100 |
| WS_GLI | Quarterly total USD credit to non-banks (`Q:USD:*:N:A:I:B:USD`) | Outside-US aggregate and 12 country series | Millions of USD (`UNIT_MULT=6`) |

The USD index is the BIS broad nominal effective-rate series `M:N:B:US`.
An increase in bilateral FX means depreciation of the local currency against USD;
an increase in an effective-rate index means appreciation.

Selected credit countries: Argentina, Brazil, Chile, China, India, Indonesia,
Malaysia, Mexico, Russia, Saudi Arabia, Türkiye and South Africa. The outside-US
aggregate uses borrower code `3P`. Regional aggregates, bank-loan-only components,
debt-security-only components, and year-over-year changes are excluded.

## Study-window quality review

Monthly matrices cover January 1994–August 2026 (392 months). Quarterly credit
covers the expected completed-quarter calendar 2000Q1–2026Q2 (106 quarters),
with only 2000Q1–2026Q1 available in this download. Reindexing explicitly marks
2026Q2 missing. No quarterly-to-monthly conversion is performed.

Blank fields and BIS `NaN` markers are treated as missing. Daily/annual/quarterly
columns do not enter monthly missingness denominators. Missing observations are
separated into leading, internal and trailing gaps, with exact contiguous gap
dates exported. Source-row and series-ID duplicates are checked on the full raw
exports. Repeated currency codes across reference areas are reported as a mapping
issue, not removed as duplicate records.

Run `python scripts/run_eda.py` or execute `notebooks/01_eda.ipynb` to regenerate
the [initial review](../reports/eda/README.md), tables, and charts. EDA output files
are prefixed `eda_`; they are descriptive matrices, not model-ready panels.

## Sample decisions still open

The exact 39-currency list is absent from the brief. All monthly end-of-period
FX series are reviewed rather than imposing an arbitrary roster. The current EER
and CPI universes exceed the brief's 63 and 62 counts. Reference-area overlap is
in `reports/eda/currency_overlap.csv`.

No missing observations are imputed, no large changes are removed, and no pegs
are excluded in this EDA. Currency groups, peg dates, shared euro policy-rate
mapping, and missing-data treatment need explicit definitions before modeling.
Credit must later enter with a two-quarter lag and a documented monthly
availability rule. These are revised historical exports, not real-time vintages.

Definitions: [FX](https://data.bis.org/topics/XRU),
[EER](https://data.bis.org/topics/EER),
[policy rates](https://data.bis.org/topics/CBPOL),
[CPI](https://data.bis.org/topics/CPI),
[global liquidity](https://data.bis.org/topics/GLI).
