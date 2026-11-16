# Robust Rolling K-Means: implementation notes

**TA temporal review, 10 November 2026:** the primary report and notebook now
lead with continuous R2K histories and observed co-clustering in sliding
five-year windows. This follows Algorithm 2's continuous update; W=6/12 counts
monthly centroid states, not five-year history. An independent NumPy replay
checks 135 subsequent updates across K=2–4 and W=1/6/12. Initial seeds, first-fit
effort and annual 60-state restarts are sensitivities; none replaces the main
continuous path. The frozen neural handoff remains a separate interface below.

**Role clarified on 9 November 2026:** teammates need fixed currency groups and
synchronized inputs/outputs for joint neural forecasting. Start with
[the central comparison and handoff notebook](../notebooks/05_r2kmeans_methodology_audit.ipynb)
or run `python scripts/build_clustering_handoff.py`. This document describes the
paper reference implementation. The [historical scalar-routing audit](r2kmeans_audit.md)
is optional and does not decide whether joint multivariate forecasting is useful.

The experiment implements Algorithm 2 in Ryan Holmes, Ali Hirsa, Federico
Klinkert and Satyan Malhotra, *Robust Rolling K-Means (R2K-Means): an Updateable
Nonlinear K-Means Clustering Methodology for Financial Time Series*, supplied
version dated 6 March 2024, [SSRN 4729393](https://ssrn.com/abstract=4729393).
The algorithm is adapted to the existing BIS currency panel; the mutual-fund
experiments and proprietary performance measures in the paper are not replicated.

Start with [the executed notebook](../notebooks/04_robust_rolling_kmeans.ipynb).
It covers the method, synthetic checks and boundaries, the FX experiment,
pre-2010 K and seed diagnostics, membership histories and incremental use.

## Run locally

```bash
python -m pip install -e ".[dev]"
python scripts/run_r2kmeans.py --tune-k
python -m pytest -q
```

Use the project's virtual environment as the notebook kernel. The experiment
needs the existing `data/ingested/observations.parquet`. If absent, first run
`scripts/download_data.py` and `scripts/process_data.py`. Neither the experiment
nor the notebook downloads data or changes source files.

CLI options include `--n-clusters`, `--windows`, `--lookback`, `--start`, `--end`,
`--evaluation-start`, `--seed`, `--ingested-dir`, `--sample-path` and `--output-dir`.
For a shorter smoke run, use:

```bash
python scripts/run_r2kmeans.py --start 2008-01 --end 2011-12 --windows 1 6
```

Changing the start date changes the initialization path, so this shorter example
does not reproduce the full-history cluster IDs. Defaults use January 1995–August
2026, report comparisons from January 2010, and discard the first 12 updates from
summary diagnostics. Run at least 13 updates to obtain a post-burn-in summary.

## Code map

| File | Responsibility |
| --- | --- |
| `models/rolling_kmeans.py` | Incremental Algorithm 2; no scaling, downloads or forecast targets |
| `data/rolling_features.py` | Exact configured FX sources, regime masks, trailing FX moments, current cross-sectional normalization |
| `evaluation/rolling_clustering.py` | Chronological experiment, independent/aligned baselines, diagnostics, calibration, auditable exports |
| `evaluation/rolling_plots.py` | Membership heatmap, diagnostic histories and true two-feature decision regions |
| `scripts/run_r2kmeans.py` | Thin command-line entry point |
| `tests/test_rolling_kmeans.py` | Algorithm rules, validation, calendar boundaries and future-data independence |

Package paths in this table are relative to `src/fx_forecasting/`.

## Correspondence to the paper

| Algorithm 2 | Implementation |
| --- | --- |
| Steps 1–6: normalized cross sections and initial K-Means | Caller supplies scaled asset-by-feature frames; first update uses seeded k-means++ with 10 starts |
| Steps 7–8: initial centroid identities and size order | Initial IDs ordered once by ascending current Lloyd counts |
| Steps 12–15: smallest-first current-point selection | Stable size priority, each centroid snaps to a distinct current feature location |
| Steps 16–18: Lloyd refinement and new priority | One start with supplied seeds; original centroid IDs retained; counts saved for the next selection |
| Steps 9 and 19: rolling classification | Minimum distance over current and up to W−1 previous centroid states, with centroid identity carried across time |

Size sorting changes the order in which centroids claim points, not the numeric
IDs attached to historical paths. Counts used for the next selection are the
current Lloyd counts, consistent with sorting before rolling classification in
Algorithm 2. Rolling labels and current Lloyd labels are exported separately.
W changes classification only, so all R2K windows share the same fitted centroid
path and Lloyd objective for the same initial seed and feature stream.

The following choices complete details not fully specified by the paper:

- Use Euclidean distances and current cross-sectional z-scores (population
  standard deviation). Constant features have scale one, map to zero, and are
  flagged. Historical centroids stay in their original normalized coordinates.
- Equal-size clusters select in persistent ID order; equal-distance points use
  sorted asset IDs. Classification ties favor the newest centroid state, then
  the lower ID. A selected feature location excludes every exact duplicate at
  that location from subsequent seed selection.
- The first fit uses k-means++ rather than random point initialization; the paper
  allows first-period variants. Later fits specify `n_init=1`, `tol=0`,
  `algorithm="lloyd"`, and a 300-iteration limit. Scikit-learn supplies the
  numerical Lloyd solver and its deterministic empty-center handling.
- An iteration limit without a Lloyd fixed point, fewer than K distinct eligible
  locations, or a collapsed final Lloyd cluster causes an error before committing
  that update.
  A fit that converges on its last allowed iteration is accepted. There is no
  fallback random restart. Rolling classification can temporarily
  use fewer than K labels without invalidating the fitted current centroids.
- K and the feature schema remain fixed; currencies can enter and exit. The
  experiment requires consecutive calendar months, failing explicitly when a
  month's cross section cannot be fitted. W counts consecutive updates.
- Numerical libraries use one worker within fits to avoid parallel overhead for
  these small cross sections. Independent baselines use a new reproducible seed
  each month, while R2K randomness is confined to the first fit.

These choices should be reviewed before comparing this implementation against
another implementation of the same paper. Determinism after initialization does
not establish independence from the initial state, universal convergence to a
best partition, or outlier resistance. The audit provides a tested counterexample
to the paper's universal unchanged-data invariance statement: the snapping step
can change memberships even when the input frame stays identical.

## Currency features and timing

The [sample specification](../config/research_sample.json) supplies the exact
39 source IDs, first eligible months, and inclusive ARS/CNY/MYR exclusions.
Missing and nonpositive FX remain missing. A full contiguous lookback of monthly
log changes is required, resetting at missing or excluded months.

With `r = 100 * log(LCU_per_USD).diff()` and a 12-month lookback, use:

- `annual_log_change = 12 * trailing_mean(r)`;
- `annualized_volatility = sqrt(12) * trailing_std(r, ddof=1)`.

Positive log change is local-currency depreciation. Both features use information
through month end t; no forecast targets enter grouping. Current cross-sectional
normalization uses only eligible currencies at t. This represents relative
positions and can move when the eligible universe or its dispersion changes.
It does not provide a stable raw economic unit for centroid displacement.

The experiment reads revised historical BIS observations and applies a
retrospective non-peg sample definition. It does not establish real-time vintage
availability. This FX-only reference experiment does not build or impute macro
inputs. The joint forecasting handoff supplies the separately timed macro panel;
neural-network training remains the teammates' downstream task.

## Comparisons and interpretation

`kmeans_raw` independently refits K-Means with 10 starts each month.
`kmeans_aligned` uses the exact same fit, with Hungarian matching to the preceding
aligned centroids. The latter removes arbitrary relabeling while preserving
the baseline's partitions, silhouette and inertia.

Switch rates use only assets present in adjacent months; entrants/exits are
reported separately. Summary switch rates weight months by their common-asset
denominators. Adjusted Rand index measures partition similarity independently
of label permutations. Centroid displacement compares persistent IDs in
normalized coordinates. Displacement for raw K-Means is deliberately sensitive
to its arbitrary IDs.

Euclidean silhouette is most appropriate for the W=1 comparison, as in paper
section 6.1. W>1 boundaries can be nonconvex or disconnected, so their scores are
descriptive. The `--tune-k` flag saves a W=1 K=2–6 grid using only months strictly
before `--evaluation-start`; it does not select K automatically. K=3 and W=6
remain prespecified illustrations. Larger W need not yield lower turnover, and
neither stability nor separation demonstrates improved forecast accuracy.

The notebook's additional three-seed diagnostic also uses only pre-evaluation
history. It is not the paper's 1,000-simulation reproduction. Results depend on
the initialization date, initial seed, features, scaling and sample restrictions.

## Outputs

Default exports are local and ignored by Git under `reports/clustering/r2kmeans/`:

| Artifact | Contents |
| --- | --- |
| `features.parquet` | Raw two-feature table, including ineligible rows with nulls |
| `labels.parquet` | Origin, currency, method, rolling cluster and current Lloyd cluster |
| `centroids.parquet` | Current normalized centroids, indexed by month, method and persistent ID |
| `metrics.csv`, `summary.csv` | Per-month diagnostics and post-burn-in evaluation summaries |
| `normalization.csv` | Month/feature mean, applied scale and constant flag |
| `coverage.csv` | Eligible count, missing count, entrants, exits and excluded currencies |
| `currency_metadata.csv` | Reporting metadata, not predictive inputs |
| `calibration_grid.csv` | Optional pre-evaluation K diagnostic |
| `run_manifest.json` | Parameters, source/config/code hashes, versions and core output hashes |

The notebook adds seed diagnostics and five PNG figures. The manifest's `outputs`
map identifies the core files belonging to the most recent CLI/notebook run;
additional files in the directory may come from earlier runs. Standardize
parameters and use separate output directories to preserve alternative runs.

The optional `run_r2kmeans_audit.py` command saves historical evidence under `audit/`.
It adds warm-start comparisons, group-size and feature-geometry checks, causal
scaling/lookback sensitivities, initialization checks and a fixed FX-only forecast
screen with common rows, mature training targets and explicit fallback behavior.
`run_cross_sections` supports `extra_baselines=True` and
`scaling="initial_cross_section"` for those documented diagnostics. The latter
fits the scaler at the very first update and holds it fixed, avoiding a full-
history scaling fit. This is a sensitivity convention, not the reference default.

## Fixed groups and dynamic descriptors have different timing rules

For joint forecasting within a fixed group, fit the grouping entirely on the
model's training window and freeze currency membership and channel order for
later validation/test forecasts. A group fitted through December 2019 may
legitimately assign all historical training observations to that group; this is
train-fitted preprocessing, not leakage into evaluation after December 2019.
A frozen final R2K snapshot can define such a group, although its meaning is a
return/risk profile rather than synchronized co-movement.

For a dynamic R2K regime descriptor, join labels by **currency and forecast
origin**, retaining the causal state computed at each origin. Do not recompute
earlier evaluation descriptors from a later fitted state, or replace a different
currency at the same neural input position when membership changes. Keep a fixed
currency axis and represent changing regimes through context labels or masks.

Under either design, a historical h-month target enters training only after its
endpoint is at or before the training cutoff. A 2019-fitted grouping must not be
used to claim a January 2010 out-of-sample experiment. Forecast usefulness needs
its own matched joint-model evaluation; scalar row-routing results do not settle
that question. See [the main report](../reports/clustering/handoff/clustering_report.md)
for the teammate dataset contract and comparison.

Numerical solver reference: [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
