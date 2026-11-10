# Optional historical R2K scalar-routing audit

This is a supporting appendix to the
[currency-clustering handoff report](../reports/clustering/handoff/clustering_report.md).
The primary workflow is `python scripts/build_clustering_handoff.py` and
[notebook 05](../notebooks/05_r2kmeans_methodology_audit.ipynb). It compares fixed
currency groups for **joint multivariate neural forecasts**, with synchronized
inputs, targets, masks and channel identities.

## What the historical screen tests

`python scripts/run_r2kmeans_audit.py` remains available to reproduce the earlier
R2K audit and scalar Ridge experiment. It routes individual currency-month rows
to separately fitted scalar models using their historical origin-time labels.
It does not consume other currencies as simultaneous input channels or predict
a joint vector of currency outcomes. Its result cannot reject joint forecasting
within fixed clusters, group-based neural minibatching or a regime descriptor
used as an auxiliary input.

The screen uses revised BIS history, expanding fits from January 2010, common
evaluation rows, four FX-only predictors, fixed Ridge regularization and mature
training targets. It reports observation-pooled and equal-currency metrics.
The post-2010 data had already been inspected; the results are exploratory and
not independent significance evidence.

## Useful reference findings and their limits

- Raw K-Means numerical labels exaggerate turnover. Hungarian alignment removes
  pure relabeling, while ordinary warm-start K-Means provides a stronger control
  for the paper's snapping step. The historical two-moment stream favored warm
  starts on stability; this does not select a joint forecasting architecture.
- Overlapping 12-month return/risk features already have high persistence. Small
  current groups and sensitivity to feature/scaling choices deserve disclosure.
  A low turnover or high silhouette score is not forecast-performance evidence.
- A reproducible 12-point example changes three memberships after an identical
  frame is supplied again. Deterministic snapping does not guarantee universal
  unchanged-data invariance. The paper algorithm was retained, with a regression
  test and exported counterexample rather than a no-op shortcut.
- The tested hard-routing scalar Ridge variants trailed pooled Ridge in the
  historical screen. This is a narrow finding about that model and feature set;
  it does not govern the user's joint multivariate goal.

## Membership timing

Groups fitted entirely within a training cutoff may be frozen and assigned to
all training observations for models evaluated afterward. This is legitimate
train-fitted preprocessing. For the agreed handoff, groups fit through December
2019 stay fixed for validation from 2020 onward. Currency/channel order remains
constant.

Dynamic regime labels have a different contract: each historical descriptor must
use its origin-time state. They cannot silently change which currency occupies
an input channel. Both designs still require mature targets and training-only
preprocessing; neither permits using 2019-fitted groups for a 2010 evaluation.

## Evidence and maintenance

Optional historical outputs live under `reports/clustering/r2kmeans/audit/`,
including partition diagnostics, scalar forecast tables, support counts,
the stationary counterexample and an audit manifest. The manifest hashes only
the explicit files produced by that invocation, excluding stale directory files.
The main handoff does not run or depend on this scalar screen.

Paper implementation details remain in [the R2K reference notes](r2kmeans.md).
The main report contains the consolidated method/data/code review and teammate
quick start; new project-wide conclusions should be made there rather than in
this historical appendix.
