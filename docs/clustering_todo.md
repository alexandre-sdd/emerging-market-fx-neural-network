**Clustering and external features — shared to-do list**

16 November 2026. Suggested owners need team agreement. Check a task only when its deliverable exists. [Detailed research plan](clustering_exogenous_plan.md).

**Today's outcome:** a reproducible BIS-only prototype, a first matched forecast comparison, and a source ledger ready for economic annotations.

Today's work session:

1. Coding support builds and runs the correlation/policy prototype, validates it, and exports the first results.
2. Completed: six formal CNY events and DKK's EUR-band parameter announcement are documented in the ledger, alongside the MYR/CHF starter records.
3. Completed: the CHF/EUR review identifies a high-correlation floor period, the January 2015 break and high correlation without the floor. Its annotated chart separates the policy episode from rolling-window memory.
4. Review the matched forecast table and choose the next feature block from validation evidence. Keep intervention, IMF/debt collection and neural training as later checklist items unless time permits.

| Workstream | Suggested owner | Deliverable |
| --- | --- | --- |
| Data and feature implementation | Alexandre + coding support | Reusable correlation/policy features and a reproducible run |
| Historical arrangements and explanations | Alexandre; Nicolas if available | Dated announcements and annotated correlation episodes |
| Forecasting and robustness | Forecasting teammate, to agree | Matched feature comparisons and sensitivity results |
| Literature and review | Shared | Paper comparison and reviewed experiment settings |

**First: BIS-only prototype**

- [x] Inspect the saved dataset and document the existing seven local and three global predictors.
- [x] Document missing policy/CPI sources, partial country-credit coverage, and the effect of existing peg exclusions.
- [x] Save the initial experiment settings before running the new forecast screen: primary horizon 3 months, correlation window 9 months, alternatives 6/12/18, and existing 2019/2022 split cutoffs. [Settings](../config/clustering_exogenous.json).
- [x] Calculate monthly rolling pair correlations on raw and median-adjusted returns; require uninterrupted paired observations and export observation counts.
- [x] Export compact EUR, CNY and fixed-group peer-correlation features, plus training-fitted high/low flags and missingness information.
- [x] Derive domestic/US policy changes and three-month policy divergence using the existing publication lag; do not turn carried observations into policy actions.
- [x] Check future-data invariance, window resets, missing values, fixed memberships and mature training targets. 47 related tests passed with a noninteractive plotting backend.
- [x] Run a matched pooled Ridge screen: existing features; + correlation; + policy changes; + both. This is the BIS-only prototype; arrangement features enter later.
- [x] Export per-currency MAE/RMSE, equal-currency averages, zero-change and own-history controls, and a correlation timeline.

First results: the prespecified nine-month correlation features slightly improve test RMSE (4.697 → 4.668) but worsen validation RMSE (5.076 → 5.106) and trail the zero-change test forecast (4.587). There is no consistent improvement in this first screen. These are exploratory pooled-regression results, separate from joint neural forecasting.

Open [the offline findings and timeline](../reports/clustering/context/findings.html) or [written findings](../reports/clustering/context/findings.md). Share the HTML as one attachment for its embedded results and interactive charts; accompanying CSV links require sharing the report folder. Reproduce with `.venv/bin/python scripts/run_currency_context.py` from the repository root. Generated reports remain local; the script and settings reproduce them from the existing data snapshot.

**Historical evidence and source recovery**

- [x] Expand [the event ledger](clustering_event_ledger.md): six formal CNY reforms, one DKK parameter announcement, and the three existing MYR/CHF events. [Numeric research registry](../config/currency_arrangement_events.json).
- [x] For the selected verified events, record document/public/effective dates, primary evidence and dating limitations. Preserve the 2012/2014 document-date discrepancies and the delayed surviving 2015 CNY page.
- [x] Distinguish anchor/reference, band and one-sided floor. Leave the unverified announcement date for CNY's 2008 de facto stabilization unknown.
- [x] Write [three CHF/EUR observations](clustering_chf_eur_case_study.md), including a counterexample, and export the [annotated chart](../reports/clustering/context/chf_eur/case_study.html), both return views and window-exit diagnostics.
- [x] Locate the original 35-predictor study report in Downloads; verify its duplicate is identical; recover the twenty published feature descriptions and their selection counts. [Source audit](data/feature_study_source_audit.md), [inventory](../config/feature_study_inventory.json).
- [x] Fetch all origin refs and perform a fast-forward-only pull in an isolated main checkout. The inspected main snapshot was `6db7402` (28 October), with the same 25-cell `fx_cleaned.ipynb`; no RFE/RFECV source was found there. This check preceded publication of the context workflow.
- [ ] Recover the actual 35-feature matrix/generating notebook and remaining fifteen definitions. Current blocker: the report is found, but a fresh main pull lacks the newer study; GitHub reports zero forks and no PRs, and no alternate repository/commit is known. Requires a source push, notebook/data bundle or link from the study's author.
- [ ] Finish reconciling the 35-predictor study's numerical inputs, transformations, coverage and release lags once its source files are available.

**Next: external features**

- [ ] Build a peg-retaining descriptive panel separately from the existing forecasting sample.
- [ ] Complete the dated arrangement/anchor ledger across the selected currencies and convert verified entries into origin-time features.
- [ ] Audit IMF intervention data and country disclosure delays; distinguish official observations, estimates and unknown coverage.
- [ ] Add IMF approval/active-program and debt-announcement features where public dates are verified. Start with three-month recent-event flags.
- [ ] Assess White House/Congress control as an optional global block. Define a reproducible source/coding rule before adding Fed-pressure features.
- [ ] Pin dataset releases and retain source, public-date, coverage and missingness metadata.

**Then: controlled forecasting and robustness**

- [ ] Run the full four-way comparison using arrangement + policy features as the economic block; compare intervention and IMF/debt additions separately.
- [x] Compare 6/9/12/18-month correlation windows and raw versus median-adjusted returns in the pooled prototype; retain the 60-month structural reference. [Complete grid](../reports/clustering/context/forecast_summary.csv).
- [ ] Test 5%/10% training-scale continuous-feature noise across ten seeds; report fixed-model input sensitivity separately from pipeline refits.
- [ ] Test an additional one-month disclosure delay and event persistence of 1/3/6 months.
- [ ] Compare pooled and fixed-group compact joint neural models on matched origins, target cells, inputs and capacity.
- [ ] Report per-currency and equal-currency forecast errors, seed variation and synchronized time-block uncertainty. Label previously inspected evaluation periods exploratory.
- [ ] Compare anchor findings with Frankel–Wei and IRR; keep R2K as the evolving-moment method reference.

**Handoff:** share this checklist, the event ledger, the generated findings and the command needed to reproduce them. Keep historical explanations and measured forecast improvements as separate outcomes.
