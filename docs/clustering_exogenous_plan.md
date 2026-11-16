**Plan: explain changing currency relationships and test their forecasting value**

Draft for the group, 16 November 2026. This records proposed decisions; it does not implement features or report new forecasting results.

Execution follow-up: [shared checklist](clustering_todo.md), [verified arrangement/event records](clustering_event_ledger.md), [CHF/EUR case study](clustering_chf_eur_case_study.md), and [35-predictor source audit](data/feature_study_source_audit.md). The original RFE report is now located, with twenty recoverable descriptions; the numerical dataset/notebook is still missing from the freshly fetched main branch.

The next question should be: **Do observable exchange-rate arrangements, policy divergence and economic events help us describe changing currency relationships, and does that information improve future FX forecasts?** Keep the current clustering results as a baseline. Build a small set of interpretable context features before adding more clustering algorithms.

The professor's allowance for looking backwards is useful: investigate conspicuous historical episodes, consult institutional histories, and formulate mechanisms. Then encode the proposed mechanism consistently across countries and dates. A historical explanation, an observable predictor and a causal explanation are different claims. Policy changes and intervention can respond to currency movements; calling them external inputs does not identify exogenous shocks.

**1. Start from the data actually available**

The canonical dataset covers 39 currencies, January 1994–August 2026. Its current features were checked against [the panel builder](../src/fx_forecasting/data/panel.py) and the saved handoff metadata. Availability below is before imputation, through December 2019; local percentages pool currency-month cells, whereas global percentages count months.

| Existing forecast input | Availability | Timing at month-end origin t | What it can contribute |
| --- | --- | --- | --- |
| Own FX changes over 1, 3 and 12 months | 95.7%, 95.1%, 92.0% | Through t | Momentum and recent depreciation; these are endogenous history features |
| Annualized trailing 12-month FX volatility | 92.0% | Through t | Recent instability; already part of the R2K representation |
| REER gap against a preceding 60-month baseline | 79.1% | Reference month t−2 | Relative valuation context, without assuming the gap must close |
| Domestic minus US policy rate | 79.6% | t−1 | Relative monetary stance; changes can measure observed divergence |
| Domestic minus US annual log inflation | 87.9% | t−2 | Relative inflation context |
| Broad USD NEER monthly change | 99.4% | t−1 | Shared dollar conditions |
| Outside-US USD credit log stock and annual growth | 75.0%, 71.2% | Origin quarter minus two quarters | Slow shared dollar-financing conditions |

Important implications for the new work:

- Raw local NEER and the domestic/US policy series exist but are not all exported as individual forecast inputs. They can support additional features without buying or collecting another dataset. NEER is another exchange-rate measure, so strong same-month associations with bilateral FX are not independent forecasting evidence. Assign and document its publication lag before using it.
- Policy sources are absent for BAM, DZD, SGD and TWD; CPI sources are absent for BAM and TWD. Japan has additional policy gaps. Missing policy data must not become a flag for monetary independence or no policy action.
- The ingested credit data include country-specific series for 11 selected currencies: ARS, BRL, CLP, CNY, IDR, INR, MXN, MYR, RUB, TRY and ZAR. The current model uses the shared aggregate. Country-level credit is a later, explicitly partial-coverage extension; the dataset supplies neither full-panel dollar exposure nor comparable exposure/GDP ratios.
- Oil appears as a live-download experiment in [fx_cleaned.ipynb](../notebooks/fx_cleaned.ipynb), rather than a versioned canonical input. The progress report describes a separate 35-predictor exercise, including trade, equities and yields; its complete source table is not in the canonical handoff. Reconcile its data, transformations and release lags before treating those predictors as available to this pipeline.
- Global USD credit, USD NEER and US politics have the same value for every currency at a given origin. They can influence forecasts, but cannot separate currencies in a same-month cross-sectional clustering by themselves. Currency differences require local variables or interactions with independently measured exposures.

See [the existing sample and treatment decisions](data/sample_and_treatment.md) and [the current handoff](../reports/clustering/handoff/clustering_report.md) for complete definitions. The feature counts above exclude availability and carry masks.

**2. Keep the forecasting sample and a regime-analysis sample**

The current configuration removes AED, HKD and SAR altogether and masks ARS, CNY and MYR peg/transition intervals. Those rules support the existing non-USD-peg forecasting experiment, but remove much of the variation needed to study a USD-peg flag. Raw observations for the persistent USD pegs are present in ingestion.

Keep the existing forecast specification unchanged for the first comparison. Create a separate analysis view that retains observed peg intervals, respects currency launches and quotation changes, and records the regime instead of deleting the observation. Persistent pegs may enter a clearly labeled descriptive extension. Never bridge a launch or missing-data gap when calculating returns. Evaluate any later change to the forecasting universe as its own experiment.

Record **anchor currency as well as flexibility**: a EUR-linked currency can move considerably against USD and track EUR/USD closely. A single generic peg dummy would lose this distinction. Zero-variance return windows have undefined correlation and must remain unavailable, rather than being assigned correlation zero.

Retrospective de facto regime classifications are valuable annotations. Predictive regime status should use dated public arrangements or a trailing, origin-time estimate. Do not assume a classification reconstructed using later observations was known at its historical reference date.

**3. Measure the changing relationship before explaining it**

Use monthly log returns in the existing quotation, `100 × Δlog(LCU per USD)`, so positive values mean local depreciation. Calculate pairwise Pearson correlations over **nine trailing monthly returns, updated every month**. Nine observations are a noisy short-run descriptor, not a precise estimate of a permanent relationship. Require all nine paired returns on the common calendar; do not interpolate, skip across gaps or manufacture a zero return.

Prespecify nine months as the primary window, with 6, 12 and 18 months as sensitivities. Keep the existing 60-month correlation analysis as the structural reference. These windows are separate from the neural input lookback, the R2K moment lookback and R2K centroid memory.

Export every eligible pair and month, with observation counts, for both original returns and the existing cross-sectional-median-adjusted returns. Use original-return descriptors for the first feature experiment; use adjusted-return descriptors as a prespecified alternative. Median adjustment is a statistical reference and can change dependence mechanically; it is not an identified pure USD shock.

Avoid passing all 741 pair correlations into the first forecast model. Start with three local summaries:

| Descriptor | Definition | Guardrail |
| --- | --- | --- |
| EUR relationship | Nine-month correlation with EUR/USD returns | EUR's self-correlation is unavailable, with a mask |
| CNY relationship | Nine-month correlation with CNY/USD returns | CNY's self-correlation and excluded CNY history are unavailable |
| Peer coherence | Mean correlation with other members of the training-fitted fixed group | Exclude self; require at least two eligible peers and export the peer count |

Compute summaries from eligible pair values; unavailable results retain masks. Freeze group membership at the training cutoff and use the same group definition for all feature ablations. If peer eligibility changes, report it rather than interpreting a composition change as an economic event. Do not correlate currencies against a USD/USD series, which is constant.

Use continuous descriptors first. To address the requested on/off regimes, derive high/low flags from each descriptor's training-period upper/lower quartiles and freeze the thresholds for validation/test; the middle half is neutral. These flags are alternative encodings of the same information. Thresholds for walk-forward experiments must be re-estimated within each training fold. A retrospective high-correlation episode must never be copied backwards into a historical forecast feature.

Build one common timeline showing correlation, regime announcements, policy divergence and event dates. Examine a few prespecified hypotheses—EUR-linked pairs, CHF/EUR policy episodes, CNY/MYR arrangement changes, and commodity-sensitive pairs if oil is added—then apply the same measurement to all available periods. Keep cases with weak or contradictory evidence. Describe mechanisms as hypotheses unless the design identifies a causal effect.

**4. Add external information in a practical order**

| Priority | Feature family | Initial representation | Data and timing decision |
| --- | --- | --- | --- |
| First | Exchange-rate arrangement and anchor | Publicly announced USD/EUR/other/no single anchor; arrangement type; months since last announced change | Dated central-bank documents plus a complete country chronology. Separate public rules from retrospective de facto classifications. Include an unknown category. |
| First | Monetary-policy divergence | Domestic and US rate changes; three-month change in the domestic–US rate differential | Derive from BIS sources using the existing t−1 convention. A rate differential already exists; test incremental changes rather than renaming it. Do not infer a policy move from a carried value. |
| Second | FX intervention | Signed reported amount where comparable, recent-public-intervention flag, official/estimated indicator and missingness | Audit IMF WPFXI coverage and country disclosure schedules. Use the date the action/quantity became public, which may differ from its transaction month. Treat undisclosed action as unknown. |
| Second | IMF arrangements | Public approval announcement, arrangement active as currently known, months since approval | IMF announcements provide the earliest verified public dates; MONA supplies structured program histories and cross-checks. Do not use eventual completion or duration before known. |
| Second | Debt stress events | Public default/restructuring announcement and recent-event flag | Maintain a dated event ledger. Annual default stocks are a historical check, not monthly announcement dates. No automatic backdating to the start of a default year. |
| Later | US institutional politics | White House, House and Senate control; divided-government flag | Use effective tenure/control dates. Global context with few independent transitions; start as a small optional block. An election result and taking office are separate events. |
| Later | Political pressure on the Fed | Only a reproducibly defined, timestamped measure | Specify a source corpus and coding rule before inclusion. Do not retrospectively rate entire presidencies or code pressure based on subsequent FX behavior. |
| Later | Oil/trade/equity/yield conditions | Small documented subset of the teammate feature set | First recover saved sources, transformations, dates and publication lags. Global oil needs measured country exposure to distinguish countries; do not infer commodity-export status from FX returns. |

“Moves independently of the Fed” should initially be called **observed policy divergence**. Same-direction moves can reflect shared inflation rather than following the Fed; different-direction moves do not prove independence. A descriptive same/opposite/no-move indicator can be a sensitivity, with thresholds fixed beforehand. More ambitious policy-response residuals need sufficient trailing history and remain statistical proxies.

For event features, use a three-month “public event in the trailing three months” flag as the initial persistence rule, with one- and six-month alternatives in robustness. Active arrangements are state variables, not arbitrary event windows. Keep separate flags for intervention, IMF approval and debt events; do not merge them into a subjective crisis score.

The [IRR authors' data page](https://www.ilzetzki.com/irr-data) currently lists monthly regime/anchor data through 2019 in the stable November 2021 version and through 2025 in an October 2026 beta version. Pin the chosen release; record gaps beyond its coverage. These retrospective datasets support interpretation rather than automatically satisfying historical availability. The [IMF intervention dataset](https://data.imf.org/en/datasets/IMF.STA%3AWPFXI) distinguishes official observations and estimates and starts in 2000. Its estimates adjust for more than simple reserve changes, so changes in reserves should not be relabeled intervention.

[IMF MONA documentation](https://www.imf.org/en/news/articles/2015/09/14/01/49/pr0909) states that database updates follow approvals/reviews by weeks; a verified public announcement may be earlier. The [BoC–BoE default database](https://www.bankofcanada.ca/2025/10/staff-analytical-note-2025-24/) reports annual default stocks; use dated primary announcements to construct monthly event features.

**5. Make historical explanations usable at a forecast origin**

Every new source observation/event needs: currency or global scope, feature/event type, value and unit, economic reference/effective date, first verified public date, source URL, retrieval date, source vintage, and coverage/verification status. Event states also need the end date *as known then*, not only the final historical end date. Keep the annotations and source records separate from numeric model exports.

At month-end t, join only information public by t. If intramonth timing cannot be verified, first make the item usable in the following month. An economic event dated in the past but disclosed later enters on disclosure. An announcement describing a future change can be recorded immediately as an announcement; effective arrangement status changes when the announced rule takes effect.

For existing macro series, retain the documented conservative reference lags. They use revised BIS history and define a pseudo-out-of-sample exercise, not a real-time-vintage experiment. A generic one-month lag does not fix a retrospective regime classification or an intervention estimate built using later data.

Missing numeric macros may use the existing training-fitted imputation with masks. Categorical regime uncertainty gets an unknown category. An event flag equals zero only where source coverage establishes that no qualifying public event was recorded; otherwise it is unknown. Keep rare event families small and report the number of countries and episodes supplying variation.

**6. Test forecasting value through a short sequence of matched experiments**

Use **three-month FX change as the primary target**, with 1 and 12 months as prespecified secondary horizons. Maintain the fixed currency axis and original FX targets. The current grouping fit ends in December 2019; validation is 2020–2022 and test starts in 2023. These dates have already been inspected, so results from the proposed exercise remain exploratory historical validation.

First use a regularized pooled forecast model to compare four input sets:

| Experiment | Inputs | Question |
| --- | --- | --- |
| A | Existing canonical inputs | Reproducible baseline |
| B | A + compact correlation descriptors | Does observed relationship history add information? |
| C | A + arrangement/anchor and policy-divergence features | Do the first economic-context features add information? |
| D | A + both blocks | Do they complement each other? |

Treat intervention and IMF/debt variables as subsequent, separately reported blocks after their coverage audit. Retain continuous correlation as the primary encoding; test on/off flags separately. Keep preprocessing, candidate regularization settings and scoring rows identical across the four runs. New optional variables should retain masks; report any coverage-driven sample loss. Compare existing features against new features on a common sample before evaluating their extra available history.

Then take one validation-selected input set to the joint neural comparison: all currencies together (K=1) versus the existing fixed K=3 groups, with K=2 as the stability control. Match origins, target cells and parameter budgets. A separate scalar regression per group is a useful baseline but does not test joint multivariate neural forecasting. Dynamic descriptors enter on the same fixed currency channels; do not swap currencies between tensor positions each month.

Report zero-change/random-walk and own-history regularized controls, per-currency MAE/RMSE and equal-currency aggregates. State the benchmark explicitly for any R². Summarize seed variation and dependence-aware error differences. Resample synchronized time blocks, preserving common shocks and dependence from overlapping targets; do not treat all currency-months or overlapping nine-month windows as independent replications.

All fitted preprocessing, feature selection, grouping, thresholds and hyperparameter decisions belong inside the relevant training/validation procedure. Training targets must have matured by its cutoff. Do not reuse full-history RFE lists or 2019-fitted clusters for a claimed 2010 forecast experiment. If adding expanding historical folds, refit those objects within each fold and keep validation/test outcomes out of fitting. Post-2019 inspection cannot be undone by calling the current test set untouched.

Use compact models: the existing common macro-plus-credit handoff has only 88 three-month training origins and 79 twelve-month origins when requiring the broad group's complete mandatory history. More currencies are not more independent time origins. Retain the current complete-history loader rule for the first controlled comparison; evaluate a mask-aware partial-history design separately if sample size remains binding.

**7. Fix robustness rules before the next runs**

- Correlation windows: 9 months primary; 6, 12 and 18 months reported together. Keep the 60-month structural reference distinct. Use the same timing convention and report coverage for every window.
- Continuous-feature perturbations: Gaussian noise with standard deviation 0%, 5% and 10% of each feature's training-period scale, over ten recorded seeds. Perturb observed continuous cells only; preserve missingness and do not add Gaussian noise to categories, masks or dates. Use training scales for validation/test perturbations as well.
- Run a test-input-noise experiment with the trained model fixed. Separately refit the clustering/feature/model pipeline under perturbed training data. Report forecast degradation and partition/descriptor changes separately; noise stability is not forecast evidence.
- Respect units and bounds. Perturb underlying continuous inputs before recomputing bounded correlation descriptors; retain unperturbed targets. Shared global-input noise is drawn once per month, rather than independently for each copy across currencies. Noise injection is a sensitivity, not a measurement-error model.
- Date uncertainty: shift event availability one additional month; test recent-event persistence of 1/3/6 months. Preserve the distinction between uncertain timing and a known zero event.
- Structural checks: raw versus median-adjusted returns; with/without retained EUR-linked currencies; original forecasting sample versus the explicitly labeled peg-retaining extension; same-sample comparisons with/without credit.
- Show per-currency results and whether any gain disappears when one crisis country or a prespecified exceptional episode is removed. Do not select and present only the favorable window, noise seed or subgroup.

**8. Use a closer economic reference alongside the R2K paper**

The most directly relevant economic benchmark found for the proposed explanation work is **Frankel and Wei (2008), Estimation of De Facto Exchange Rate Regimes: Synthesis of the Techniques for Inferring Flexibility and Basket Weights**. It estimates anchor weights and flexibility rather than treating co-movement as unexplained geometry. Compare whether known anchor relationships and changes match the observed correlation patterns. [Paper](https://www.imf.org/external/pubs/ft/staffp/2008/03/pdf/frankel.pdf).

Start with one-anchor exposures using sufficient trailing history. A fuller Frankel–Wei-style exercise needs a consistent non-USD numeraire and several anchor return series; USD/USD returns cannot estimate a USD coefficient. Use the available bilateral quotes to construct consistent crosses and avoid choosing a candidate anchor as the numeraire. With monthly observations, use a longer prespecified estimation window (36 months, with 60 as sensitivity), check rank/collinearity and label this an adaptation. Nine observations are insufficient for a credible multi-anchor model. Replicating the paper's flexibility/market-pressure specification requires additional reserve/pressure information absent from the current canonical panel.

Use **Ilzetzki, Reinhart and Rogoff (2019), Exchange Arrangements Entering the 21st Century: Which Anchor Will Hold?** as a classification/chronology reference ([authors' paper](https://www.ilzetzki.com/_files/ugd/b3763a_b6b99ab2b81f4b7f84dfd88788b69ae8.pdf)). Report agreement and disagreement, including dates and sample differences. These studies are economic benchmarks, not evidence that our neural forecasts should improve.

Keep the already implemented R2K comparison as a method reference for evolving return/volatility states; its objective differs from correlation grouping. The existing audit is documented in [the R2K notes](r2kmeans.md). No additional algorithm reproduction is needed before the first feature experiment.

**9. Suggested division and concrete handoffs**

The discussion names Ali and Joerg as sources of feedback; it does not establish team assignments. The following is a proposed division by workstream.

| Workstream | Proposed responsibility | Reviewable output |
| --- | --- | --- |
| Alexandre | External features and data organization, consistent with the message to Nicolas | Source/availability registry; arrangement chronology; BIS policy-divergence features; coverage audit for intervention and IMF/debt |
| Nicolas, if agreed | Monthly relationship measurements and explanation timeline | All-pair rolling table; compact descriptors; raw/adjusted views; dated case studies with counterexamples |
| Forecasting teammate, to agree | Matched experiments and robustness | Four feature ablations, baselines, common scoring index, noise/window/date sensitivities and per-currency errors |
| Shared review | Literature comparison and specification freeze | One paper-comparison table; one committed experiment specification and run manifest |

Execute in this order:

1. **Inventory and specification:** reconcile the separate predictor study; choose feature definitions, sample views, timing rules, primary horizon and experiment grid. Record these before looking at the new results.
2. **BIS-only prototype:** calculate the rolling relationship table, compact descriptors and policy divergence. Reproduce the existing baseline on the same scoring rows.
3. **Dated arrangements:** complete the regime/anchor ledger for the selected currencies; annotate historical relationships and build the forecast-eligible arrangement features.
4. **First decision:** run A–D and all prespecified short-window sensitivities. Decide from validation whether there is enough evidence to carry the new blocks into the compact neural comparison.
5. **Additional events and final comparison:** add intervention/IMF/debt only where coverage and dates are usable; run matched ablations, neural comparisons and the remaining robustness grid. Report both helpful and unhelpful blocks.

Suggested implementation locations are extensions to the existing workflow, not a second parallel pipeline: derive BIS features in `data/panel.py`; add a small source/event loader and rolling-relationship module; join their outputs and masks through `data/neural_dataset.py`; keep reporting in `evaluation/`. The current `features/predictors.py` is a placeholder, so it should not be mistaken for the implemented canonical feature layer. Notebooks should call the reusable modules rather than own data transformations.

Proposed deliverables: `config/clustering_exogenous.json` for frozen run choices; a source/event registry with provenance; `rolling_correlations.parquet` with pair counts and method/window metadata; `currency_context.parquet` with local/global scope and availability masks; a concise correlation/event timeline; a same-sample forecast and robustness table; and one literature-comparison page. Generated artifacts belong with existing ignored report outputs; commit code, definitions and source documentation according to the repository's current conventions.

The implementation's key checks should verify that appending future observations cannot change earlier dynamic features, public dates govern event joins, carry flags do not create false policy actions, return windows reset at gaps, group channels remain fixed, preprocessing uses training data only, and all compared forecasts score the same mature targets. Those checks should accompany implementation; no model tests or pipeline changes are part of this planning document.

The first milestone is a reproducible answer to **whether a small, economically interpretable context block adds forecasting information beyond the existing FX/macro panel**. A convincing historical explanation remains useful even if its feature fails that test; report the two outcomes separately.
