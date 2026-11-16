**Source audit: the 35-predictor feature-selection study**

16 November 2026. **The original results report is located; the numerical dataset and generating notebook are not yet located.** Twenty feature descriptions can be recovered. The other fifteen cannot be reconstructed from this report without guessing.

The source is `rfe_selection.pdf`, found in the local Downloads directory: four pages, titled *Feature selection with RFE and RFECV, then Ridge and neural networks*. Its numbers and feature names match the study cited in the local 11 November progress report. `rfe_selection (1).pdf` is byte-identical, rather than a second version. SHA-256: `08e72b8f946ae6f55cb370be4ed48769671b01929f1f83a24332498e3cd1b387`.

A local archival copy and extracted text are saved under `reports/clustering/context/research_sources/`. The [machine-readable inventory](../../config/feature_study_inventory.json) records the twenty descriptions, selection counts, scope, local availability and unresolved reproduction requirements. The report copy and extraction remain ignored local artifacts; share the PDF with this note when handing off outside this checkout.

**Recovered feature descriptions**

These are the twenty ranked rows on page 2, not the full candidate set. EWMA denotes an exponentially weighted moving average. Reported EWMA half-life is two months for the rows below. A reported publication delay is a recipe description, not verified historical availability.

| Rank | Predictor | Reported transformation/timing | What is available locally |
| --- | --- | --- | --- |
| 1 | China imports | Three-month growth; two-month delay; EWMA | External raw series and provider/series ID missing |
| 2 | US goods imports | Three-month growth; two-month delay; EWMA | External raw series and provider/series ID missing |
| 3 | German exports | Three-month growth; two-month delay; EWMA | External raw series and provider/series ID missing |
| 4 | Shanghai stock index | Monthly return; EWMA | Exact index, price convention and source missing |
| 5 | Emerging-market equities, EEM | Monthly return; EWMA | Ticker named; source snapshot and return convention missing |
| 6 | US ten-year yield | Monthly change; EWMA | Source ID, yield units and month aggregation missing |
| 7 | Monetary-policy uncertainty | Log index, previous month | Exact index/source and release dates missing |
| 8 | China CPI | Monthly change from latest publication known at previous month-end | BIS CPI ingredients exist; exact availability join needs code |
| 9 | S&P 500 | Monthly return including current month; EWMA | Price/total-return definition and raw source missing |
| 10 | Gold | Monthly percentage change; EWMA | Instrument, quotation and raw source missing |
| 11 | Dollar index | Monthly return; EWMA | Exact index missing; do not substitute BIS USD NEER silently |
| 12 | Economic-policy uncertainty | Log index, previous month | Exact index/source and release dates missing |
| 13 | Geopolitical risk | Log index, previous month | Exact index/source and release dates missing |
| 14 | Domestic minus US policy rate | Level, previous month | Existing canonical `rate_diff`, with documented structural gaps |
| 15 | Oil | Monthly percentage change; EWMA | Old notebook downloads `CL=F`, but this report's instrument and raw snapshot remain unverified |
| 16 | VIX | Monthly change; EWMA | Raw source and month aggregation missing |
| 17 | EUR move minus CHF move | Current month | BIS bilateral FX ingredients exist; confirm quotation, units and sign |
| 18 | Copper futures | Current-month percentage change | Contract, roll/adjustment convention and raw source missing |
| 19 | EUR move against USD | Current month | BIS EUR/USD ingredients exist; confirm quotation and units |
| 20 | Own FX move | Previous month | BIS FX ingredients exist; apply lag within currency and respect gaps |

Ranks 8, 14, 17, 19 and 20 have raw BIS ingredients in the repository. This does not establish exact reproduction: the report does not provide the generating expressions, all unit conventions or the missing-data procedure. Oil's live download in `notebooks/fx_cleaned.ipynb`, cell 22, is a related earlier experiment, not proof that the 35-feature study used the same data.

**What the report establishes about the experiment**

Page 1 describes 39 currencies, a three-month target, eight sliding calendar windows from 2000–2019 through 2007–2026, chronological 80/10/10 splits with three-month gaps, random-forest RFE/RFECV, and pooled Ridge/neural models. Page 2 names only the top twenty predictors. Page 3 uses fixed ten- and nineteen-feature lists formed across those windows; page 4 adds the three existing currency groups. It reports R² relative to a currency's training mean, whereas the new context prototype reports MAE/RMSE and zero-change comparisons. The numeric scores are not directly comparable across these designs.

The report itself notes that later training windows overlap earlier validation/test periods when constructing fixed lists. Keep those lists as retrospective research evidence. Any new forecasting comparison must select features within its own training/validation procedure. The PDF does not provide enough detail to recover exact dates, imputation, scalers, random seeds, or all transformations.

**Why feature nature matters for clustering**

Eighteen of the twenty described predictors are shared global time series. Only the domestic–US policy differential and own lagged FX move vary across currencies at a given date. Global series can describe common conditions and support forecasts, but cannot distinguish currencies in a same-month cross-sectional feature clustering by themselves. Country-specific sensitivities or measured exposures are needed for that purpose. Do not infer those exposures from unverified country labels or use full-history sensitivity estimates in a historical forecast.

This makes a small dated arrangement/anchor block useful alongside the existing global feature candidates. It also means that finding the original global data and their release conventions matters more than immediately adding many new model inputs.

**Search completed and remaining artifact**

Checked the working repository and Git history, the existing progress-report source archive, relevant Deep Learning folders, candidate Downloads/Desktop files and archives, narrowly targeted connected Drive/Gmail searches, and GitHub default-branch code search for RFECV. These checks located the original four-page report and its identical duplicate, but no 35-column feature matrix or generating RFE notebook. Connected search results are bounded by the accessible accounts and refs; this is not a claim that the teammate never saved the data elsewhere.

During the 16 November source inspection, ran `git fetch origin main`, refreshed all origin refs, created a separate detached checkout of `origin/main`, and ran `git pull --ff-only origin main` there. The pull reported **Already up to date**. The configured remote is `https://github.com/alexandre-sdd/emerging-market-fx-neural-network.git`; at that inspection, main's tip was **`6db74028b2a2f90265936610bb0be95528019818`**, authored/committed on **28 October 2026**, message **Add files via upload**. This identifies the inspected source snapshot before publication of the new context workflow.

That main tree contains the 25-cell `notebooks/fx_cleaned.ipynb`. Its SHA-256 is `55c1384936c71250cc5042b89fed1eb757e7ca41baefba25af36c5af7d2e7fff`, identical to the working copy. Its code contains no RFE/RFECV/random-forest selector, Shanghai series, VIX or copper inputs. No newer study file appears in the other fetched origin refs. The current working branch already contains that main commit; no merge was necessary, and its uncommitted files were preserved. The remaining check is the teammate's actual repository URL or commit hash, rather than another pull of the same remote tip.

A further connected GitHub check reports **zero forks/network repositories** for the project and finds no pull requests. No accessible contributor fork containing the study was found. The user does not know another source repository or commit. Consequently the remaining dependency is a source push, notebook/data bundle, or link from the study's author; repeating the same pull cannot recover files absent from that remote.

The older `fx_cleaned.ipynb` contains BIS transformations, driver experiments and a live oil download. It does not contain the report's RFECV procedure, trade series, equity series or uncertainty-index inputs. The original November progress-report ZIP contains report sources and figures, not the underlying feature-study data.

To complete reproduction, obtain:

- The notebook/script that constructs all 35 candidate predictors, with its exact input paths or download calls.
- Saved raw series or a dated, unscaled feature matrix, plus provider names and series/ticker IDs.
- The remaining fifteen feature names and exact growth/return/yield formulas, quote directions, EWMA settings and lag order.
- Coverage, release dates/vintages, missing-data/scaling rules, target definitions and window boundaries.
- Per-window selected features, seeds and fitted-model settings needed to match the report.

A concise request the team can copy: “Please share the notebook/script and saved inputs behind `rfe_selection.pdf`, including all 35 feature definitions and provider/series IDs. The PDF names only the top 20. We also need publication lags, EWMA settings, missing-data rules and per-window selections to connect the study to the common forecasting pipeline.” No message has been sent.
