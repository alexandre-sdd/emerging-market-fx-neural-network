**CHF/EUR: three observations for the group**

16 November 2026. Monthly BIS observations, existing sample rules, trailing Pearson return correlations updated monthly. This is a descriptive case study supporting feature design; it does not estimate a causal policy effect or report a new forecast experiment.

Open the [annotated interactive case study](../reports/clustering/context/chf_eur/case_study.html). Reproduce its evidence with `.venv/bin/python scripts/analyze_chf_eur_episode.py` after the existing context export has been built. The script verifies every saved nine-month CHF/EUR correlation against a direct calculation and checks source/export hashes.

**1. The floor is a plausible explanation for strong co-movement.**

For endpoints June 2012–December 2014, mean nine-month raw correlation is **0.945**, ranging from 0.861 to 0.995. The median-adjusted mean is 0.832. The SNB introduced its CHF 1.20 per EUR minimum on **6 September 2011** ([contemporary SNB statement](https://www.snb.ch/en/publications/communication/speeches/2011/ref_20110906_pmh), [ECB confirmation](https://www.ecb.europa.eu/press/pr/date/2011/html/pr110906.de.html)).

The mechanism is concrete. The dataset quotes CHF per USD and EUR per USD. Their ratio is CHF per EUR; the difference between their monthly log returns is the CHF/EUR log return. If the cross rate stays close to a binding floor, CHF and EUR tend to move together against USD. This is a plausible explanation of the observed correlation, rather than independent evidence of predictive power.

It was a **one-sided floor**, allowing CHF per EUR to rise above 1.20, rather than a fixed parity. In the local month-end data, the 2013 cross rate ranges from 1.2195 to 1.2406. Strong but imperfect return correlation is therefore compatible with the arrangement. Floor-period endpoints begin in June 2012 so all nine return months come after the September 2011 introduction month; earlier rolling windows mix regimes.

**2. The January 2015 break matches the event, but its apparent duration depends heavily on window memory.**

The SNB discontinued the floor on **15 January 2015** ([press release](https://www.snb.ch/en/publications/communication/press-releases/2015/pre_20150115)). Raw nine-month correlation falls from **0.983 in December 2014 to −0.335 in January 2015**. The January month-end log changes are **−6.724** for CHF per USD and **+7.134** for EUR per USD, in percent log units. These opposite-signed monthly moves, including the policy episode, strongly affect the small rolling sample. They are monthly returns, not the announcement-day move.

Every nine-month window ending January–September 2015 contains January's observation. Their mean raw correlation is −0.111. September's estimate is **−0.065**; October's is **0.862**, when January is no longer in the window. A retrospective influence check removes January from September's existing window and obtains **0.860 on the remaining eight observations**. This supports a substantial window-composition effect; the rebound alone is not evidence of a new October policy regime.

The actual nine-month estimator and all forecasting inputs retain January. The eight-observation calculation is a labeled diagnostic only. A shorter/longer window drops January in a different month; all five existing window lengths are exported in [the window-exit table](../reports/clustering/context/chf_eur/window_exits.csv). Also, the floor's removal coincided with other policy actions, so this comparison does not isolate a causal floor effect.

**3. High correlation can return without the floor: a counterexample to “high correlation means pegging.”**

In **2021**, mean nine-month raw correlation is **0.926**, with all twelve endpoints between 0.906 and 0.948. Median-adjusted correlation averages 0.676. This is high co-movement several years after the floor's discontinuation. The [SNB's historical definition](https://www.snb.ch/en/services-events/digital-services/glossary) dates that floor to September 2011–January 2015; its [December 2021 policy assessment](https://www.snb.ch/public/asset/en/www-snb-ch/publications/communication/press-releases/2021/pre_20211216/publications0_en/pre_20211216.en.pdf) describes the negative policy rate and willingness to intervene as necessary.

An announced floor is therefore not necessary for high CHF/EUR return correlation. Common USD movements, shared regional conditions and other monetary influences are plausible contributors. This case study does not establish their separate contributions or claim an intervention occurred in a particular month. Median adjustment changes the dependence measure; it is not an identified causal USD shock.

**Evidence across the full selected episodes**

| Endpoint interval | Endpoints | Mean raw nine-month correlation | Mean median-adjusted correlation |
| --- | --- | --- | --- |
| January 2010–August 2011, before introduction | 20 | 0.701 | 0.100 |
| June 2012–December 2014, windows wholly after introduction month | 31 | 0.945 | 0.832 |
| January–September 2015, windows containing the January shock | 9 | −0.111 | −0.643 |
| October 2015–December 2016, after the shock leaves the nine-month window | 15 | 0.853 | 0.380 |
| January–December 2021, floor absent | 12 | 0.926 | 0.676 |

These are means of overlapping estimates, with unequal episode lengths. Endpoint counts are not independent sample sizes. The episodes were chosen for historical interpretation after inspecting the timeline; they are not pre-registered tests. The full monthly evidence, both return views and all window exits are exported alongside the figure.

**Feature-design implication**

Keep a dated **announced arrangement state** separate from the **trailing observed relationship**. A correlation flag cannot substitute for a peg/floor flag, and a low-correlation run may partly reflect one large return remaining in the window. Add an announced change/recent-event feature only from its public date, retain the continuous correlation and raw returns, and assess the added arrangement block on matched forecast observations. Do not remove January 2015 to improve the backtest.

Related records: [arrangement/event ledger](clustering_event_ledger.md), [machine-readable event registry](../config/currency_arrangement_events.json), [shared checklist](clustering_todo.md).
