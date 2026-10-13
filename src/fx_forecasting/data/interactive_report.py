"""HTML presentation for the forecast-oriented BIS exploration."""

from __future__ import annotations

import html
import pandas as pd
from fx_forecasting.data.report_theme import render_document


def render_html(figures,banks,tables,meta,source,features,windows,plotly_js):
    def esc(value):
        return html.escape(str(value),quote=True)

    def plot(key, bank=None, selection=None):
        if bank:
            return f'<div class="chart" id="{key}" data-bank="{bank}" data-selection="{selection}"></div>'
        return f'<div class="chart" id="{key}" data-figure="{key}"></div>'

    def table(frame, columns=None, limit=None):
        frame=frame if columns is None else frame[columns]
        frame=frame if limit is None else frame.head(limit)
        frame=frame.rename(columns=lambda col:col.replace("_"," ").capitalize())
        return '<div class="table-scroll">'+frame.to_html(index=False,border=0,
            float_format=lambda v:f"{v:,.3f}",na_rep="—",escape=True,classes="data-table")+'</div>'

    def download(name,label=None):
        return f'<a class="download" href="interactive/{name}.csv" download>{esc(label or name.replace("_"," "))} ↗ CSV</a>'

    recent=tables["currency_metrics"].query("window == 'later'")
    ac=recent.lag1_autocorrelation.abs().median()
    flags=tables["peg_review_flags"]
    peg_n=int(flags.review_flag.sum())
    pre=tables["predictive_correlations"].query("horizon == 1 and method == 'spearman' and window == 'train'").copy()
    ranking=pre.assign(absolute=pre.correlation.abs()).groupby("feature").absolute.median().sort_values(ascending=False)
    best=ranking.index[0]
    correlations=tables["predictive_correlations"]
    def period_medians(feature,horizon):
        pair=correlations.query("feature == @feature and horizon == @horizon and method == 'spearman'").pivot(
            index="area",columns="window",values="correlation")[["train","later"]].dropna()
        return pair.train.median(),pair.later.median(),len(pair)
    momentum_pre,momentum_later,momentum_n=period_medians("fx_1m",1)
    credit1_pre,credit1_later,credit1_n=period_medians("credit_growth",1)
    credit12_pre,credit12_later,credit12_n=period_medians("credit_growth",12)
    baseline=tables["baseline_metrics"].query("horizon == 1")
    positive=int(baseline.mean_r2_os.gt(0).sum()); evaluable=int(baseline.mean_r2_os.notna().sum())
    pc=figures["pca"]["layout"]["meta"]
    opts=''.join(f'<option value="{esc(a)}" {"selected" if a == "BR" else ""}>{esc(row["Reference area"])} · {esc(row.CURRENCY)}</option>' for a,row in meta.iterrows())
    window_opts=''.join(f'<option value="{esc(k)}">{esc(v)}</option>' for k,v in windows.items())
    metric_columns=["economy","currency","n","annual_log_drift","annual_volatility","skewness","excess_kurtosis","lag1_autocorrelation","p01","p99"]
    snapshot=', '.join(sorted({m["downloaded_at_utc"][:10] for m in source["downloads"].values()})) or "Unrecorded"
    all_baselines=tables["baseline_metrics"]
    horizon_summary=[]
    for h in (1,3,12):
        valid=all_baselines.loc[all_baselines.horizon.eq(h)&all_baselines.mean_r2_os.notna()]
        horizon_summary.append({"Horizon":f"{h} month{'s' if h>1 else ''}",
            "Evaluable currencies":len(valid),"Median RW RMSE (log pp)":valid.rw_rmse.median(),
            "Historical mean wins":int(valid.mean_r2_os.gt(0).sum()),
            "Median mean-forecast R² (%)":valid.mean_r2_os.median()})
    horizon_summary=pd.DataFrame(horizon_summary)
    usable=tables["core_feature_counts"].copy()
    usable.index=usable.index.astype(str)
    key_months=[month for month in ("2010-01","2025-12","2026-07","2026-08") if month in usable.index]
    usable_summary=usable.loc[key_months].rename_axis("Origin").reset_index().rename(
        columns={"core":"Complete core features","core_and_credit":"Complete core + credit"})
    content=f'''
<header><div class="eyebrow">COLUMBIA · DEEP LEARNING · RESEARCH EXPLORATION</div>
<h1>Emerging-market FX.<br><span>Where might the signal be?</span></h1>
<p class="lede">A forecast-oriented exploration of BIS exchange rates, macro fundamentals and dollar credit, guided by Filippou, Rapach, Taylor &amp; Zhou.</p>
<div class="meta">January 1994–August 2026 · BIS snapshot {esc(snapshot)} · interactive Plotly · works offline</div>
<div class="report-switch"><a href="data_quality.html">01 / Data quality</a><a href="data_eda.html" aria-current="page">02 / Data EDA</a></div></header>
<nav aria-label="Report sections"><a href="#findings">Findings</a><a href="#sample">Sample &amp; quality notes</a><a href="#behavior">Behaviour</a><a href="#common">Co-movement</a><a href="#signals">Predictive signals</a><a href="#credit-section">Credit &amp; timing</a><a href="#benchmarks">Benchmarks</a><a href="#method">Methods</a><a href="data_quality.html">Data quality ↗</a></nav>
<main>
<section id="findings"><div class="eyebrow">01 / THE RESEARCH QUESTION</div><h2>Can a shared network improve on no change?</h2>
<p>The project compares a shared neural network, a network with currency embeddings, and separate currency networks at 1-, 3-, and 12-month horizons. Dollar credit is an additional hypothesis. This report describes the data and potential signals; forecast success must be established by expanding-window evaluation from January 2010.</p>
<div class="kpis"><div><strong>{len(meta)}</strong><span>provisional distinct currencies</span></div><div><strong>{pc['pc1_variance_pct']:.1f}%</strong><span>return variation explained by PC1²</span></div><div><strong>{recent.annual_volatility.median():.1f}%</strong><span>median annualised FX volatility¹</span></div><div><strong>{positive}/{evaluable}</strong><span>historical mean beats random walk · 1m</span></div></div>
<div class="insights"><article><h3>Pooling has a rationale</h3><p>The first principal component explains <b>{pc['pc1_variance_pct']:.1f}%</b> of standardised monthly return variation in a balanced {pc['n_currencies']}-currency subset. Common movement supports testing shared representations; it also reduces the number of independent observations in a pooled panel.</p></article>
<article><h3>Short-horizon persistence changes sign</h3><p>Across {momentum_n} currencies estimable in both periods, median next-month momentum correlation shifts from <b>{momentum_pre:.3f}</b> before 2010 to <b>{momentum_later:.3f}</b> later. Median absolute lag-1 Pearson correlation is only <b>{ac:.3f}</b> from 2010. Historical momentum may transfer poorly. <a href="#signals">Inspect stability ↗</a></p></article>
<article><h3>Credit deserves a longer-horizon test</h3><p>Median credit-growth correlation with the 12m target is <b>{credit12_pre:.3f}</b> before 2010 and <b>{credit12_later:.3f}</b> later, versus {credit1_pre:.3f} and {credit1_later:.3f} at 1m. Overlapping outcomes and repeated quarterly values limit independent evidence. <a href="#credit-section">Inspect timing ↗</a></p></article>
<article><h3>Set a demanding benchmark</h3><p>An expanding historical mean beats zero change for <b>{positive}/{evaluable}</b> evaluable currencies at 1m; median R² against random walk is <b>{baseline.mean_r2_os.median():.2f}%</b>. Neural networks should be compared with regularised linear models as well as these basic benchmarks.</p></article></div>
<p class="caption">¹ Monthly changes from January 2010. ² Descriptive PCA on {pc['n_currencies']} currencies and {pc['n_months']} complete months. Percentage values are log changes, not simple investment returns. Correlations are descriptive rather than evidence of forecast gains.</p>
<div class="verdict"><strong>Research takeaway:</strong> shared structure and longer-horizon credit associations justify controlled experiments. Time instability and the historical mean’s weak benchmark performance make regularisation, matched samples and expanding-window validation central. Neural-network superiority remains untested.</div>
<aside><b>What the article contributes.</b> The July 2, 2021 working paper examines monthly developed-market exchange rates with country characteristics, global variables and interactions. It emphasises pooling and strong regularisation to control overfitting, and evaluates forecasts against no change. That motivates the co-movement, interaction and benchmark diagnostics here. The BIS panel, policy-rate proxies, emerging-market focus, dollar-credit feature and longer horizons are project extensions, not a replication. <a href="{esc(source['paper_url'])}" target="_blank" rel="noopener">Read the paper ↗</a></aside></section>

<section id="sample"><div class="eyebrow">02 / SAMPLE &amp; QUALITY REFERENCES</div><h2>Sample scope and the quality checks that matter here.</h2>
<p>The source review includes 192 FX reference areas. For currency comparisons, this report uses non-USD currencies with broad REER coverage, retains the euro-area EUR representative and omits national EUR backcasts. EUR observations before January 1999 are masked. The resulting {len(meta)} currencies are an exploratory universe; the requested 39-currency roster and emerging/advanced classification are not defined in the repository.</p>
<aside><b>Quick quality reference.</b> The earlier review found no duplicate source records. Policy rates have the largest missing share (23.45%), including discontinued national histories and Japan’s internal gaps; US CPI is missing October 2025. Credit ends at 2026Q1. Missing values and extreme moves remain unfilled and untrimmed. <a href="data_quality.html#audit">Duplicate checks ↗</a> · <a href="data_quality.html#coverage">Coverage and exact gap dates ↗</a> · <a href="data_quality.html#mapping">Currency mappings ↗</a></aside>
<p><b>{peg_n}</b> currencies have a 60-month window with at least 95% near-zero FX changes. These are period-review flags, not established peg dates or automatic exclusions; the final study must define regime treatment explicitly.</p>
<div class="links">{download('sample_mapping')}<a class="download" href="data_quality.html">Full earlier data-quality report ↗</a></div>
<details><summary>Inspect the currency mapping and exclusions</summary>{table(tables['sample_mapping'],['REF_AREA','Reference area','CURRENCY','keep','reason'])}</details></section>

<section id="behavior"><div class="eyebrow">03 / TARGET BEHAVIOUR</div><h2>Drift, tails and volatility vary across currencies.</h2>
<p>FX is quoted as local currency units per US dollar. A positive change means <b>local-currency depreciation</b>. Future targets are 100 × [log(S at t+h) − log(S at t)]. Explore an individual currency below; its histogram and autocorrelation use all available months, while the indexed path starts in 2010.</p>
<div class="controls"><label>Currency <select id="currency-select">{opts}</select></label><span>Hover for values · drag to zoom · double-click to reset · camera icon to export</span></div>
{plot('currency-chart','currency','BR')}
<div class="two-col">{plot('dispersion')}{plot('tails')}</div>
<p>The dispersion plot shows why identical raw scaling or identical loss weights may favour high-volatility currencies. Heavy tails make RMSE sensitive to a few months; report MAE and currency-level results alongside RMSE. Annualised volatility is a descriptive √12 scaling and does not establish independent monthly returns.</p>
<details><summary>Currency statistics · from January 2010</summary>{table(recent.sort_values('annual_volatility',ascending=False),metric_columns)}</details>
<div class="links">{download('currency_metrics')}</div>
<h3>Large changes require source and regime review.</h3><p>The largest absolute monthly changes are retained. They can reflect market stress, denomination changes or source breaks; this report does not identify their causes. See the <a href="data_quality.html#breaks">earlier source-universe large-change flags ↗</a>.</p>
{table(tables['extreme_moves'],['period','economy','area','monthly_log_change'],10)}{download('extreme_moves')}
{plot('pegs')}{download('peg_review_flags')}</section>

<section id="common"><div class="eyebrow">04 / SHARED STRUCTURE</div><h2>Currencies share global exposure, with different loadings.</h2>
<p>This heatmap compares contemporaneous monthly log changes from January 2010, using at least 60 paired observations. It describes co-movement, not forecasting power. A shared USD numeraire can induce common movement. Repeated EUR histories are removed; remaining pegs and managed regimes are retained.</p>
{plot('comovement')}{plot('pca')}
<p class="caption">PCA uses full-period standardisation for descriptive analysis only: currencies with at least 160 post-2010 observations and monthly standard deviation above 0.05 log percentage points, then complete months across that subset. {pc['n_months']} balanced months enter. Component signs are arbitrary. This filtering is not a proposed model-sample rule.</p>
{plot('risk')}<p>The global FX risk proxy is the cross-currency median of rolling 12-month monthly-return standard deviations. The median limits the influence of a single currency’s extreme move; changing coverage can still affect it. It differs from the paper’s daily-data volatility measure. Median rolling USD correlation uses complete 36-month pairs.</p></section>

<section id="signals"><div class="eyebrow">05 / POTENTIAL PREDICTORS</div><h2>Relate predictors at t to changes after t.</h2>
<p>These are per-currency correlations with <b>future</b> targets. Default results use pre-2010 origins whose targets have realised by December 2009. Spearman correlations reduce sensitivity to magnitudes; Pearson correlations retain linear-scale information. Empty cells mean fewer than 36 pairs or a nearly constant series. Colours saturate at ±0.6; hover shows the exact value and sample size.</p>
<div class="controls"><label>Horizon <select id="signal-horizon"><option value="1">1 month</option><option value="3">3 months</option><option value="12">12 months</option></select></label><label>Period <select id="signal-window">{window_opts}</select></label><label>Correlation <select id="signal-method"><option value="spearman">Spearman rank</option><option value="pearson">Pearson linear</option></select></label></div>
{plot('signal-chart','correlation','1|train|spearman')}
<div class="two-col">{plot('stability')}{plot('regimes')}</div>
<p><b>Persistence weakens in the later period.</b> Across {momentum_n} currencies with estimable correlations in both periods, the median signed relationship between the latest FX move and the next-month change shifts from <b>{momentum_pre:.3f}</b> before 2010 to <b>{momentum_later:.3f}</b> from 2010. A momentum relationship estimated on early history may therefore transfer poorly. The median absolute chart measures strength rather than direction; individual-country results can differ from both summaries.</p>
<p>The stress split uses a fixed pre-2010 risk threshold of <b>{source['risk_threshold_pre2010_p75']:.3f}</b> log percentage points (75th percentile). The regime bars are median signed next-month correlations across currencies, with 36 pairs required within each regime. Changing signs or strengths motivate interaction tests and currency embeddings. Regime summaries may use different valid countries; they are exploratory comparisons.</p>
<aside>These correlations do not establish causality or out-of-sample predictability. At 3m and 12m, consecutive targets overlap. Global features repeat across currencies, so pooled row counts are not independent sample sizes. The later period is visible for diagnostics and must not be used to tune a claimed untouched test. Compare credit and interaction models on common samples.</aside>
<div class="links">{download('predictive_correlations')}{download('regime_correlations')}</div></section>

<section id="credit-section"><div class="eyebrow">06 / CREDIT &amp; AVAILABILITY</div><h2>Measure the credit hypothesis without confusing trends with signal.</h2>
<p>The stock of outside-US dollar credit is quarterly and measured in millions of USD. The chart converts it to trillions. The correlation feature uses year-on-year log growth and a two-quarter lag, helping distinguish credit dynamics from its long-run trend. Within each origin quarter, the lagged value is held constant.</p>
{plot('credit')}{plot('usable')}
<p><b>Credit associations are stronger at the longer horizon.</b> On {credit12_n} currencies estimable in both periods, the median signed credit-growth correlation with the 12m target is <b>{credit12_pre:.3f}</b> before 2010 and <b>{credit12_later:.3f}</b> from 2010. Corresponding 1m medians are {credit1_pre:.3f} and {credit1_later:.3f}. This supports testing a horizon-specific credit specification, but repeated quarterly values and overlapping annual outcomes provide fewer independent observations than the monthly pair counts suggest.</p>
<p>For an August 2026 origin (Q3), the credit feature uses 2026Q1. A July 2026 origin uses the same Q1 observation. The 60-month REER history, 12-month inflation comparison, conservative macro lags and credit growth warm-up all shorten usable history. The missing 2026Q2 credit stock is not required by the lag rule within this study window.</p>
<h3>A single global-reference gap can remove an entire forecast origin.</h3>{table(usable_summary)}
<p>The October 2025 US CPI gap propagates to December 2025 inflation differentials under the two-month lag: <b>no currency has all core features at that origin</b>. A later feature pipeline must specify how that origin is handled and apply the same rule across every model. <a href="data_quality.html#coverage">Inspect the original missing-value evidence ↗</a></p>
<p class="caption">Core coverage requires latest FX change, 3m momentum, 12m volatility, REER gap, policy differential, inflation differential and USD-index change. Horizon lines additionally require an observed future target; core + credit requires credit growth but not a target. These are descriptive counts, not train/test sample sizes.</p>
<aside>Latest revised exports are not real-time vintages. Policy differential is lagged 1m and inflation differential 2m as explicit availability proxies; actual release calendars still need validation. REER and NEER reference months do not prove availability at month end. Credit’s two-quarter alignment is a convention, not release-date evidence.</aside>
<div class="links">{download('feature_availability')}{download('core_feature_counts')}{download('exploratory_panel')}</div></section>

<section id="benchmarks"><div class="eyebrow">07 / ERROR SCALE &amp; SIMPLE FORECASTS</div><h2>Know how much error the random walk leaves to explain.</h2>
<p>The no-change forecast is zero at every horizon; its RMSE is √mean(y²). The historical-mean comparator uses at least 60 realised labels per currency and expands through each forecast origin. At origin t for horizon h, its most recent usable label is y at t−h, whose outcome ends at t. Results start in January 2010 and compare forecasts on matched observations.</p>
<div class="controls"><label>Horizon <select id="baseline-horizon"><option value="1">1 month</option><option value="3">3 months</option><option value="12">12 months</option></select></label></div>
{table(horizon_summary)}<p class="caption">Currency-level medians use finite benchmark ratios. Historical-mean wins count currencies with positive R²; they do not imply statistical significance. Horizon samples differ as future outcomes become unavailable near the end of the data.</p>
{plot('baseline-chart','baseline','1')}{plot('loss')}
<p>R² = 100 × [1 − MSPE(model)/MSPE(random walk)]. Positive values mean lower squared forecast error. The cumulative plot rises when the historical mean saves squared error and falls when it loses; it is not an investment return. Pegged currencies may have near-zero denominators, making ratios fragile or undefined. Horizon RMSE is cumulative and is not annualised.</p>
<details><summary>Forecast metrics, directional scores and paired sample dates</summary>{table(tables['baseline_metrics'])}</details>
<p class="caption">Directional accuracy for the historical mean uses sign agreement on nonzero forecasts and targets; its denominator is reported separately. A zero-change forecast has no directional call, so its directional accuracy is undefined. No significance tests or neural-network performance are claimed.</p>{download('baseline_metrics')}</section>

<section id="method"><div class="eyebrow">08 / REPRODUCIBILITY &amp; NEXT EXPERIMENTS</div><h2>Turn these diagnostics into a fair model comparison.</h2>
<ol><li>Confirm the 39-currency roster, emerging/advanced groups, and time-varying peg exclusions. Review source breaks and historical currency representations.</li><li>Validate release timing and missing-data handling. Use training-window scaling and transformations; fit no learned preprocessing on later observations.</li><li>Compare random walk, AR(1), Taylor-style and pooled elastic-net benchmarks with all three network designs on identical origins and inputs.</li><li>Test credit growth, alternative lagged credit stock specifications, and fundamental × global-condition interactions on common samples. Include a linear interaction baseline to distinguish interaction gains from neural nonlinearity.</li><li>Use expanding windows from 2010 with training labels realised before each origin. Purge overlapping targets in validation; report per-currency RMSE, MAE, R², directional accuracy and EM/advanced summaries.</li><li>Assess uncertainty with methods that account for overlapping horizons and shared shocks. Use Clark–West only for suitable nested forecast comparisons; do not treat these descriptive correlations as significance tests.</li></ol>
<details open><summary>Exact exploratory feature definitions</summary><div class="table-scroll"><table class="data-table"><thead><tr><th>Feature</th><th>Definition at forecast origin t</th></tr></thead><tbody>
<tr><td>Latest FX / momentum</td><td>100 × log(S_t / S_(t−1)); 100 × log(S_t / S_(t−3)).</td></tr>
<tr><td>Local volatility</td><td>Standard deviation of 12 adjacent monthly log changes, including t; no annualisation in the feature.</td></tr>
<tr><td>REER gap</td><td>100 × [log(REER_t) − mean(log(REER_(t−60 … t−1)))]; requires 60 observed calendar months.</td></tr>
<tr><td>Policy differential</td><td>Domestic minus US annual policy rate, in percentage points, observed at t−1. No yield or forward-rate substitution.</td></tr>
<tr><td>Inflation differential</td><td>Domestic minus US 12-month CPI log inflation, evaluated at t−2. CPI index levels are not compared across countries.</td></tr>
<tr><td>Dollar index</td><td>100 × monthly log change of BIS broad US NEER. This is the BIS index, not ICE DXY. Rising NEER means USD appreciation.</td></tr>
<tr><td>Dollar credit</td><td>100 × log(credit_q / credit_(q−4)), where q is the quarter containing t minus two quarters.</td></tr>
<tr><td>Global risk / interactions</td><td>Cross-currency median of rolling 12m monthly-return standard deviations; multiply by policy or inflation differential. No centering or learned scaling is applied here.</td></tr>
</tbody></table></div></details>
<p>All log transforms require positive values. Negative policy rates remain valid. Targets require positive FX at both calendar endpoints; intermediate missing FX cells do not invalidate an endpoint target. Adjacent monthly returns and rolling volatility do require their input observations. FX data are end-of-month; EER indices are monthly averages.</p>
<p><b>Reproduce:</b> <code>python scripts/run_interactive_eda.py</code> after installing <code>python -m pip install -e .</code>. The script reads local raw BIS files, uses existing selection rules, verifies download hashes when recorded, and writes this page plus CSVs. The Plotly library is embedded; no network is needed to view charts. CSV downloads require the accompanying report folder.</p>
<div class="links"><a class="download" href="interactive/analysis_manifest.json" download>Source hashes &amp; conventions ↗ JSON</a><a class="download" href="interactive/README.md">Written findings ↗</a><a class="download" href="data_quality.html">Earlier data-quality review ↗</a></div>
<p class="sources">Sources: <a href="{esc(source['paper_url'])}">Filippou et al. (July 2021 working paper)</a>; <a href="https://data.bis.org/bulkdownload">BIS bulk data</a>; <a href="https://data.bis.org/topics/XRU">FX</a>; <a href="https://data.bis.org/topics/EER">EER</a>; <a href="https://data.bis.org/topics/CBPOL">Policy</a>; <a href="https://data.bis.org/topics/CPI">CPI</a>; <a href="https://data.bis.org/topics/GLI">Global liquidity</a>. The repository’s JFQA citation is not used to label this working-paper version as a published 2021 article.</p></section>
</main><footer>EMERGING-MARKET FX NEURAL NETWORK · DESCRIPTIVE RESEARCH REPORT · {esc(source['plotly_version'])} Plotly</footer>
'''
    eda_figures={key:value for key,value in figures.items() if key not in {"coverage","missing"}}
    return render_document("Emerging-market FX · Data EDA",content,eda_figures,banks,plotly_js)
