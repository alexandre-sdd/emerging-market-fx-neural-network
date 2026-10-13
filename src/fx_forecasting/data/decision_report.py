"""Explain the selected research design and audit its FX-only sample implications.

Outputs are report artifacts, not a production panel or model inputs.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from fx_forecasting.config import PROCESSED_DIR, PROJECT_ROOT
from fx_forecasting.data.quality_report import chart_spec
from fx_forecasting.data.report_theme import render_document


def _table(headers, rows):
    def cells(values, tag):
        return ''.join(f'<{tag}>{html.escape(str(value))}</{tag}>' for value in values)
    return ('<div class="table-scroll"><table class="data-table decision-table"><thead><tr>'
            + cells(headers, 'th') + '</tr></thead><tbody>'
            + ''.join('<tr>' + cells(row, 'td') + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _eligible_levels(fx, choices, formal_china=False):
    eligible = fx.notna() & fx.gt(0)
    for currency in choices['currencies']:
        eligible.loc[eligible.index < pd.Period(currency['first_eligible_month'], 'M'), currency['area']] = False
    for interval in choices['excluded_month_intervals']:
        end = '2005-07' if formal_china and interval['area'] == 'CN' and interval['start'] == '1994-01' else interval['end']
        eligible.loc[interval['start']:end, interval['area']] = False
    return eligible


def _eligible_origins(levels, horizon, history=False):
    origins = levels.copy()
    for offset in range(1, horizon + 1):
        origins &= levels.shift(-offset, fill_value=False)
    if history:
        # Twelve consecutive monthly returns require thirteen levels.
        origins &= levels.rolling(13, min_periods=13).sum().eq(13)
    return origins


def build_decision_report(output_dir: Path | None = None) -> Path:
    """Build an offline third report from the canonical choices and saved evidence."""
    destination = Path(output_dir) if output_dir else PROJECT_ROOT / 'reports' / 'eda'
    assets = destination / 'decision_assets'
    assets.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / 'config' / 'research_sample.json'
    choices = json.loads(config_path.read_text())
    quality_root = PROJECT_ROOT / 'reports' / 'eda'
    summary = json.loads((quality_root / 'summary.json').read_text())
    if choices['evidence']['source_sha256'] != {key: value['sha256'] for key, value in summary['sources'].items()}:
        raise ValueError('Decision specification and quality evidence use different BIS snapshots')
    rows = choices['currencies']
    calendar = pd.period_range(**choices['monthly_window'], freq='M')
    inputs = {name: PROCESSED_DIR / f'eda_{name}.csv' for name in ('fx', 'usd_credit', 'cpi')}
    raw_fx = pd.read_csv(inputs['fx'], index_col=0)
    raw_fx.index = pd.PeriodIndex(raw_fx.index, freq='M')
    fx = raw_fx[[row['source_series']['fx'] for row in rows]].reindex(calendar)
    fx.columns = [row['area'] for row in rows]
    if len(rows) != 39 or fx.columns.duplicated().any():
        raise ValueError('Expected 39 unique selected currency representatives')
    observed = fx.notna() & fx.gt(0)
    launched = observed.copy()
    for row in rows:
        launched.loc[calendar < pd.Period(row['first_eligible_month'], 'M'), row['area']] = False
    levels = _eligible_levels(fx, choices)
    formal_levels = _eligible_levels(fx, choices, formal_china=True)
    credit = pd.read_csv(inputs['usd_credit'], index_col=0)
    credit.index = pd.PeriodIndex(credit.index, freq='Q')
    stock = credit[choices['macro_availability']['usd_credit_series']]
    lag_quarters = calendar.asfreq('Q') - choices['macro_availability']['usd_credit_quarter_lag']
    stock_valid = stock.notna() & stock.gt(0)
    growth_valid = stock_valid & stock_valid.shift(4, fill_value=False)
    credit_available = pd.Series(stock_valid.reindex(lag_quarters, fill_value=False).to_numpy(), index=calendar)
    growth_available = pd.Series(growth_valid.reindex(lag_quarters, fill_value=False).to_numpy(), index=calendar)

    figures, banks = {}, {'decision_eligibility': {}}
    monthly_records, totals = [], []
    for horizon in (1, 3, 12):
        for history in (False, True):
            eligible = _eligible_origins(levels, horizon, history)
            alternative = _eligible_origins(formal_levels, horizon, history)
            matched = eligible.mul(credit_available, axis=0)
            matched_growth = eligible.mul(growth_available, axis=0)
            counts = pd.DataFrame({
                'primary_fx_only': eligible.sum(axis=1),
                'formal_china_cutoff': alternative.sum(axis=1),
                'stock_available': matched.sum(axis=1),
                'growth_available': matched_growth.sum(axis=1),
            })
            for period, values in counts.iterrows():
                monthly_records.append({'month': str(period), 'horizon': horizon,
                                        'history': '12_returns' if history else 'path_only', **values.to_dict()})
            totals.append({'horizon': horizon, 'history': '12_returns' if history else 'path_only',
                           **counts.sum().to_dict(),
                           'primary_fx_only_from_2010': int(counts.loc['2010-01':, 'primary_fx_only'].sum())})
            fig = go.Figure()
            for column, label, color, dash in (
                ('primary_fx_only', 'Selected regime rules', '#147d92', 'solid'),
                ('formal_china_cutoff', 'China: July 2005 alternative', '#b48c2e', 'dot'),
                ('stock_available', 'Selected + observed lagged stock', '#765ba7', 'solid'),
                ('growth_available', 'Selected + observed lagged growth', '#c35d38', 'dash'),
            ):
                fig.add_scatter(x=calendar.to_timestamp(), y=counts[column], name=label,
                                mode='lines', line=dict(color=color, dash=dash),
                                hovertemplate='%{x|%Y-%m}<br>%{y} eligible currencies<extra>%{fullData.name}</extra>')
            fig.update_yaxes(title='Currencies with eligible FX paths', range=[0, 40], dtick=10)
            fig.update_xaxes(title='Forecast-origin month')
            title = f'{horizon}-month target · ' + ('12-return history required' if history else 'FX path only')
            banks['decision_eligibility'][f'{horizon}|{int(history)}'] = chart_spec(fig, title, height=520)
    totals = pd.DataFrame(totals)
    pd.DataFrame(monthly_records).to_csv(assets / 'eligibility_by_month.csv', index=False)
    totals.to_csv(assets / 'eligibility_totals.csv', index=False)
    stages = [len(calendar) * len(rows), int(observed.sum().sum()),
              int(launched.sum().sum()), int(levels.sum().sum())]
    stage_labels = ['39 × 392 calendar slots', 'Observed positive FX', 'After currency launches', 'After peg-period exclusions']
    pd.DataFrame({'stage': stage_labels, 'currency_months': stages}).to_csv(assets / 'level_eligibility.csv', index=False)
    fig = go.Figure(go.Bar(x=stages, y=stage_labels, orientation='h', text=[f'{n:,}' for n in stages],
                          textposition='auto', marker_color=['#9badb8', '#4273b4', '#32875d', '#147d92']))
    fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(title='Currency-month levels; before target and feature requirements')
    figures['decision-stages'] = chart_spec(fig, 'Why 39 currencies does not mean a balanced panel', 380)
    figures['decision-stages']['layout']['margin']['l'] = 210

    for source in (config_path, PROJECT_ROOT / 'docs' / 'data' / 'sample_and_treatment.md',
                   PROJECT_ROOT / 'docs' / 'project_brief.md'):
        if source.suffix == '.md':
            text = source.read_text().replace('](../../config/research_sample.json)', '](research_sample.json)').replace(
                '](data/sample_and_treatment.md)', '](sample_and_treatment.md)')
            (assets / source.name).write_text(text)
        else:
            shutil.copyfile(source, assets / source.name)
    manifest = {
        'purpose': 'Report-only FX eligibility audit; not a production feature panel or model result',
        'decision_status': choices['status'], 'production_status': choices['implementation_status'],
        'source_sha256': choices['evidence']['source_sha256'],
        'input_sha256': {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [config_path, *inputs.values()]},
        'definitions': {
            'path_only': 'Positive observed FX at origin and every month to t+h, after launch and inclusive regime masks',
            '12_returns': 'Path eligibility plus 13 consecutive eligible FX levels ending at origin; illustrative warm-up',
            'stock_available': 'FX origins additionally require observed positive aggregate USD credit at q(t)-2',
            'growth_available': 'Stock at q(t)-2 and q(t)-6 must both be observed positive',
            'exclusions': 'No REER gap, macro differential, dollar-index availability, training or model checks applied',
            'interpretation': 'Upper bounds on usable origins; regime filtering is retrospective',
        },
    }
    (assets / 'audit_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def source(key, label):
        return f'<a href="{html.escape(choices["references"][key], quote=True)}">{html.escape(label)} ↗</a>'

    roster_table = _table(['Currency', 'Economy', 'Group', 'First eligible month', 'Policy source', 'CPI source'],
                          [(r['currency'], r['economy'], 'Advanced' if r['group'] == 'advanced' else 'EMDE',
                            r['first_eligible_month'], 'Present' if r['source_series']['policy'] else 'Absent',
                            'Present' if r['source_series']['cpi'] else 'Absent') for r in rows])
    interval_table = _table(['Currency', 'Exclude from', 'Through (inclusive)', 'Reason'],
                            [(next(r['currency'] for r in rows if r['area'] == i['area']), i['start'], i['end'], i['reason'])
                             for i in choices['excluded_month_intervals']])
    lag_table = _table(['Input', 'Reference available at origin t', 'Reason / implication'], [
        ('FX', 'Month t', 'End-of-month target observation; no fill'),
        ('Policy rates', 't − 1 month', 'Allow a reference-month publication delay'),
        ('YoY inflation', 't − 2 months', 'Conservative monthly CPI proxy; derive before carry'),
        ('Broad REER', 't − 2 months', 'Avoid using the latest reference month as immediately available'),
        ('Broad US NEER', 't − 1 month', 'BIS dollar index; not the ICE DXY basket'),
        ('USD credit stock', 'q(t) − 2 quarters', 'Exact quarterly lookup; held fixed within origin quarter'),
    ])
    short_totals = totals.query("history == 'path_only'")
    totals_table = _table(['Horizon', 'Selected FX paths', 'With lagged stock', 'With lagged growth', 'China alternative: extra origins'],
                          [(f'{r.horizon} months', f'{r.primary_fx_only:,}', f'{r.stock_available:,}',
                            f'{r.growth_available:,}', f'{r.formal_china_cutoff - r.primary_fx_only:,}')
                           for r in short_totals.itertuples()])
    cpi = pd.read_csv(inputs['cpi'], index_col=0)
    us = cpi['M:US:628']
    september_inflation = 100 * np.log(us.loc['2025-09'] / us.loc['2024-09'])
    if pd.notna(us.loc['2025-10']):
        raise ValueError('The documented October 2025 CPI gap no longer matches the local snapshot')
    summary_table = _table(['Choice', 'Why selected', 'Cost / alternative'], [
        ('39 distinct currencies; unbalanced panel', 'Match the non-USD, broad-REER study scope without selecting on forecast success', 'Exact executive roster remains unconfirmed; complete-case selection would remove economies'),
        ('Exclude USD pegs and specified peg months', 'Focus on currencies that can move against the dollar', 'Loses history; China’s formal vs de facto cutoff needs sensitivity analysis'),
        ('Carry optional macro data at most one month; flag it', 'Handle isolated gaps without inventing long histories', 'Assumes a temporarily unchanged rate or inflation; strict no-carry sensitivity required'),
        ('Training-only median plus missing indicators for optional macros', 'Keep currencies with structurally absent policy/CPI sources', 'Median is a modeling assumption; structural gaps cannot become observed economics'),
        ('Conservative lags; native quarterly credit', 'Reduce obvious timing leakage and avoid artificial monthly credit changes', 'Latest revised data still do not establish real-time release availability'),
        ('Stock is the credit question; growth is a robustness check', 'The executive brief explicitly asks whether the outside-US dollar-credit stock helps', 'Functional form remains open; common credit sample begins only in July 2000'),
        ('Retain observed large FX moves', 'Stress periods are part of the forecasting problem', 'Sensitive losses require diagnostics; verified source breaks need separate treatment'),
    ])
    content = f'''
<header><div class="eyebrow">BIS / Research design / Report 03</div>
<h1>The choices.<br><span>The reasons. The consequences.</span></h1>
<p class="lede">A focused explanation of the selected currency sample and data treatment, with the alternatives and the sample costs made visible.</p>
<div class="meta">Decisions: {choices['decision_date']} · BIS snapshot: 8 October 2026 · Monthly scope: January 1994–August 2026</div>
<div class="report-switch"><a href="data_quality.html">01 / Data quality</a><a href="data_eda.html">02 / Data EDA</a><a href="research_choices.html" aria-current="page">03 / Research choices</a></div></header>
<nav aria-label="Report sections"><a href="#choices">Choices</a><a href="#roster">Currency sample</a><a href="#regimes">Regimes &amp; impact</a><a href="#missing">Missing data</a><a href="#timing">Timing &amp; credit</a><a href="#open">Open decisions</a><a href="#evidence">Evidence</a></nav>
<main><section id="choices"><div class="eyebrow">The selected approach</div><h2>Preserve the economics; restrict eligibility explicitly.</h2>
<p>Use an unbalanced 39-currency panel, preserve the observed BIS values, remove dollar-peg exposure from the primary FX paths, and treat sparse macro gaps causally. Apply the same sample and treatment to the compared models.</p>
<div class="kpis"><div><strong>39</strong><span>Selected currency representatives</span></div><div><strong>16 / 23</strong><span>Advanced / IMF emerging &amp; developing</span></div><div><strong>1 month</strong><span>Maximum optional macro carry</span></div><div><strong>2 quarters</strong><span>Lag for dollar credit</span></div></div>
<aside><b>Where we are.</b> These choices are documented in the canonical configuration. Production processing, features and benchmark models remain deferred. The charts below are a new <b>report-only FX eligibility audit</b>; they do not apply the full feature pipeline. Report 2 still describes its original 42-currency exploratory universe.</aside>
{summary_table}
</section>
<section id="roster"><div class="eyebrow">Choice 1 / Who belongs in the study?</div><h2>One representative per currency, without permanent dollar pegs.</h2>
<p><b>Selected:</b> start with the EDA’s 42 distinct non-USD currencies with broad REER coverage, then remove AED, HKD and SAR. Represent EUR only by the euro area (XM) from January 1999. Keep countries with missing optional macro sources, and let available histories vary by month.</p>
<p><b>Why:</b> FX reference areas are not distinct currencies. Repeated national EUR histories would give the same currency extra weight. Permanent dollar pegs create almost-flat targets; excluding them fits the executives’ non-peg research scope. Selecting on predictive correlations would bias later evaluation.</p>
<p><b>Trade-off:</b> 39 is our selected roster, not a verified reconstruction of the executives’ undisclosed list. Policy coverage is 35 of 39 here, versus 34 in their brief. Counts alone cannot resolve the difference. <a href="data_quality.html#brief-comparison">Report 1: brief versus observed data ↗</a></p>
<details><summary>Expand: exact roster, launch dates and missing-source flags</summary>{roster_table}
<p>BRL begins July 1994, BAM July 1998 (first full month after its June launch), and EUR January 1999. Earlier backcasts are excluded. Groups use {source('imf_groups', 'April 2026 IMF WEO Table B')} as static reporting labels, not historical features. EMDE is broader than an investable emerging-market index.</p>
<p>Sources: {source('brl_launch', 'Central Bank of Brazil')} · {source('bam_launch', 'Central Bank of Bosnia and Herzegovina')} · {source('eur_launch', 'ECB')} · {source('ae_peg', 'CBUAE')} · {source('ae_history', 'AED historical dollar relationship')} · {source('hk_peg', 'HKMA')} · {source('sa_peg', 'SAMA')}.</p></details>
<details><summary>Expand: alternatives and the comparison they change</summary><p>A balanced, complete-case panel would improve uniformity but remove countries and early histories. Keeping all 42 would mix permanent dollar pegs into the primary forecast comparison. Both are useful sensitivities, rather than the selected design. EUR-linked arrangements remain because their currencies can move against USD; excluding them separately tests common euro exposure. {source('dk_eur_peg', 'Danmarks Nationalbank')} explains Denmark’s euro link.</p></details>
</section>
<section id="regimes"><div class="eyebrow">Choice 2 / Which months belong?</div><h2>Exclude entire peg periods, including transition months.</h2>
<p><b>Selected:</b> remove the documented ARS, CNY and MYR intervals. A target needs positive observed FX at the origin and <b>every month through its horizon</b>, outside those intervals. Restart FX lag and rolling histories after launches, exclusions and missing observations.</p>
<p><b>Why:</b> a forecast should not bridge an excluded regime or a source gap. Whole-month exclusions avoid treating a partially pegged transition month as a freely moving monthly observation. China’s primary cutoff follows the more conservative de facto classification, rather than stopping at the formal July 2005 announcement.</p>
<details><summary>Expand: exact interval schedule and the China alternative</summary>{interval_table}
<p>The {source('china_regimes', 'IMF regime chronology, footnote 6')} places the end of China’s conventional fixed peg at 1 August 2006 and the later stabilized arrangement from 1 June 2008 to 21 June 2010. The sensitivity ends the initial exclusion in July 2005, following the {source('china_2005_reform', 'formal reform announcement')}; retain the later exclusion in both versions. Other evidence: {source('china_2010_reform', 'June 2010 reform')} · {source('malaysia_regimes', 'Bank Negara Malaysia chronology')} · {source('argentina_regime', 'IMF Argentina evaluation')}.</p>
<p><b>Interpretation limit:</b> this is a retrospective non-peg sample definition. It does not imply that a future regime transition was known when a forecast was made.</p></details>
<div class="chart" data-figure="decision-stages"></div>
<p class="caption">{stages[0] - stages[1]:,} calendar slots lack observed positive FX; currency launches remove a further {stages[1] - stages[2]:,} observed levels; peg intervals remove {stages[2] - stages[3]:,} more. These are level counts, before target horizons or feature completeness.</p>
<h3>Zoom: how many currencies can contribute at each origin?</h3>
<div class="controls"><label>Target horizon <select id="decision-horizon"><option value="1">1 month</option><option value="3">3 months</option><option value="12">12 months</option></select></label><label>History requirement <select id="decision-history"><option value="0">FX path only</option><option value="1">Plus 12 monthly returns</option></select></label><span>FX eligibility upper bound; macro features not checked</span></div>
<div class="chart" id="decision-eligibility" data-bank="decision_eligibility" data-selection="1|0"></div>
<p class="caption">Change the horizon to see target-endpoint losses; require 12 returns to see history resets. The dotted China alternative isolates the cutoff choice. Credit lines additionally require the exact lagged stock, or stock plus its four-quarter predecessor for growth. They share the same FX masks. A 60-month REER-gap window and other mandatory inputs would reduce eligibility further.</p>
<details><summary>Expand: total eligible currency-origin pairs and reproducible counts</summary>{totals_table}<p>Totals span the full 1994–2026 calendar, with FX-path requirements only. They are neither independent statistical observations nor production training counts. Credit/no-credit model comparisons must use the same credit-eligible origins.</p><div class="links"><a class="download" href="decision_assets/eligibility_by_month.csv">Monthly counts ↗ CSV</a><a class="download" href="decision_assets/eligibility_totals.csv">Totals &amp; history sensitivity ↗ CSV</a><a class="download" href="decision_assets/level_eligibility.csv">Level audit ↗ CSV</a></div></details>
</section>
<section id="missing"><div class="eyebrow">Choice 3 / What do we do with gaps?</div><h2>Separate short carry assumptions from genuinely missing economics.</h2>
<div class="insights"><article><h3>Policy: at most one reference month</h3><p>Carry only the last available observation, once, and flag it. Retain valid zero and negative rates. Japan’s 116 internal missing months across three long runs stay mostly null; they are not a zero-rate series.</p></article><article><h3>Inflation: carry the annual rate</h3><p>Compute 100 × log(CPIₜ / CPIₜ₋₁₂) first. Carry that derived rate at most once. Carrying the CPI level would imply zero monthly inflation and create an artificial annual-inflation path.</p></article><article><h3>Mandatory inputs: observed only</h3><p>No fill for FX, REER, NEER or credit. Missing FX invalidates targets and dependent history. A missing mandatory feature removes that origin from the relevant model sample.</p></article><article><h3>Optional macros: training-only imputation</h3><p>Persist policy and inflation nulls. At fitting time, use pooled training-window medians with missing indicators and structural-source flags. Never fit a replacement using future evaluation data.</p></article></div>
<p><b>Why:</b> isolated gaps can otherwise discard many usable currency forecasts. A one-month cap limits the economic assumption, while flags make it testable. Long gaps and never-existing series require an explicit model treatment rather than invented histories.</p>
<aside><b>Concrete example — US CPI, October 2025.</b> The local snapshot has no October CPI. September YoY log inflation is <b>{september_inflation:.3f}%</b>. The selected rule uses that September rate as a <b>flagged October fallback</b>, entering the December 2025 forecast origin after the two-month lag. This is an assumption, not an observed October value; the raw CPI file remains unchanged. Report 2’s original complete-case calculation still loses that origin. {source('us_cpi_gap', 'BLS explanation of the missing survey')}.</aside>
<details><summary>Expand: structural gaps, rejected treatments and robustness checks</summary>
<p>No policy source exists in this selection for BAM, DZD, SGD or TWD; CPI is absent for BAM and TWD. Their countries stay in the primary roster, with source-absent indicators. A training feature that is entirely missing must be dropped identically for all compared models or trigger a failure; it cannot use evaluation observations to learn a median.</p>
<p>Unlimited forward fill would fabricate long unchanged histories. Backward fill and two-sided interpolation violate the selected causal rule. Full-panel medians leak future distribution information. Dropping every incomplete row can change the EM/advanced composition.</p>
<p>Japan’s three internal policy gaps are March 1999–July 2000 (17 months), April 2001–February 2006 (59), and May 2013–August 2016 (40). A one-month carry would cover at most the first month of each run; it does not resolve the long missing history.</p>
<p>Compare the selected treatment with: strict complete cases on matched origins; removal of all carried rows; and omission of optional macro features. For each comparison, all models share the same sample and treatment. The median is an input assumption, not an estimate of the missing economic statistic.</p>
<p><a href="data_quality.html#coverage">Report 1: gap evidence ↗</a> · <a href="data_eda.html#credit-section">Report 2: original feature-availability consequences ↗</a>.</p></details>
<h3>Choice 4: retain observed extreme FX moves.</h3><p>Large moves are part of the forecasting objective and its stress periods. Preserve them; apply log transforms only to positive levels. Exclude a break only when independent evidence establishes a unit or source discontinuity, with the rationale recorded. Full-sample trimming would change the outcome being predicted.</p>
<details><summary>Expand: what this means for evaluation</summary><p>Show where extreme errors occur and whether gains survive stress-period diagnostics. Compare all models on the same untreated targets; do not infer neural-network gains from removing hard-to-predict months. <a href="data_quality.html#breaks">Report 1: large-change flags ↗</a> are review prompts, not diagnoses.</p></details>
</section>
<section id="timing"><div class="eyebrow">Choices 5 &amp; 6 / What can the model know?</div><h2>Conservative timing; stock first for the credit question.</h2>
<p><b>Selected timing:</b> use the following reference-period proxies from an end-of-month forecast origin. These reduce obvious look-ahead exposure, but are assumptions rather than a reconstructed publication calendar.</p>{lag_table}
<p><b>Selected credit requirement:</b> the primary question uses the outside-US aggregate USD-credit <b>stock</b>, series <code>{choices['macro_availability']['usd_credit_series']}</code>. Report 2’s annual log-growth predictor measures change in that stock and cannot answer exactly the same question. Growth remains a robustness specification; the stock’s level/log transformation is still open.</p>
<div class="insights"><article><h3>No synthetic monthly credit</h3><p>July and August 2026 use 2026Q1 because the origin quarter minus two is Q1. The missing 2026Q2 does not block those origins. No interpolation, additional carry or pre-2000 backfill is selected.</p></article><article><h3>Match the credit comparison</h3><p>Lagged stock first becomes available in July 2000; annual growth in July 2001. Credit/no-credit ablations share credit-eligible origins. A separate no-credit run can use earlier history, with a different sample clearly labeled.</p></article></div>
<details><summary>Expand: revised history, aggregate versus country credit, and remaining feature choices</summary><p>The BIS exports are latest revised histories. Reference-period lags do not remove later revisions or reproduce actual releases. Describe the evaluation as <b>pseudo out of sample</b>. Apply transformations, scaling and imputation within each expanding training window when implemented.</p>
<p>The outside-US aggregate fits the executives’ global-dollar-stock question and supplies a common input for the full roster. The 12 economy-specific credit series are supplementary candidates, not substitutes for the required aggregate. Their use and the stock functional form require an explicit specification before fitting.</p>
<p>The EDA’s past-return and 60-month REER-gap constructions are exploratory examples. Exact production lag sets, gap window, normalization and benchmark specifications remain to be locked. This report implements none of them.</p></details>
</section>
<section id="open" class="brief-discussion"><div class="eyebrow">What still needs to be settled</div><h2>The first choices are documented. These distinctions remain open.</h2>
<div class="discussion-questions"><article><span class="discussion-status">Scope correspondence / unconfirmed</span><h3>Is this the executives’ intended 39?</h3><p>Our 42 minus three USD pegs gives 39. Their original names were not supplied, and our policy-source coverage is 35 rather than 34. Confirm correspondence using the actual roster; the number alone is insufficient.</p></article><article><span class="discussion-status">Specification / not yet selected</span><h3>How should the stock enter the model?</h3><p>The brief requires stock; the EDA used growth. Choose the stock’s level/log form, train-window scaling and any country-credit addition before the benchmark comparison. Keep growth as a separately labeled sensitivity.</p></article><article><span class="discussion-status">Sensitivity / selected to test later</span><h3>Do results depend on the conservative restrictions?</h3><p>Compare China cutoffs, EUR-linked exclusions, strict no-carry rows and optional-macro treatments on matched samples. These checks are planned; there are no model results on the primary 39-currency design yet.</p></article><article><span class="discussion-status">Implementation / deferred</span><h3>What makes the next phase reviewable?</h3><p>A production processor must output the roster, eligibility masks and treatment flags; a feature script must enforce the lag and training rules. Then benchmarks can establish a random-walk comparison before neural networks.</p></article></div>
<div class="project-position"><h3>Current position</h3><div class="position-grid"><div><span class="position-label">Completed</span><p>Source ingestion, frozen quality review, exploratory EDA and this decision-impact audit.</p></div><div><span class="position-label">Selected</span><p>Roster, reporting groups, launch/peg restrictions, gap treatment and availability proxies.</p></div><div><span class="position-label">Pending</span><p>Roster correspondence, production feature details, stock functional form, benchmarks and networks.</p></div></div><p>The executive evaluation target remains unchanged: forecasts at 1, 3 and 12 months from January 2010, expanding windows, emerging versus advanced results, and a fair random-walk comparison. No decision here establishes predictability or model performance.</p></div>
</section>
<section id="evidence"><div class="eyebrow">Traceable and reproducible</div><h2>Read the specification or inspect the counts.</h2><p>The canonical JSON and decision record supply the selected rules. The audit checks positive observed FX paths, inclusive regime masks, optional 12-return warm-up and exact quarterly credit availability. It does not fill values, construct production features or fit models.</p>
<div class="links"><a class="download" href="decision_assets/research_sample.json">Canonical choices ↗ JSON</a><a class="download" href="decision_assets/sample_and_treatment.md">Decision record &amp; sources ↗</a><a class="download" href="decision_assets/project_brief.md">Executive brief ↗</a><a class="download" href="decision_assets/audit_manifest.json">Audit definitions &amp; input hashes ↗ JSON</a><a class="download" href="data_quality.html">01 / Data quality ↗</a><a class="download" href="data_eda.html">02 / Data EDA ↗</a></div>
<p class="sources">Generate this page with <code>python scripts/run_decision_report.py</code>; generate the three-report set with <code>python scripts/run_interactive_eda.py</code>. All plots embed Plotly for offline use. Data and calculations refer to the local 8 October BIS snapshot; sources cited in each expanded decision explain the historical classifications.</p></section></main>
<footer>EMERGING-MARKET FX / RESEARCH CHOICES / SELECTED DESIGN, PRODUCTION DEFERRED</footer>
'''
    path = destination / 'research_choices.html'
    path.write_text(render_document('Emerging-market FX · Research choices', content, figures, banks, get_plotlyjs()), encoding='utf-8')
    return path
