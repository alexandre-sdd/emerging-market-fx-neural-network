"""Report-only EDA on the canonical 39-currency sample and selected treatments.

Illustrative feature definitions do not constitute a production feature contract.
Optional macro gaps remain missing; no learned imputation is fitted here.
"""

from __future__ import annotations

import hashlib
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from fx_forecasting.config import PROCESSED_DIR, PROJECT_ROOT
from fx_forecasting.data.interactive_eda import currency_metrics, random_walk_metrics, safe_corr, window_mask
from fx_forecasting.data.quality_report import chart_spec


FEATURES = {
    'fx_1m': 'Latest FX change', 'momentum_3m': '3m FX momentum',
    'volatility_12m': '12m FX volatility', 'reer_gap': 'Lagged REER gap (60m)',
    'rate_diff': 'Policy differential', 'inflation_diff': 'Inflation differential',
    'usd_move': 'Lagged broad USD move', 'credit_stock': 'USD credit stock (log)',
    'credit_growth': 'USD credit YoY growth',
}
GROUPS = {'all': 'All 39', 'advanced': 'Advanced', 'emerging_and_developing': 'EMDE'}
WINDOWS = {'train': 'Targets realised by Dec 2009', 'later': 'Origins from Jan 2010'}


def _table(frame):
    return '<div class="table-scroll">' + frame.to_html(index=False, classes='data-table', border=0, float_format=lambda v: f'{v:.3f}', na_rep='—') + '</div>'


def _carry_once(frame, limit):
    carried = frame.ffill(limit=limit)
    return carried, frame.isna() & carried.notna()


def calculate_selected_eda(fx, levels, macro, choices):
    """Apply causal carries and lag proxies for descriptive, pairwise analysis."""
    index = fx.index
    masked_fx = fx.where(levels)
    logfx = np.log(masked_fx)
    returns = 100 * logfx.diff()
    # Require every intervening eligible level for multi-month past and future changes.
    momentum = (100 * logfx.diff(3)).where(levels.rolling(4, min_periods=4).sum().eq(4))
    targets = {}
    for horizon in (1, 3, 12):
        valid = levels.copy()
        for offset in range(1, horizon + 1):
            valid &= levels.shift(-offset, fill_value=False)
        targets[horizon] = (100 * (logfx.shift(-horizon) - logfx)).where(valid)
    lags = choices['macro_availability']['reference_lags_months']
    policy, policy_carried = _carry_once(macro['policy'], choices['missing_data']['policy']['causal_carry_limit_months'])
    annual_inflation = 100 * np.log(macro['cpi'].where(macro['cpi'].gt(0))).diff(12)
    inflation, inflation_carried = _carry_once(annual_inflation, choices['missing_data']['inflation']['causal_carry_limit_months'])
    rate_diff = policy.sub(policy['US'], axis=0).shift(lags['policy'])
    inflation_diff = inflation.sub(inflation['US'], axis=0).shift(lags['cpi_inflation'])
    reer = np.log(macro['reer'].where(macro['reer'].gt(0))).shift(lags['reer'])
    dollar_move = (100 * np.log(macro['neer']['US'].where(macro['neer']['US'].gt(0))).diff()).shift(lags['usd_neer'])
    stock = macro['usd_credit']['3P'].where(macro['usd_credit']['3P'].gt(0))
    quarter = index.asfreq('Q') - choices['macro_availability']['usd_credit_quarter_lag']
    credit_stock = pd.Series(np.log(stock).reindex(quarter).to_numpy(), index=index)
    credit_growth = pd.Series((100 * np.log(stock).diff(4)).reindex(quarter).to_numpy(), index=index)

    def broadcast(series):
        return pd.DataFrame({area: series for area in fx}, index=index)

    features = {
        'fx_1m': returns, 'momentum_3m': momentum,
        'volatility_12m': returns.rolling(12, min_periods=12).std(),
        'reer_gap': 100 * (reer - reer.shift(1).rolling(60, min_periods=60).mean()),
        'rate_diff': rate_diff, 'inflation_diff': inflation_diff,
        'usd_move': broadcast(dollar_move), 'credit_stock': broadcast(credit_stock),
        'credit_growth': broadcast(credit_growth),
    }
    features = {key: value.reindex(index=index, columns=fx.columns).where(levels) for key, value in features.items()}
    flags = {}
    for prefix, values, lag in [('policy', policy_carried, lags['policy']),
                                ('inflation', inflation_carried, lags['cpi_inflation'])]:
        flags[f'{prefix}_carried_domestic'] = values.reindex(columns=fx.columns).shift(lag, fill_value=False).where(levels, False)
        flags[f'{prefix}_carried_us'] = broadcast(values['US'].shift(lag, fill_value=False)).where(levels, False)
    return masked_fx, returns, features, targets, flags


def build_selected_eda(fx, levels, choices, destination):
    """Return an HTML section, charts and selectors, writing auditable CSVs."""
    assets = destination / 'selected_39'
    assets.mkdir(parents=True, exist_ok=True)
    index = fx.index
    meta = pd.DataFrame(choices['currencies']).set_index('area', drop=False)
    meta['Reference area'], meta['CURRENCY'] = meta['economy'], meta['currency']
    labels = {area: f'{row.currency} · {row.economy}' for area, row in meta.iterrows()}
    macro, input_paths = {}, []
    for component in ('reer', 'neer', 'policy', 'cpi', 'usd_credit'):
        path = PROCESSED_DIR / f'eda_{component}.csv'
        input_paths.append(path)
        raw = pd.read_csv(path, index_col=0)
        raw.index = pd.PeriodIndex(raw.index, freq='Q' if component == 'usd_credit' else 'M')
        if component == 'usd_credit':
            macro[component] = pd.DataFrame({'3P': raw[choices['macro_availability']['usd_credit_series']]})
        else:
            selected = {row['area']: raw[row['source_series'][component]] if row['source_series'][component] else pd.Series(np.nan, index=raw.index)
                        for row in choices['currencies']}
            us_key = {'reer': 'M:R:B:US', 'neer': 'M:N:B:US', 'policy': 'M:US', 'cpi': 'M:US:628'}[component]
            selected['US'] = raw[us_key]
            macro[component] = pd.DataFrame(selected).reindex(index)
    masked_fx, returns, features, targets, flags = calculate_selected_eda(fx, levels, macro, choices)
    metrics = currency_metrics(returns, meta).merge(meta.reset_index(drop=True)[['area', 'group']], on='area')
    baseline, _ = random_walk_metrics(targets, meta)
    baseline = baseline.merge(meta.reset_index(drop=True)[['area', 'group', 'currency']], on='area')
    correlation_rows, availability_rows = [], []
    carried_any = pd.DataFrame(False, index=index, columns=fx.columns)
    for values in flags.values():
        carried_any |= values
    mandatory = features['fx_1m'].notna() & features['momentum_3m'].notna() & features['volatility_12m'].notna() & features['reer_gap'].notna() & features['usd_move'].notna()
    optional = features['rate_diff'].notna() & features['inflation_diff'].notna()
    for horizon, target in targets.items():
        for window in WINDOWS:
            time_mask = window_mask(index, horizon, window)
            for area in fx:
                for key, value in features.items():
                    rho, n = safe_corr(value.loc[time_mask, area], target.loc[time_mask, area])
                    pair = time_mask & value[area].notna().to_numpy() & target[area].notna().to_numpy()
                    rho_uncarried, n_uncarried = safe_corr(value.loc[pair & ~carried_any[area].to_numpy(), area], target.loc[pair & ~carried_any[area].to_numpy(), area])
                    correlation_rows.append(dict(horizon=horizon, window=window, method='spearman', area=area,
                        currency=meta.at[area, 'currency'], group=meta.at[area, 'group'], feature=key,
                        correlation=rho, n=n, no_carry_correlation=rho_uncarried, no_carry_n=n_uncarried))
        eligible = target.notna()
        for month in index:
            core = eligible.loc[month] & mandatory.loc[month]
            availability_rows.append(dict(month=str(month), horizon=horizon, fx_targets=int(eligible.loc[month].sum()),
                mandatory_complete=int(core.sum()), macro_complete=int((core & optional.loc[month]).sum()),
                with_credit_stock=int((core & features['credit_stock'].loc[month].notna()).sum()),
                fully_observed_with_stock=int((core & optional.loc[month] & features['credit_stock'].loc[month].notna()).sum()),
                carried_macro_rows=int((core & optional.loc[month] & carried_any.loc[month]).sum())))
    correlations = pd.DataFrame(correlation_rows)
    availability = pd.DataFrame(availability_rows)
    group_stats = metrics.query("window == 'later'").groupby('group').agg(currencies=('area', 'size'),
        median_months=('n', 'median'), median_annual_volatility=('annual_volatility', 'median'),
        median_annual_drift=('annual_log_drift', 'median'), median_abs_acf=('lag1_autocorrelation', lambda s: s.abs().median()),
        median_tail_gt10_pct=('tail_gt10_pct', 'median'))
    group_stats.index.name = 'group'
    benchmark_summary = baseline.groupby(['horizon', 'group']).agg(currencies=('area', 'size'),
        currency_origins=('n', 'sum'), median_rw_rmse=('rw_rmse', 'median'),
        mean_wins=('mean_r2_os', lambda s: int(s.gt(0).sum())), median_mean_r2_os=('mean_r2_os', 'median')).reset_index()
    overlap_pairs = []
    for horizon in (1, 3, 12):
        for area in fx:
            before = correlations.query("horizon == @horizon and area == @area and window == 'train'").set_index('feature')
            after = correlations.query("horizon == @horizon and area == @area and window == 'later'").set_index('feature')
            for key in FEATURES:
                if pd.notna(before.at[key, 'correlation']) and pd.notna(after.at[key, 'correlation']):
                    overlap_pairs.append(dict(horizon=horizon, area=area, feature=key,
                        before=before.at[key, 'correlation'], later=after.at[key, 'correlation']))
    stability = pd.DataFrame(overlap_pairs)
    credit_summary = stability.loc[stability.feature.isin(['credit_stock', 'credit_growth'])].groupby(['horizon', 'feature']).agg(
        matched_currencies=('area', 'size'), median_before=('before', 'median'), median_later=('later', 'median')).reset_index()
    credit_summary.to_csv(assets / 'credit_signal_summary.csv', index=False)
    group_stats.reset_index().to_csv(assets / 'group_summary.csv', index=False)
    for filename, frame in [('currency_metrics', metrics), ('predictive_correlations', correlations),
                            ('baseline_metrics', baseline), ('benchmark_summary', benchmark_summary),
                            ('feature_availability', availability), ('signal_stability', stability)]:
        frame.to_csv(assets / f'{filename}.csv', index=False)
    # Preserve calendar rows and treatment flags for report reproduction, not model ingestion.
    panel_parts = []
    for area in fx:
        part = pd.DataFrame({'month': index.astype(str), 'area': area, 'currency': meta.at[area, 'currency'],
            'group': meta.at[area, 'group'], 'fx_level_eligible': levels[area].to_numpy(),
            'fx': masked_fx[area].to_numpy(), 'monthly_return': returns[area].to_numpy()})
        for key, values in {**features, **flags}.items():
            part[key] = values[area].to_numpy()
        for horizon, target in targets.items():
            part[f'target_{horizon}m'] = target[area].to_numpy()
        part['policy_source_absent'] = meta.at[area, 'source_series']['policy'] is None
        part['cpi_source_absent'] = meta.at[area, 'source_series']['cpi'] is None
        panel_parts.append(part)
    pd.concat(panel_parts, ignore_index=True).to_csv(assets / 'descriptive_panel.csv', index=False)

    figures, banks = {}, {'selected_signals': {}, 'selected_baseline': {}, 'selected_currency': {}}
    later = returns.loc['2010-01':]
    comovement = later.corr(min_periods=60)
    comovement.to_csv(assets / 'return_correlations.csv')
    balanced = later.dropna()
    if len(balanced) < 60 or balanced.std().le(1e-8).any():
        raise ValueError('Insufficient balanced variation for descriptive 39-currency PCA')
    pca = PCA().fit((balanced - balanced.mean()) / balanced.std())
    pc1 = float(100 * pca.explained_variance_ratio_[0])
    pd.DataFrame({'component': np.arange(1, 40), 'explained_variance_pct': 100 * pca.explained_variance_ratio_}).to_csv(assets / 'pca_variance.csv', index=False)
    fig = go.Figure(go.Heatmap(z=comovement.to_numpy(), x=[meta.at[a, 'currency'] for a in fx],
        y=[meta.at[a, 'currency'] for a in fx], zmin=-1, zmax=1, colorscale='RdBu', reversescale=True,
        customdata=later.notna().astype(int).T.dot(later.notna().astype(int)).to_numpy(),
        hovertemplate='%{y} / %{x}<br>Pearson r=%{z:.3f}<br>Paired months=%{customdata}<extra></extra>'))
    figures['selected-comovement'] = chart_spec(fig, 'Return co-movement · selected 39 · origins from January 2010', 740)
    order = metrics.query("window == 'later'").sort_values('annual_volatility', ascending=False)
    fig = go.Figure()
    for group, color in [('emerging_and_developing', '#c35d38'), ('advanced', '#147d92')]:
        values = order.query('group == @group')
        fig.add_bar(x=values.currency, y=values.annual_volatility, name=GROUPS[group], marker_color=color,
            customdata=values[['n', 'annual_log_drift', 'lag1_autocorrelation', 'tail_gt10_pct']],
            hovertemplate='%{x}<br>Annualised volatility=%{y:.2f} log%<br>Months=%{customdata[0]}<br>Annualised drift=%{customdata[1]:.2f} log%<br>Lag-1 correlation=%{customdata[2]:.3f}<br>|move| >10 log%=%{customdata[3]:.2f}%<extra></extra>')
    fig.update_xaxes(categoryorder='array', categoryarray=order.currency.tolist(), tickangle=-45)
    fig.update_yaxes(title='Monthly SD × √12 (log %)')
    figures['selected-behaviour'] = chart_spec(fig, 'Risk differs across the selected currencies · monthly returns from 2010', 460)
    for horizon in (1, 3, 12):
        for window in WINDOWS:
            for group in GROUPS:
                areas = meta.index if group == 'all' else meta.index[meta.group.eq(group)]
                subset = correlations.query('horizon == @horizon and window == @window').pivot(index='area', columns='feature', values='correlation').reindex(index=areas, columns=FEATURES)
                sizes = correlations.query('horizon == @horizon and window == @window').pivot(index='area', columns='feature', values='n').reindex(index=areas, columns=FEATURES)
                fig = go.Figure(go.Heatmap(z=subset.to_numpy(), x=list(FEATURES.values()), y=[meta.at[a, 'currency'] for a in areas],
                    customdata=sizes.to_numpy(), zmin=-.6, zmax=.6, colorscale='RdBu', reversescale=True,
                    hovertemplate='%{y}<br>%{x}<br>Spearman ρ=%{z:.3f}<br>Paired origins=%{customdata}<extra></extra>'))
                fig.update_xaxes(tickangle=-30)
                banks['selected_signals'][f'{horizon}|{window}|{group}'] = chart_spec(fig,
                    f'{horizon}m future FX change · {GROUPS[group]} · {WINDOWS[window]}', 350 + len(areas) * 15)
                banks['selected_signals'][f'{horizon}|{window}|{group}']['layout']['margin']['b'] = 135
        b = baseline.query('horizon == @horizon').sort_values('rw_rmse', ascending=False)
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Random-walk error scale', 'Expanding mean vs random walk'))
        colors = ['#147d92' if g == 'advanced' else '#c35d38' for g in b.group]
        fig.add_bar(x=b.currency, y=b.rw_rmse, customdata=b.n, marker_color=colors,
            hovertemplate='%{x}<br>RW RMSE=%{y:.3f} log pp<br>Matched origins=%{customdata}<extra></extra>', row=1, col=1)
        fig.add_bar(x=b.currency, y=b.mean_r2_os, customdata=b.n, marker_color=colors,
            hovertemplate='%{x}<br>Mean R² OS=%{y:.3f}%<br>Matched origins=%{customdata}<extra></extra>', row=1, col=2)
        fig.update_xaxes(tickangle=-60)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title='RMSE (log pp)', row=1, col=1)
        fig.update_yaxes(title='100 × [1 − MSPE(mean) / MSPE(RW)]', row=1, col=2)
        banks['selected_baseline'][str(horizon)] = chart_spec(fig, f'{horizon}m · FX-only diagnostic · from January 2010', 490)
        a = availability.query('horizon == @horizon')
        fig = go.Figure()
        for key, label in [('fx_targets', 'Eligible FX target'), ('mandatory_complete', 'Mandatory illustrative inputs'),
                           ('macro_complete', 'Plus available policy / inflation'), ('fully_observed_with_stock', 'Plus credit stock')]:
            fig.add_scatter(x=pd.PeriodIndex(a.month, freq='M').to_timestamp(), y=a[key], name=label, mode='lines')
        fig.update_yaxes(title='Currency-origin pairs in month', range=[0, 40])
        figures[f'selected-availability-{horizon}'] = chart_spec(fig, f'{horizon}m · coverage after selected treatments and illustrative warm-up', 480)
    for area in fx:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('FX path · first eligible month from 2010 = 100', 'Monthly log changes · selected history'))
        path = masked_fx.loc['2010-01':, area]
        path = 100 * path / path.dropna().iloc[0]
        fig.add_scatter(x=path.index.to_timestamp(), y=path, mode='lines', connectgaps=False, name='FX index', row=1, col=1)
        fig.add_histogram(x=returns[area].dropna(), nbinsx=65, name='Monthly changes', row=1, col=2)
        fig.update_yaxes(type='log', title='Indexed LCU/USD · log scale', row=1, col=1)
        fig.update_xaxes(title='100 × monthly Δlog(LCU/USD)', row=1, col=2)
        fig.update_layout(showlegend=False)
        banks['selected_currency'][area] = chart_spec(fig, labels[area], 430)

    em = group_stats.loc['emerging_and_developing']
    advanced = group_stats.loc['advanced']
    b1 = baseline.query('horizon == 1')
    matched_credit = stability.query("horizon == 12 and feature == 'credit_growth'")
    matched_stock = stability.query("horizon == 12 and feature == 'credit_stock'")
    matched_momentum = stability.query("horizon == 1 and feature == 'fx_1m'")
    group_display = group_stats.reset_index().rename(columns={'group': 'Group', 'currencies': 'Currencies', 'median_months': 'Median months',
        'median_annual_volatility': 'Median volatility (log %)', 'median_annual_drift': 'Median drift (log %)',
        'median_abs_acf': 'Median |lag-1 r|', 'median_tail_gt10_pct': 'Median >10 log% months (%)'})
    group_display['Group'] = group_display.Group.map(GROUPS)
    summary_display = benchmark_summary.rename(columns={'horizon': 'Horizon (m)', 'group': 'Group', 'currencies': 'Currencies',
        'currency_origins': 'Matched currency-origins', 'median_rw_rmse': 'Median RW RMSE (log pp)',
        'mean_wins': 'Mean beats RW', 'median_mean_r2_os': 'Median mean R² OS (%)'})
    summary_display['Group'] = summary_display.Group.map(GROUPS)
    settings = {
        'purpose': 'Selected-39 descriptive EDA; no production features, learned imputation or neural network fitting',
        'sample': 'Canonical 39 currencies; launch and inclusive peg masks; every month in target horizon eligible',
        'input_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [*input_paths, PROCESSED_DIR / 'eda_fx.csv', PROJECT_ROOT / 'config/research_sample.json']},
        'features': {
            'fx_1m': '100 log(S_t/S_(t-1)); both eligible',
            'momentum_3m': '100 log(S_t/S_(t-3)); all four levels eligible',
            'volatility_12m': 'SD of 12 consecutive eligible monthly returns',
            'reer_gap': '100[log(REER_(t-2)) - mean(log(REER_(t-62)...REER_(t-3)))]; full 60 reference months',
            'rate_diff': 'Domestic - US policy rate at t-1, after flagged one-month causal carry',
            'inflation_diff': 'Domestic - US YoY log inflation at t-2; derive first, then one-month causal carry',
            'usd_move': '100 log(US broad NEER_(t-1)/US broad NEER_(t-2))',
            'credit_stock': 'log(outside-US credit_q), q=quarter(t)-2; illustrative transform, not production form selection',
            'credit_growth': '100 log(credit_q/credit_(q-4)), same q; separate robustness input',
        },
        'pca': {'n_currencies': 39, 'n_complete_months': len(balanced), 'pc1_variance_pct': pc1,
                'scope': 'All 39; complete post-2010 monthly returns, full-period standardization for description only'},
        'correlations': {'method': 'Spearman', 'minimum_pairs': 36, 'constant_tolerance': 1e-8,
            'train': 'Target realised by December 2009', 'later': 'Forecast origins January 2010 onward',
            'no_carry_sensitivity': 'Omit origins with any domestic or US policy/inflation carry; same raw pair requirement'},
        'benchmark': 'FX-only random walk and per-currency expanding historical mean; >=60 realised training labels and >=36 matched evaluation origins; not full benchmark pipeline',
        'limitations': ['Revised history and reference-period lag proxies: pseudo out of sample',
            'Pairwise predictor samples differ; stock/growth correlations are not a matched predictive-performance ablation',
            'Overlapping targets and repeated global predictors reduce independent information',
            'Stock trends can produce nonstationary correlations; no causality or forecast-success claim',
            'Reporting groups fixed to April 2026 IMF; FX regime filter retrospective'],
    }
    (assets / 'analysis_manifest.json').write_text(json.dumps(settings, indent=2) + '\n')
    options = ''.join(f'<option value="{area}">{html.escape(labels[area])}</option>' for area in fx)
    section = f'''
<section id="selected-eda"><div class="eyebrow">Study EDA / Selected 39 / New analysis</div>
<h2>What the chosen sample says about the forecasting study.</h2>
<p>This analysis uses the selected <b>39 currencies</b>, currency starts, peg-period masks and causal macro carries, with the selected availability lags. Report 2 remains the original 42-currency exploration. FX paths and macro inputs here are report calculations; production feature definitions and learned imputation remain deferred.</p>
<div class="kpis"><div><strong>{em.median_annual_volatility:.1f}%</strong><span>Median EMDE annualised FX volatility · from 2010</span></div><div><strong>{advanced.median_annual_volatility:.1f}%</strong><span>Median advanced annualised FX volatility · from 2010</span></div><div><strong>{pc1:.1f}%</strong><span>Return variation explained by PC1 · all 39</span></div><div><strong>{int(b1.mean_r2_os.gt(0).sum())}/{len(b1)}</strong><span>Expanding mean beats random walk · 1m diagnostic</span></div></div>
<div class="insights"><article><h3>Shared structure supports the global-network experiment</h3><p>The first component explains {pc1:.1f}% of standardised return variation across all 39 currencies in {len(balanced)} complete post-2010 months. Shared exposure motivates pooling; different risk profiles motivate currency embeddings. A common USD numeraire also induces co-movement.</p></article><article><h3>Emerging versus advanced needs separate reporting</h3><p>Median annualised volatility is {em.median_annual_volatility:.1f} log% for EMDE and {advanced.median_annual_volatility:.1f} log% for advanced currencies. Report relative random-walk error as well as absolute errors: one high-volatility currency can dominate pooled squared loss.</p></article><article><h3>Persistence should be tested for stability</h3><p>Across {len(matched_momentum)} currencies estimable in both periods, median latest-move / next-month Spearman correlation changes from {matched_momentum.before.median():.3f} before 2010 to {matched_momentum.later.median():.3f} later. These are descriptive associations, not a deployable momentum rule.</p></article><article><h3>Stock and growth tell different stories</h3><p>For {len(matched_credit)} currencies estimable in both periods, median credit-growth / 12m-target correlation is {matched_credit.before.median():.3f} before 2010 and {matched_credit.later.median():.3f} later. The corresponding stock medians are {matched_stock.before.median():.3f} and {matched_stock.later.median():.3f} across {len(matched_stock)} currencies. Different pairwise origin samples and stock trends prevent a forecast-performance conclusion.</p></article></div>
<aside><b>Study implication.</b> Compare the shared network, embeddings and separate networks on identical inputs and origins, with equal attention to EMDE performance and 3m / 12m horizons. The expanding mean’s median 1m R² against random walk is {b1.mean_r2_os.median():.2f}%; a neural network still has to establish an out-of-sample gain. No neural networks or production benchmarks are fitted here.</aside>
<details><summary>Expand: currency behaviour, tails and group metrics</summary>
{_table(group_display)}<div class="chart" data-figure="selected-behaviour"></div>
<p class="caption">Group statistics are medians of currency-level estimates, not pooled returns; coverage varies. Volatility = monthly SD × √12, drift = monthly mean × 12. Annualisation is descriptive and does not establish independent returns. Large moves are retained; hover for tails, drift and persistence.</p>
<div class="controls"><label>Selected currency <select id="selected-currency">{options}</select></label></div>
<div class="chart" id="selected-currency-chart" data-bank="selected_currency" data-selection="DZ"></div>
<p class="caption">Path indexing uses each currency’s first eligible observation from 2010; it is not a common investment start. Missing and excluded intervals break the line. Histogram uses all eligible adjacent monthly changes in 1994–2026.</p></details>
<details><summary>Expand: common movement across all 39 currencies</summary>
<div class="chart" data-figure="selected-comovement"></div>
<p class="caption">Contemporaneous Pearson correlations, at least 60 paired post-2010 months; hover shows counts. PCA uses all 39 on {len(balanced)} complete months and standardises within that descriptive full-period subset. PCA is not fitted as a production feature or on a future training split.</p></details>
<h3>Zoom: which inputs relate to future changes?</h3>
<div class="controls"><label>Horizon <select id="selected-signal-horizon"><option value="1">1 month</option><option value="3">3 months</option><option value="12">12 months</option></select></label><label>Period <select id="selected-signal-window"><option value="train">Targets realised before 2010</option><option value="later">Origins from 2010</option></select></label><label>Group <select id="selected-signal-group"><option value="all">All 39</option><option value="emerging_and_developing">EMDE</option><option value="advanced">Advanced</option></select></label></div>
<div class="chart" id="selected-signal-chart" data-bank="selected_signals" data-selection="1|train|all"></div>
<p class="caption">Per-currency Spearman correlations, minimum 36 pairs; blank cells indicate insufficient or nearly constant inputs. Colours saturate at ±0.6; hover shows actual values and counts. Pre-2010 targets must end by December 2009. Later associations are visible diagnostics and cannot be used to tune an untouched test.</p>
<details><summary>Expand: exact timing, treatment and interpretation limits</summary>
<p>All FX lag paths reset at excluded or missing levels. Momentum requires four consecutive levels; volatility twelve adjacent monthly changes. The illustrative REER gap compares log REER at t−2 with its preceding 60 reference months. US broad NEER changes are lagged one month. Policy differentials use t−1; annual log-inflation differentials use t−2 after the selected one-month carry of the annual rate.</p>
<p>The credit column uses log stock at q(t)−2 only for this EDA; its Spearman rank correlation equals that of the positive stock level. This does not lock the production functional form. Annual growth uses q(t)−2 versus four quarters earlier. Stock and growth columns have different pairwise samples; compare model performance on a common credit-eligible sample later.</p>
{_table(credit_summary.rename(columns={'horizon': 'Horizon (m)', 'feature': 'Credit input', 'matched_currencies': 'Currencies estimable in both periods', 'median_before': 'Median ρ before 2010', 'median_later': 'Median ρ from 2010'}))}
<p>Optional gaps remain null; no training median or future-fitted transformation is applied here. The correlation CSV also reports a sensitivity removing every origin with any carried macro input. The October 2025 US annual-inflation fallback now prevents the original all-currency December gap, but absent domestic sources and longer gaps persist.</p>
<p>Overlapping 3m/12m targets, repeated quarterly credit and global USD exposure reduce independent information. Trending stock relationships can be spurious. Neither correlations nor this descriptive PCA establish causality or predictive success. See <a href="data_quality.html#coverage">Report 1 for the underlying source gaps ↗</a>.</p></details>
<details><summary>Expand: availability after the selected treatment</summary><div class="chart" data-figure="selected-availability-1"></div>
<p class="caption">1m future targets plus mandatory illustrative inputs; policy/inflation and credit added separately. Optional macro carries are flagged. Counts before learned imputation are not final model sample sizes; structural gaps remain missing. August 2026 has no observed 1m outcome in this snapshot, so its zero count reflects missing future labels rather than absent forecast inputs. CSV includes all three horizons.</p>
{_table(availability.query("horizon == 1 and month in ['2010-01','2025-12','2026-07','2026-08']"))}</details>
<details><summary>Expand: random-walk difficulty and an expanding-mean diagnostic</summary>
<div class="controls"><label>Horizon <select id="selected-baseline-horizon"><option value="1">1 month</option><option value="3">3 months</option><option value="12">12 months</option></select></label></div>
<div class="chart" id="selected-baseline-chart" data-bank="selected_baseline" data-selection="1"></div>{_table(summary_display)}
<p class="caption">Teal: advanced; orange: EMDE. Mean forecasts use only h-month labels realised by each origin, with at least 60 historical labels and 36 evaluation pairs. Both forecasts share the same eligible FX targets. This FX-only diagnostic does not require macro-feature completeness and does not implement the AR(1), Taylor-rule or elastic-net benchmark pipeline. Positive R² means smaller squared error; wins do not imply significance. RW makes no directional call; mean direction accuracy and its nonzero-pair denominator are in the CSV.</p></details>
<div class="links"><a class="download" href="selected_39/currency_metrics.csv">Currency metrics ↗ CSV</a><a class="download" href="selected_39/group_summary.csv">EMDE / advanced summary ↗ CSV</a><a class="download" href="selected_39/predictive_correlations.csv">Signals &amp; no-carry sensitivity ↗ CSV</a><a class="download" href="selected_39/baseline_metrics.csv">Diagnostic forecast metrics ↗ CSV</a><a class="download" href="selected_39/feature_availability.csv">Availability ↗ CSV</a><a class="download" href="selected_39/descriptive_panel.csv">Report-only panel &amp; flags ↗ CSV</a><a class="download" href="selected_39/analysis_manifest.json">Definitions &amp; provenance ↗ JSON</a></div>
</section>
'''
    return section, figures, banks
