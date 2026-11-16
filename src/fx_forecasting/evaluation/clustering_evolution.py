"""Chronological currency memberships and genuine trailing-five-year co-clustering.

R2K follows Algorithm 2 continuously. Sixty months is the reporting window,
not its centroid bank W or its feature lookback. No bootstrap is used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from fx_forecasting.data.rolling_features import make_rolling_features
from fx_forecasting.evaluation.rolling_clustering import run_cross_sections
from fx_forecasting.models.currency_clustering import _geometry

OUTPUTS = ['evolution_labels.npz', 'evolution_metadata.json', 'evolution_metrics.csv',
           'evolution_summary.csv', 'evolution_pairs.csv', 'evolution_currency.csv',
           'evolution_initialization.csv']
PAPER_REFERENCE = {'url': 'https://ssrn.com/abstract=4729393', 'version': '2024-03-06',
                   'sha256': 'c9b7df42c5307b5802058eb8b22217247dd07d194d175ce5863cc9227d665e9d', 'algorithm': 'Algorithm 2, page 4'}
NAMES = {'r2k_w1': 'R2K · W=1 month', 'r2k_w6': 'R2K · W=6 months',
         'r2k_w12': 'R2K · W=12 months', 'adjusted_average': 'Adjusted returns · 60-month refits',
         'returns_average': 'Original returns · 60-month refits'}


def pair_frequencies(labels: np.ndarray, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Count co-membership only when both currencies are observed; -1 means absent.

    A full calendar window is required, but currencies may have fewer eligible
    months within it. Return numerator and denominator, not filled probabilities.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or window < 1 or np.any(labels < -1):
        raise ValueError('Expected monthly labels [T,C], missing=-1 and a positive window')
    observed = labels >= 0
    eligible = observed[:, :, None] & observed[:, None, :]
    together = eligible & (labels[:, :, None] == labels[:, None, :])
    def total(x):
        acc = np.concatenate([np.zeros((1, *x.shape[1:]), dtype=np.int32), x.cumsum(axis=0)], axis=0)
        out = np.zeros_like(x, dtype=np.int16)
        if len(x) >= window:
            out[window-1:] = acc[window:] - acc[:-window]
        return out
    return total(together), total(eligible)


def align_partition(current: np.ndarray, previous: np.ndarray, k: int) -> np.ndarray:
    """Maximum-overlap IDs for refitted correlation partitions, never for R2K."""
    valid = (current >= 0) & (previous >= 0)
    overlap = np.zeros((k, k), dtype=int)
    np.add.at(overlap, (previous[valid], current[valid]), 1)
    old, new = linear_sum_assignment(-overlap)
    mapping = np.empty(k, dtype=int); mapping[new] = old
    return np.where(current >= 0, mapping[np.maximum(current, 0)], -1)


def describe_path(labels: np.ndarray, dates: pd.PeriodIndex, method: str, k: int) -> pd.DataFrame:
    rows = []
    for t, current in enumerate(labels):
        valid = current >= 0; counts = np.bincount(current[valid], minlength=k)
        if not valid.any():
            continue
        props = counts[counts > 0] / valid.sum()
        row = {'method': method, 'k': k, 'period': str(dates[t]), 'n_assets': int(valid.sum()),
               'active_groups': int((counts > 0).sum()), 'sizes': json.dumps(counts.tolist()),
               'largest_share': float(props.max()), 'effective_groups': float(np.exp(-(props*np.log(props)).sum())),
               'singleton_assets': int((counts == 1).sum()), 'singleton_fraction': float((counts == 1).sum()/valid.sum()),
               'pair_together_fraction': float((counts*(counts-1)).sum()/(valid.sum()*(valid.sum()-1))) if valid.sum()>1 else np.nan}
        for lag in (1, 12, 60):
            common = valid & (labels[t-lag] >= 0) if t >= lag else np.zeros(len(current), dtype=bool)
            row[f'n_common_lag{lag}'] = int(common.sum())
            row[f'ari_lag{lag}'] = float(adjusted_rand_score(labels[t-lag, common], current[common])) if common.sum() > 1 else np.nan
            if lag == 1:
                row['switch_rate'] = float(np.mean(labels[t-1, common] != current[common])) if common.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@threadpool_limits.wrap(limits=1)
def analyze_evolution(fx: pd.DataFrame, *, k_values=(2, 3, 4, 5, 6), seed=42,
                      summary_start=None, audit_initialization=True):
    """Use all available causal states; retain a fixed roster and explicit gaps."""
    if not isinstance(fx.index, pd.PeriodIndex) or fx.index.freqstr != 'M':
        raise ValueError('FX must have a monthly calendar')
    if not fx.index.equals(pd.period_range(fx.index.min(), fx.index.max(), freq='M')):
        raise ValueError('A complete ascending calendar is required')
    fx = fx.reindex(columns=sorted(fx.columns))
    features = make_rolling_features(fx, 12)
    good = features.notna().all(axis=1).groupby(level='period').sum()
    start = good.index[good >= max(k_values)].min()
    if pd.isna(start):
        raise ValueError('No initial cross section can support all requested K')
    features = features.loc[features.index.get_level_values('period') >= start]
    dates = pd.period_range(start, fx.index.max(), freq='M'); currencies = fx.columns.tolist()
    paths, metric_frames, init_rows = {}, [], []
    returns = 100*np.log(fx).diff()
    full_roster = returns.notna().rolling(60, min_periods=60).sum().eq(60).all(axis=1)
    common_start = str(full_roster.index[full_roster][0]) if full_roster.any() else None
    summary_start = summary_start or common_start or str(dates[0])
    for k in k_values:
        result = run_cross_sections(features, n_clusters=k, windows=(1, 6, 12), seed=seed,
                                    baselines=False, burn_in=12)
        for method in ('r2k_w1', 'r2k_w6', 'r2k_w12'):
            part = result.labels.loc[result.labels.model.eq(method)]
            table = part.pivot(index='period', columns='currency', values='cluster')
            table.index = pd.PeriodIndex(table.index, freq='M')
            labels = table.reindex(index=dates, columns=currencies).fillna(-1).to_numpy(dtype=np.int16)
            paths[f'{method}:k{k}'] = labels
        for method in ('returns_average', 'adjusted_average'):
            labels = np.full((len(dates), len(currencies)), -1, dtype=np.int16)
            previous = labels[0].copy()
            for t, date in enumerate(dates):
                history = returns.loc[:date].tail(60)
                eligible = history.notna().all(axis=0) if len(history) == 60 else pd.Series(False, index=currencies)
                if eligible.sum() <= k:
                    continue
                _, _, tree = _geometry(history.loc[:, eligible], method)
                current = np.full(len(currencies), -1, dtype=np.int16)
                current[eligible] = cut_tree(tree, n_clusters=[k]).ravel()
                current = align_partition(current, previous, k)
                labels[t] = previous = current
            paths[f'{method}:k{k}'] = labels
    for key, labels in paths.items():
        method, k = key.split(':k'); metric_frames.append(describe_path(labels, dates, method, int(k)))
    metrics = pd.concat(metric_frames, ignore_index=True)
    selected = metrics.loc[metrics.period.ge(summary_start)]
    summary = selected.groupby(['method', 'k']).agg(
        months=('period', 'size'), mean_assets=('n_assets', 'mean'),
        mean_ari_1m=('ari_lag1', 'mean'), mean_ari_12m=('ari_lag12', 'mean'),
        mean_ari_60m=('ari_lag60', 'mean'), mean_switch_rate=('switch_rate', 'mean'),
        mean_largest_share=('largest_share', 'mean'), mean_effective_groups=('effective_groups', 'mean'),
        mean_singleton_fraction=('singleton_fraction', 'mean'),
        mean_pair_together=('pair_together_fraction', 'mean')).reset_index()
    pair_rows, currency_rows = [], []
    scope = dates >= pd.Period(summary_start, 'M')
    for key, path in paths.items():
        method, k = key.split(':k'); k = int(k); a = path[scope]; valid = a >= 0
        for i, c in enumerate(currencies):
            singleton = sum((row == row[i]).sum() == 1 for row in a if row[i] >= 0)
            currency_rows.append({'method': method, 'k': k, 'currency': c, 'observed_months': int(valid[:, i].sum()),
                                  'singleton_months': int(singleton),
                                  'singleton_fraction': singleton/valid[:, i].sum() if valid[:, i].any() else np.nan})
            for j in range(i+1, len(currencies)):
                both = valid[:, i] & valid[:, j]; same = both & (a[:, i] == a[:, j])
                pair_rows.append({'method': method, 'k': k, 'currency_i': c, 'currency_j': currencies[j],
                                  'observed_months': int(both.sum()), 'together_months': int(same.sum()),
                                  'frequency': same.sum()/both.sum() if both.any() else np.nan})
    # Main K=3: replay initial seeds and first-fit effort, without changing the reference path.
    if audit_initialization and 3 in k_values:
        reference = paths['r2k_w6:k3']
        for n_init in (1, 10):
            for initial_seed in (0, 42, 99):
                alternative = run_cross_sections(features, n_clusters=3, windows=(6,), seed=initial_seed,
                                                 n_init=n_init, baselines=False, burn_in=12)
                a = alternative.labels.pivot(index='period', columns='currency', values='cluster')
                a.index = pd.PeriodIndex(a.index, freq='M')
                a = a.reindex(index=dates, columns=currencies).fillna(-1).to_numpy(dtype=np.int16)
                for t in np.flatnonzero(scope):
                    common = (a[t] >= 0) & (reference[t] >= 0)
                    init_rows.append({'kind': 'initial_seed', 'period': str(dates[t]), 'seed': initial_seed,
                                      'n_init': n_init, 'n_common': int(common.sum()),
                                      'ari_to_continuous': float(adjusted_rand_score(reference[t, common], a[t, common]))})
        # Annual endpoints keep this audit modest. Restart means replay 60 origin-time states;
        # their trailing features still use the preceding 12 actual FX returns.
        for t in range(59, len(dates)):
            if dates[t].month != 12 and t != len(dates)-1:
                continue
            subset = features.loc[(features.index.get_level_values('period') >= dates[t-59]) &
                                  (features.index.get_level_values('period') <= dates[t])]
            replay = run_cross_sections(subset, n_clusters=3, windows=(6,), seed=seed, baselines=False)
            final = replay.labels.loc[replay.labels.period.eq(str(dates[t]))].set_index('currency').cluster.reindex(currencies)
            common = final.notna().to_numpy() & (reference[t] >= 0)
            init_rows.append({'kind': 'restart_60m', 'period': str(dates[t]), 'seed': seed, 'n_init': 10,
                              'n_common': int(common.sum()),
                              'ari_to_continuous': float(adjusted_rand_score(reference[t, common], final.to_numpy()[common]))})
    metadata = {'paper_reference': PAPER_REFERENCE, 'dates': dates.astype(str).tolist(), 'currencies': currencies, 'names': NAMES,
                'k_values': list(k_values), 'observation_window': 60, 'feature_lookback': 12,
                'centroid_windows': [1, 6, 12], 'initialization': str(start), 'summary_start': summary_start,
                'common_roster_start': common_start,
                'summary_end': str(dates[-1]), 'seed': seed, 'n_init': 10,
                'interpretation': 'Descriptive temporal analysis, including previously inspected post-2019 data; not forecast validation',
                'missing_label': -1, 'eligibility': 'R2K: 12 consecutive FX returns; correlation refits: 60 consecutive returns',
                'label_ids': 'R2K persistent algorithm identities; correlation maximum-overlap matching for display only',
                'restart_protocol': 'K3 W6; annual and final endpoints; 60 origin-time cross sections with causal pre-window feature warm-up'}
    return paths, metadata, {'evolution_metrics': metrics, 'evolution_summary': summary,
                            'evolution_pairs': pd.DataFrame(pair_rows), 'evolution_currency': pd.DataFrame(currency_rows),
                            'evolution_initialization': pd.DataFrame(init_rows)}


def write_evolution_diagnostics(folder: Path, *, k_values=(2, 3, 4, 5, 6), seed=42) -> list[str]:
    folder = Path(folder)
    panel = pd.read_parquet(folder/'monthly_local.parquet', columns=['fx_level'])
    fx = panel.fx_level.unstack('currency')
    paths, metadata, tables = analyze_evolution(fx, k_values=k_values, seed=seed)
    np.savez_compressed(folder/'evolution_labels.npz', **paths)
    (folder/'evolution_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for name, frame in tables.items():
        frame.to_csv(folder/f'{name}.csv', index=False)
    return OUTPUTS.copy()
