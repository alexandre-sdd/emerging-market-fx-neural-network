"""Training-cutoff currency groups for synchronized multivariate forecasting.

The representations deliberately answer different questions. Correlation groups
same-month movements, DTW permits phase shifts, and R2K groups current moments.
Silhouette values therefore describe their own geometry, not forecast quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from threadpoolctl import threadpool_limits

from fx_forecasting.data.rolling_features import make_rolling_features, normalize_cross_section
from fx_forecasting.models.rolling_kmeans import RobustRollingKMeans


@dataclass
class ClusterComparison:
    scores: pd.DataFrame
    memberships: pd.DataFrame
    stability: pd.DataFrame
    distances: dict[str, pd.DataFrame]
    histories: dict[str, pd.DataFrame]


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Comparison histories require finite values on the common calendar")
    scale = frame.std(ddof=0)
    if scale.le(np.finfo(float).eps).any():
        names = scale.index[scale.le(np.finfo(float).eps)].tolist()
        raise ValueError(f"Constant currency histories cannot define correlation/shape: {names}")
    return (frame - frame.mean()) / scale


def _canonical(labels: np.ndarray, currencies: list[str]) -> np.ndarray:
    order = sorted(np.unique(labels), key=lambda group: min(np.asarray(currencies)[labels == group]))
    mapping = {group: i for i, group in enumerate(order)}
    return np.array([mapping[group] for group in labels], dtype=int)


def _correlation_distance(history: pd.DataFrame) -> np.ndarray:
    correlation = np.clip(history.corr().to_numpy(), -1, 1)
    distance = np.sqrt(np.maximum(0, 2 * (1 - correlation)))
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return distance


def _geometry(history: pd.DataFrame, method: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    values = history.sub(history.median(axis=1), axis=0) if method.startswith("adjusted_") else history
    standardized = _standardize(values)
    if method == "dtw":
        try:
            from tslearn.metrics import cdist_dtw
        except ImportError as error:
            raise ImportError('DTW comparison requires python -m pip install -e ".[clustering]"') from error
        distance = cdist_dtw(standardized.to_numpy().T[:, :, None],
                             global_constraint="sakoe_chiba", sakoe_chiba_radius=2, n_jobs=1)
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)
        return standardized, distance, None
    distance = _correlation_distance(standardized)
    tree = linkage(standardized.to_numpy().T, method="ward") if method.endswith("_ward") else linkage(
        squareform(distance, checks=False), method="average",
    )
    return standardized, distance, tree


def _fit(standardized: pd.DataFrame, method: str, k: int, tree: np.ndarray | None,
         seed: int, *, initial: bool = False) -> tuple[np.ndarray, float]:
    if method != "dtw":
        return cut_tree(tree, n_clusters=[k]).ravel(), np.nan
    from tslearn.clustering import TimeSeriesKMeans

    # Main comparison reproduces the modest multi-start protocol of notebook 02.
    # Stability fits retain three starts but one seed, avoiding a single-start
    # initialization artifact while keeping a 50-replicate audit tractable.
    seeds = (seed, seed + 1, seed + 2) if initial else (seed,)
    candidates = []
    for state in seeds:
        model = TimeSeriesKMeans(
            n_clusters=k, metric="dtw", metric_params={
                "global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 2,
            }, n_init=3, max_iter=20, max_iter_barycenter=10,
            random_state=state, n_jobs=1,
        ).fit(standardized.to_numpy().T[:, :, None])
        if len(np.unique(model.labels_)) != k:
            raise ValueError(f"DTW fit produced fewer than K={k} nonempty clusters")
        candidates.append((float(model.inertia_), model.labels_.copy()))
    inertia, labels = min(candidates, key=lambda pair: pair[0])
    return labels, inertia


def _summarize(method: str, k: int, labels: np.ndarray, distance: np.ndarray,
               currencies: list[str], raw_correlation: np.ndarray,
               start: str, end: str, months: int, representation: str,
               inertia: float = np.nan) -> tuple[dict, list[dict]]:
    labels = _canonical(labels, currencies)
    n_labels = len(np.unique(labels))
    silhouette = float(silhouette_score(distance, labels, metric="precomputed")) if 1 < n_labels < len(labels) else np.nan
    sample_silhouette = silhouette_samples(distance, labels, metric="precomputed") if 1 < n_labels < len(labels) else np.full(len(labels), np.nan)
    sizes = np.bincount(labels, minlength=k)
    upper = np.triu(np.ones_like(distance, dtype=bool), k=1)
    same = labels[:, None] == labels[None, :]
    within, between = raw_correlation[upper & same], raw_correlation[upper & ~same]
    score = {"method": method, "k": k, "fit_start": start, "fit_end": end,
             "n_months": months, "n_currencies": len(currencies), "representation": representation,
             "silhouette": silhouette, "sizes": json.dumps(sizes.tolist()),
             "min_cluster_size": int(sizes.min()), "max_cluster_size": int(sizes.max()),
             "n_singletons": int((sizes == 1).sum()),
             "within_correlation": float(within.mean()) if len(within) else np.nan,
             "between_correlation": float(between.mean()) if len(between) else np.nan,
             "inertia": inertia}
    rows = []
    for group in range(k):
        members = np.flatnonzero(labels == group)
        if not len(members):
            continue
        representative = members[np.argmin(distance[np.ix_(members, members)].mean(axis=1))]
        for index in members:
            rows.append({"method": method, "k": k, "currency": currencies[index],
                         "cluster": group, "representative": currencies[representative],
                         "channel_index": int(index), "silhouette": float(sample_silhouette[index]),
                         "distance_to_representative": float(distance[index, representative])})
    return score, rows


def _block_indices(rng: np.random.Generator, n: int, length: int = 12) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    return np.concatenate([(start + np.arange(length)) % n for start in starts])[:n]


@threadpool_limits.wrap(limits=1)
def compare_clusterings(
    fx: pd.DataFrame, *, fit_start: str = "2011-01", fit_end: str = "2019-12",
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6), bootstrap_reps: int = 50,
    seed: int = 42, include_dtw: bool = True,
) -> ClusterComparison:
    """Compare frozen groups, using only observations at/before the fit cutoff.

    ``fx`` must already apply the project's launch, peg and source rules. No
    currency is silently dropped or filled. All static methods use the same
    complete monthly returns/calendar. R2K is a reference on the first complete
    12-month moment cross section onward, whose shorter history is disclosed.

    Circular block bootstrap concatenates sampled *observed returns*, never
    recomputing returns across resampled joins. Identical month draws are used
    across static methods. Level histories sample the corresponding level
    months. R2K snapshot references are not bootstrapped because arbitrarily
    concatenating moment states would change the update-time interpretation.
    DTW stability refits use one seed and three starts, while the main fit uses
    three seeds and three starts each; scores record the protocol explicitly.
    """
    if not isinstance(fx.index, pd.PeriodIndex) or fx.index.freqstr != "M":
        raise ValueError("FX requires a monthly PeriodIndex")
    if fx.empty or not fx.index.is_unique or not fx.columns.is_unique or not fx.index.is_monotonic_increasing:
        raise ValueError("FX requires unique currencies and an ascending unique calendar")
    if not all(isinstance(c, str) for c in fx.columns):
        raise ValueError("Currency IDs must be strings")
    if isinstance(bootstrap_reps, bool) or not isinstance(bootstrap_reps, int) or bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be a nonnegative integer")
    ks = tuple(k_values)
    if not ks or len(set(ks)) != len(ks) or any(isinstance(k, bool) or not isinstance(k, int) or not 2 <= k < len(fx.columns) for k in ks):
        raise ValueError("Use unique integer K values between 2 and number of currencies minus one")
    first, last = pd.Period(fit_start, "M"), pd.Period(fit_end, "M")
    if first > last:
        raise ValueError("fit_start must not exceed fit_end")
    currencies = sorted(fx.columns)
    prefix = fx.loc[fx.index <= last, currencies].copy()
    expected_prefix = pd.period_range(prefix.index.min(), last, freq="M") if not prefix.empty else pd.PeriodIndex([], freq="M")
    if prefix.empty or not prefix.index.equals(expected_prefix):
        raise ValueError("FX must contain every month through the fit cutoff")
    positive = prefix.where(np.isfinite(prefix) & prefix.gt(0))
    returns = (100 * np.log(positive).diff()).reindex(pd.period_range(first, last, freq="M"))
    levels = positive.reindex(returns.index)
    if len(returns) < 12 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Comparison requires at least 12 complete common monthly returns, including the preceding FX level")
    raw_correlation = returns.corr().to_numpy()
    methods = ["returns_average", "returns_ward", "adjusted_average", "adjusted_ward", "levels_average"]
    if include_dtw:
        methods.append("dtw")
    descriptions = {"returns_average": "standardized synchronous returns",
                    "returns_ward": "standardized synchronous returns",
                    "adjusted_average": "cross-currency-median-adjusted standardized returns",
                    "adjusted_ward": "cross-currency-median-adjusted standardized returns",
                    "levels_average": "standardized raw FX levels",
                    "dtw": "standardized returns, constrained DTW radius 2"}
    scores, members, stability = [], [], []
    distances, histories = {}, {}
    baseline = {}
    width = min(len(returns), (len(returns) + 12) // 2)
    for method in methods:
        history = levels if method == "levels_average" else returns
        standardized, distance, tree = _geometry(history, method)
        histories[method] = standardized
        distances[method] = pd.DataFrame(distance, index=currencies, columns=currencies)
        sub = [_geometry(history.iloc[:width], method), _geometry(history.iloc[-width:], method)]
        for k in ks:
            labels, inertia = _fit(standardized, method, k, tree, seed, initial=True)
            labels = _canonical(labels, currencies)
            baseline[(method, k)] = labels
            score, membership = _summarize(method, k, labels, distance, currencies, raw_correlation,
                                            str(first), str(last), len(history), descriptions[method], inertia)
            a, _ = _fit(sub[0][0], method, k, sub[0][2], seed)
            b, _ = _fit(sub[1][0], method, k, sub[1][2], seed)
            score["subwindow_ari"] = float(adjusted_rand_score(a, b))
            score["subwindow_1"] = f"{history.index[0]}/{history.index[width-1]}"
            score["subwindow_2"] = f"{history.index[-width]}/{history.index[-1]}"
            score["bootstrap_protocol"] = "circular 12-month blocks; DTW one seed, three starts per refit" if method == "dtw" else "circular 12-month blocks; same month draws across methods"
            scores.append(score)
            members.extend(membership)
            stability.append({"method": method, "k": k, "kind": "subwindow", "replicate": 0,
                              "ari": score["subwindow_ari"], "n_months": width, "block_length": 12, "seed": seed})

    rng = np.random.default_rng(seed)
    for replicate in range(bootstrap_reps):
        indices = _block_indices(rng, len(returns))
        for method in methods:
            history = (levels if method == "levels_average" else returns).iloc[indices]
            standardized, _, tree = _geometry(history, method)
            for k in ks:
                labels, _ = _fit(standardized, method, k, tree, seed + replicate)
                stability.append({"method": method, "k": k, "kind": "bootstrap", "replicate": replicate,
                                  "ari": float(adjusted_rand_score(baseline[(method, k)], labels)),
                                  "n_months": len(indices), "block_length": 12, "seed": seed + replicate})

    # The shared static return calendar is complete; moment histories can still
    # need additional warm-up after an exclusion immediately before that window.
    moments = make_rolling_features(positive)
    pieces = {period: moments.xs(period).reindex(currencies)
              for period in returns.index if period in moments.index.get_level_values("period")}
    complete = [period for period, frame in pieces.items() if np.isfinite(frame.to_numpy()).all()]
    if not complete:
        raise ValueError("No complete 12-month moment cross section for the R2K reference")
    state_start = min(complete)
    state_dates = pd.period_range(state_start, last, freq="M")
    if not all(period in complete for period in state_dates):
        raise ValueError("R2K reference requires a consecutive complete cross-sectional calendar")
    for k in ks:
        for window in (1, 6):
            model = RobustRollingKMeans(k, window, random_state=seed)
            for period in state_dates:
                cross, _ = normalize_cross_section(pieces[period])
                step = model.update(cross)
            method = f"r2k_w{window}"
            distance = cdist(cross.to_numpy(), cross.to_numpy())
            distances[method] = pd.DataFrame(distance, index=currencies, columns=currencies)
            score, membership = _summarize(method, k, step.labels.reindex(currencies).to_numpy(),
                                            distance, currencies, raw_correlation, str(state_start), str(last),
                                            len(state_dates), "frozen rolling mean-change/volatility snapshot", step.inertia)
            score.update({"subwindow_ari": np.nan, "subwindow_1": None, "subwindow_2": None,
                          "bootstrap_protocol": "snapshot reference only; no state-order resampling"})
            scores.append(score)
            members.extend(membership)

    score_table = pd.DataFrame(scores)
    stability_table = pd.DataFrame(stability)
    bootstrap = stability_table.loc[stability_table.kind.eq("bootstrap")]
    aggregates = bootstrap.groupby(["method", "k"]).ari.agg(
        bootstrap_mean_ari="mean", bootstrap_p10_ari=lambda values: values.quantile(.1),
        bootstrap_reps="count",
    ).reset_index()
    score_table = score_table.merge(aggregates, on=["method", "k"], how="left")
    score_table["bootstrap_reps"] = score_table.bootstrap_reps.fillna(0).astype(int)
    return ClusterComparison(score_table, pd.DataFrame(members), stability_table, distances, histories)
