"""Chronological R2K-Means experiments and label-aware clustering diagnostics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT
from fx_forecasting.data.rolling_features import (
    FEATURES, load_selected_fx, make_rolling_features, normalize_cross_section,
)
from fx_forecasting.models.rolling_kmeans import RobustRollingKMeans, is_lloyd_fixed_point

PAPER_URL = "https://ssrn.com/abstract=4729393"


@dataclass
class RollingExperiment:
    labels: pd.DataFrame
    centroids: pd.DataFrame
    metrics: pd.DataFrame
    normalization: pd.DataFrame
    coverage: pd.DataFrame


def partition_metrics(current: pd.Series, previous: pd.Series | None) -> dict:
    """Switches and ARI on assets present in consecutive observation periods.

    ARI is permutation invariant; switches are not. Entrants/exits are excluded
    from the switch denominator and are counted separately in coverage.
    """
    if previous is None:
        return {"n_common": 0, "n_switches": 0, "switch_rate": np.nan, "ari": np.nan}
    common = previous.index.intersection(current.index)
    if not len(common):
        return {"n_common": 0, "n_switches": 0, "switch_rate": np.nan, "ari": np.nan}
    a, b = previous.loc[common], current.loc[common]
    switches = int(a.ne(b).sum())
    return {"n_common": len(common), "n_switches": switches,
            "switch_rate": switches / len(common),
            "ari": float(adjusted_rand_score(a, b)) if len(common) >= 2 else np.nan}


def safe_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    n_labels = len(np.unique(labels))
    if not 1 < n_labels < len(values):
        return np.nan
    return float(silhouette_score(values, labels))


def align_centers(centers: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian baseline: match new centers to last period's aligned centers."""
    distances = np.sum((previous[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    old_id, new_id = linear_sum_assignment(distances)
    new_to_old = np.empty(len(centers), dtype=int)
    new_to_old[new_id] = old_id
    aligned = np.empty_like(centers)
    aligned[old_id] = centers[new_id]
    return aligned, new_to_old


@threadpool_limits.wrap(limits=1)
def run_cross_sections(
    features: pd.DataFrame, *, n_clusters: int = 3, windows: tuple[int, ...] = (1, 6, 12),
    seed: int = 42, n_init: int = 10, burn_in: int = 12, baselines: bool = True,
    extra_baselines: bool = False, scaling: str = "cross_sectional",
) -> RollingExperiment:
    """Run normalized, finite cross sections in order without revising history.

    Input is indexed by (monthly period, currency); all feature columns are used.
    Cross sections below K distinct locations fail explicitly instead of silently
    stretching the centroid window over missing months. No forward targets enter.
    ``burn_in`` excludes the first N updates from summaries, not from fitting.
    """
    if not isinstance(features.index, pd.MultiIndex) or features.index.names != ["period", "currency"]:
        raise ValueError("Expected MultiIndex names ('period', 'currency')")
    if not features.index.is_unique or features.empty:
        raise ValueError("Feature rows must be nonempty and unique")
    periods = features.index.get_level_values("period").unique().sort_values()
    if not isinstance(periods, pd.PeriodIndex) or periods.freqstr != "M":
        raise ValueError("Periods must be monthly")
    if not periods.equals(pd.period_range(periods.min(), periods.max(), freq="M")):
        raise ValueError("Every calendar month must be represented")
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("windows must be nonempty and unique")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    if scaling not in ("cross_sectional", "initial_cross_section"):
        raise ValueError("Unknown scaling convention")
    models = {f"r2k_w{w}": RobustRollingKMeans(n_clusters, w, random_state=seed, n_init=n_init)
              for w in windows}
    label_rows, center_rows, metric_rows, norm_rows, coverage_rows = [], [], [], [], []
    previous_labels, previous_centers = {}, {}
    previous_assets: set[str] = set()
    rng = np.random.default_rng(seed)
    initial_stats = None
    for step, period in enumerate(periods, 1):
        raw = features.xs(period, level="period").sort_index()
        valid = np.isfinite(raw.to_numpy(dtype=float)).all(axis=1)
        eligible = raw.loc[valid]
        if len(np.unique(eligible.to_numpy(), axis=0)) < n_clusters:
            raise ValueError(f"{period}: fewer than K={n_clusters} distinct eligible assets")
        cross, stats = normalize_cross_section(eligible)
        if initial_stats is None:
            initial_stats = stats.copy()
        if scaling == "initial_cross_section":
            # Causal fixed coordinates: fit once on the first update, never on
            # the full history or a later calibration/evaluation window.
            stats = initial_stats.copy()
            cross = (eligible - stats["mean"]) / stats["scale"]
        stats = stats.rename_axis("feature").reset_index()
        stats["period"] = str(period)
        norm_rows.extend(stats.to_dict("records"))
        assets = set(cross.index)
        coverage_rows.append({"period": str(period), "n_roster": len(raw), "n_eligible": len(cross),
                              "n_missing": int((~valid).sum()),
                              "n_entrants": len(assets - previous_assets),
                              "n_exits": len(previous_assets - assets),
                              "excluded_currencies": ",".join(raw.index[~valid])})
        previous_assets = assets
        candidates = {}
        for name, model in models.items():
            result = model.update(cross)
            candidates[name] = (result.labels, result.lloyd_labels, result.centers,
                                result.inertia, result.n_iter)
        if baselines:
            # A new reproducible draw each period represents independent refits.
            month_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=month_seed,
                        algorithm="lloyd", tol=0.0, max_iter=300).fit(cross)
            unfinished = km.n_iter_ >= 300 and not is_lloyd_fixed_point(
                cross.to_numpy(), km.cluster_centers_, km.labels_,
            )
            if unfinished or len(np.unique(km.labels_)) != n_clusters:
                raise ValueError(f"{period}: baseline did not converge to K nonempty clusters")
            current = pd.Series(km.labels_, index=cross.index)
            candidates["kmeans_raw"] = (current, current, km.cluster_centers_, km.inertia_, km.n_iter_)
            if "kmeans_aligned" in previous_centers:
                aligned, mapping = align_centers(km.cluster_centers_, previous_centers["kmeans_aligned"])
            else:
                order = np.argsort(np.bincount(km.labels_, minlength=n_clusters), kind="stable")
                aligned, mapping = km.cluster_centers_[order], np.argsort(order)
            relabeled = pd.Series(mapping[km.labels_], index=cross.index)
            candidates["kmeans_aligned"] = (relabeled, relabeled, aligned, km.inertia_, km.n_iter_)
        if extra_baselines:
            if "kmeans_warm" in previous_centers:
                init = previous_centers["kmeans_warm"]
                starts = 1
            else:
                # Identical first-period solution and identities to R2K.
                init = next(iter(candidates.values()))[2]
                starts = 1
            warm = KMeans(n_clusters=n_clusters, init=init, n_init=starts,
                          random_state=seed, algorithm="lloyd", tol=0.0, max_iter=300).fit(cross)
            unfinished = warm.n_iter_ >= 300 and not is_lloyd_fixed_point(
                cross.to_numpy(), warm.cluster_centers_, warm.labels_,
            )
            if unfinished or len(np.unique(warm.labels_)) != n_clusters:
                raise ValueError(f"{period}: warm-start baseline did not converge to K clusters")
            warm_labels = pd.Series(warm.labels_, index=cross.index)
            candidates["kmeans_warm"] = (warm_labels, warm_labels, warm.cluster_centers_, warm.inertia_, warm.n_iter_)
        for name, (labels, lloyd_labels, centers, inertia, n_iter) in candidates.items():
            changes = partition_metrics(labels, previous_labels.get(name))
            shift = np.nan
            if name in previous_centers:
                shift = float(np.linalg.norm(centers - previous_centers[name], axis=1).mean())
            metric_rows.append({"period": str(period), "model": name, "step": step,
                                "after_burn_in": step > burn_in, "n_assets": len(cross),
                                "n_active_labels": labels.nunique(),
                                "min_cluster_size": int(labels.value_counts().reindex(range(n_clusters), fill_value=0).min()),
                                "silhouette": safe_silhouette(cross.to_numpy(), labels.to_numpy()),
                                "lloyd_inertia_per_asset": float(inertia / len(cross)),
                                "centroid_displacement": shift, "lloyd_iterations": n_iter,
                                "rolling_disagreement": float(labels.ne(lloyd_labels).mean()), **changes})
            for currency in cross.index:
                label_rows.append({"period": str(period), "currency": currency, "model": name,
                                   "cluster": int(labels.loc[currency]),
                                   "lloyd_cluster": int(lloyd_labels.loc[currency])})
            for cluster, center in enumerate(centers):
                center_rows.append({"period": str(period), "model": name, "cluster": cluster,
                                    **dict(zip(cross.columns, map(float, center)))})
            previous_labels[name] = labels.copy()
            previous_centers[name] = centers.copy()
    return RollingExperiment(pd.DataFrame(label_rows), pd.DataFrame(center_rows),
                             pd.DataFrame(metric_rows), pd.DataFrame(norm_rows), pd.DataFrame(coverage_rows))


def summarize_experiment(result: RollingExperiment, evaluation_start: str = "2010-01") -> pd.DataFrame:
    """Monthly mean diagnostics after burn-in; switch rates use asset-weighting."""
    rows = result.metrics.loc[result.metrics.after_burn_in & result.metrics.period.ge(evaluation_start)]
    summaries = []
    for model, group in rows.groupby("model", sort=True):
        denominator = int(group.n_common.sum())
        summaries.append({"model": model, "n_months": len(group),
                          "mean_assets": group.n_assets.mean(),
                          "switch_rate": group.n_switches.sum() / denominator if denominator else np.nan,
                          "mean_ari": group.ari.mean(), "mean_silhouette": group.silhouette.mean(),
                          "mean_centroid_displacement": group.centroid_displacement.mean(),
                          "mean_min_cluster_size": group.min_cluster_size.mean(),
                          "fraction_months_min_cluster_lt5": group.min_cluster_size.lt(5).mean(),
                          "mean_rolling_disagreement": group.rolling_disagreement.mean(),
                          "mean_lloyd_inertia_per_asset": group.lloyd_inertia_per_asset.mean()})
    return pd.DataFrame(summaries)


def calibration_grid(features: pd.DataFrame, *, cutoff: str = "2009-12",
                     k_values: tuple[int, ...] = (2, 3, 4, 5, 6), seed: int = 42) -> pd.DataFrame:
    """W=1 silhouette comparison entirely before the evaluation period.

    This is a diagnostic, not an automatic K selector. W>1 boundaries need not
    be convex, so their silhouettes are not used to tune the rolling window.
    """
    historical = features.loc[features.index.get_level_values("period") <= pd.Period(cutoff, "M")]
    if historical.empty:
        raise ValueError("No calibration observations at or before cutoff")
    rows = []
    for k in k_values:
        result = run_cross_sections(historical, n_clusters=k, windows=(1,), seed=seed, baselines=False)
        summary = summarize_experiment(result, str(historical.index.get_level_values("period").min()))
        if summary.empty:
            raise ValueError("Calibration window must exceed the 12-period burn-in")
        rows.append({"k": k, "calibration_cutoff": cutoff, **summary.iloc[0].to_dict()})
    return pd.DataFrame(rows).drop(columns="model")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_bis_experiment(
    *, ingested_dir: Path = INGESTED_DIR,
    sample_path: Path = PROJECT_ROOT / "config/research_sample.json",
    output_dir: Path = PROJECT_ROOT / "reports/clustering/r2kmeans",
    start: str = "1995-01", end: str | None = None, evaluation_start: str = "2010-01",
    n_clusters: int = 3, windows: tuple[int, ...] = (1, 6, 12), lookback: int = 12,
    seed: int = 42, tune_k: bool = False,
) -> tuple[RollingExperiment, pd.DataFrame, dict]:
    """Read the local BIS snapshot, run an experiment, and save audit artifacts."""
    fx, metadata, cfg = load_selected_fx(ingested_dir, sample_path)
    stop = pd.Period(end or cfg["monthly_window"]["end"], "M")
    first = pd.Period(start, "M")
    if first < fx.index.min() or stop > fx.index.max() or first > stop:
        raise ValueError("Requested experiment window lies outside the configured FX calendar")
    raw_features = make_rolling_features(fx, lookback)
    months = raw_features.index.get_level_values("period")
    features = raw_features.loc[(months >= first) & (months <= stop)]
    calibration = calibration_grid(features, cutoff=str(pd.Period(evaluation_start, "M") - 1), seed=seed) if tune_k else None
    result = run_cross_sections(features, n_clusters=n_clusters, windows=windows, seed=seed)
    summary = summarize_experiment(result, evaluation_start)
    output_dir = Path(output_dir)
    # Never allow experiment exports to overwrite a source directory.
    for protected in (Path(ingested_dir), PROJECT_ROOT / "data/raw"):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("Experiment outputs must be outside raw and ingested source directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / "features.parquet")
    result.labels.to_parquet(output_dir / "labels.parquet", index=False)
    result.centroids.to_parquet(output_dir / "centroids.parquet", index=False)
    for name in ("metrics", "normalization", "coverage"):
        getattr(result, name).to_csv(output_dir / f"{name}.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    metadata.to_csv(output_dir / "currency_metadata.csv")
    if calibration is not None:
        calibration.to_csv(output_dir / "calibration_grid.csv", index=False)
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow")
    manifest = {
        "paper": {"title": "Robust Rolling K-Means (R2K-Means): an Updateable Nonlinear K-Means Clustering Methodology for Financial Time Series",
                  "authors": "Holmes, Hirsa, Klinkert, Malhotra", "version": "2024-03-06", "url": PAPER_URL},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"start": str(first), "end": str(stop), "evaluation_start": evaluation_start,
                     "n_clusters": n_clusters, "windows": list(windows), "lookback": lookback,
                     "seed": seed, "n_init": 10, "max_iter": 300, "tol": 0.0, "burn_in": 12,
                     "feature_columns": list(FEATURES), "scaling": "current cross-sectional z-score, ddof=0",
                     "calibration_grid": tune_k},
        "sources": {str(path.name): {"sha256": _sha256(path)} for path in
                    (Path(ingested_dir) / "observations.parquet", Path(sample_path))},
        "implementation_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in
                                  (Path(__file__), PROJECT_ROOT / "src/fx_forecasting/models/rolling_kmeans.py",
                                   PROJECT_ROOT / "src/fx_forecasting/data/rolling_features.py")},
        "environment": {"python": platform.python_version(),
                        **{name: importlib.metadata.version(name) for name in packages}},
        "limitations": ["FX adaptation; not a replication of the paper's mutual-fund dataset",
                        "Revised histories and retrospective regime exclusions; not real-time vintages",
                        "Cross-sectional normalization is an explicit choice; scaling unspecified in paper",
                        "K=3 is a prespecified illustration, not an established optimum",
                        "See docs/r2kmeans_audit.md before using these illustrative groups for forecasting",
                        "Rolling W counts consecutive updates; periods with fewer than K distinct assets fail",
                        "No forecast accuracy or neural-network claim follows from clustering metrics"],
    }
    manifest["outputs"] = {p.name: {"sha256": _sha256(p)} for p in sorted(output_dir.iterdir())
                           if p.is_file() and p.name in {"features.parquet", "labels.parquet", "centroids.parquet",
                               "metrics.csv", "normalization.csv", "coverage.csv", "summary.csv", "currency_metadata.csv"}}
    if calibration is not None:
        manifest["outputs"]["calibration_grid.csv"] = {"sha256": _sha256(output_dir / "calibration_grid.csv")}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result, summary, manifest
