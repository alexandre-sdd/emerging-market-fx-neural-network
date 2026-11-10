"""Training-only group diagnostics and exploratory factor-reference comparisons.

These assess descriptive structure, not forecast performance or an optimal K.
Existing correlation geometries and twelve-month resampling are reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from fx_forecasting.models.currency_clustering import _block_indices, _geometry

CORRELATION_METHODS = ("returns_average", "returns_ward", "adjusted_average", "adjusted_ward")
OUTPUTS = ["group_quality.csv", "group_recovery.csv", "bootstrap_coassignment.npz",
           "factor_reference.csv", "factor_spectrum.csv", "theory_metadata.json"]


def _partition(tree: np.ndarray, k: int) -> np.ndarray:
    return cut_tree(tree, n_clusters=[k]).ravel()


def _best_jaccard(reference: np.ndarray, labels: np.ndarray) -> float:
    """Compare a fixed currency set with its best matching resampled group."""
    return max(np.count_nonzero(reference & (labels == label)) /
               np.count_nonzero(reference | (labels == label)) for label in np.unique(labels))


def _shift_residuals(residual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent circular shifts preserve each residual's values and spectrum."""
    shifts = rng.integers(0, len(residual), size=residual.shape[1])
    return np.column_stack([np.roll(residual[:, j], shift) for j, shift in enumerate(shifts)])


@threadpool_limits.wrap(limits=1)
def analyze_structure(
    returns: pd.DataFrame, memberships: pd.DataFrame, *, fit_start="2011-01", fit_end="2019-12",
    bootstrap_reps: int = 50, reference_reps: int = 200, seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict]:
    """Describe frozen groups and compare silhouettes to rank-1/rank-3 references.

    PCA is fitted to standardized training returns. For each fixed rank, retain
    its fitted common component, independently circular-shift the residual
    columns, and transform back to return units before rerunning each grouping
    transformation. This destroys residual alignment while retaining the fitted
    factor paths/loadings and residual marginal distributions/serial spectra.
    A rank-3 reference can retain regional factors and group-like loading
    patterns. It asks whether those continuous exposures suffice, rather than
    asserting that its samples contain no cluster structure. Percentiles are
    conditional sensitivity diagnostics, not calibrated significance tests.
    """
    if not isinstance(returns.index, pd.PeriodIndex) or returns.index.freqstr != "M":
        raise ValueError("Structure diagnostics require monthly return periods")
    if bootstrap_reps < 0 or reference_reps < 1:
        raise ValueError("Bootstrap counts must be nonnegative and reference counts positive")
    x = returns.loc[fit_start:fit_end].reindex(columns=sorted(returns.columns)).copy()
    if len(x) < 12 or x.shape[1] < 4 or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Need at least twelve complete months and four nonconstant currencies")
    if (x.std(ddof=0) <= np.finfo(float).eps).any():
        raise ValueError("Constant return histories are not supported")
    currencies = x.columns.tolist()
    index = {currency: i for i, currency in enumerate(currencies)}
    corr = x.corr().to_numpy()
    methods = [m for m in CORRELATION_METHODS if m in memberships.method.unique()]
    if not methods:
        raise ValueError("No correlation partitions available for structure diagnostics")
    ks = sorted(memberships.loc[memberships.method.isin(methods), "k"].unique().tolist())
    baseline, geometry, quality = {}, {}, []
    for method in methods:
        _, distance, tree = _geometry(x, method)
        geometry[method] = (distance, tree)
        for k in ks:
            rows = memberships.loc[memberships.method.eq(method) & memberships.k.eq(k)]
            if set(rows.currency) != set(currencies) or rows.currency.duplicated().any():
                raise ValueError("Memberships must cover the fixed currency roster exactly once")
            labels = rows.set_index("currency").reindex(currencies).cluster.to_numpy(dtype=int)
            if adjusted_rand_score(labels, _partition(tree, k)) != 1:
                raise ValueError("Saved memberships disagree with the requested training geometry")
            baseline[(method, k)] = labels
    for (method, k), rows in memberships.groupby(["method", "k"], sort=True):
        for cluster, group in rows.groupby("cluster", sort=True):
            ids = np.array([index[currency] for currency in group.currency])
            other = np.array([j for j in range(len(currencies)) if j not in ids])
            within = corr[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)]
            between = corr[np.ix_(ids, other)]
            quality.append({"method": method, "k": k, "cluster": cluster, "n_currencies": len(ids),
                            "representative": group.representative.iloc[0],
                            "within_raw_correlation": float(within.mean()) if len(within) else np.nan,
                            "outside_raw_correlation": float(between.mean()),
                            "mean_silhouette": float(group.silhouette.mean()),
                            "negative_silhouettes": int(group.silhouette.lt(0).sum())})

    recovery = {key: {label: [] for label in np.unique(labels)} for key, labels in baseline.items()}
    coassignment = {key: np.zeros((len(currencies), len(currencies))) for key in baseline}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_reps):
        sampled = x.iloc[_block_indices(rng, len(x))]
        for method in methods:
            _, _, tree = _geometry(sampled, method)
            for k in ks:
                key = (method, k); labels = _partition(tree, k)
                coassignment[key] += labels[:, None] == labels[None, :]
                for label in recovery[key]:
                    recovery[key][label].append(_best_jaccard(baseline[key] == label, labels))
    group_recovery = []
    for (method, k), labels in recovery.items():
        for label, values in labels.items():
            group_recovery.append({"method": method, "k": k, "cluster": label,
                                   "mean_jaccard": np.mean(values) if values else np.nan,
                                   "p10_jaccard": np.quantile(values, .1) if values else np.nan,
                                   "bootstrap_reps": bootstrap_reps})
    coassignment = {f"{method}:k{k}": (values/bootstrap_reps).astype(np.float32)
                    for (method, k), values in coassignment.items()} if bootstrap_reps else {}

    # Continuous factor references preserve heterogeneous loadings, unlike
    # subtracting the same median from every currency or an iid white-noise null.
    mean, scale = x.mean().to_numpy(), x.std(ddof=0).to_numpy()
    z = (x.to_numpy()-mean)/scale
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    spectrum = []
    for name, values in (("raw_standardized", z), ("median_adjusted_standardized", _geometry(x, "adjusted_average")[0].to_numpy())):
        variance = np.linalg.svd(values, compute_uv=False)**2
        shares = variance/variance.sum()
        spectrum += [{"representation": name, "component": j+1, "variance_share": share,
                      "cumulative_share": shares[:j+1].sum()} for j, share in enumerate(shares)]
    references = []
    for rank in (1, 3):
        common = (u[:, :rank]*singular[:rank])@vt[:rank]
        residual = z-common
        rng = np.random.default_rng(seed+1000+rank)
        simulated = {(method, k): [] for method in methods for k in ks}
        for _ in range(reference_reps):
            sample = pd.DataFrame((common+_shift_residuals(residual, rng))*scale+mean,
                                  index=x.index, columns=x.columns)
            for method in methods:
                _, distance, tree = _geometry(sample, method)
                for k in ks:
                    simulated[(method, k)].append(silhouette_score(distance, _partition(tree, k), metric="precomputed"))
        for (method, k), values in simulated.items():
            distance = geometry[method][0]
            observed = silhouette_score(distance, baseline[(method, k)], metric="precomputed")
            references.append({"method": method, "k": k, "factor_rank": rank, "observed_silhouette": observed,
                               "reference_p05": np.quantile(values, .05), "reference_median": np.median(values),
                               "reference_p95": np.quantile(values, .95), "reference_reps": reference_reps})
    metadata = {"fit_start": str(x.index.min()), "fit_end": str(x.index.max()), "n_months": len(x),
                "currencies": currencies, "methods": methods, "k_values": ks, "seed": seed,
                "bootstrap_reps": bootstrap_reps, "block_months": 12, "reference_reps": reference_reps,
                "factor_ranks": [1, 3], "reference": "fixed PCA factors plus independently circular-shifted residuals, reconstructed in original return units",
                "interpretation": "Exploratory training-only reference ranges; not a gap statistic, formal significance test, optimal K or forecast-performance estimate"}
    return {"group_quality": pd.DataFrame(quality), "group_recovery": pd.DataFrame(group_recovery),
            "factor_reference": pd.DataFrame(references), "factor_spectrum": pd.DataFrame(spectrum)}, coassignment, metadata


def write_structure_diagnostics(folder: Path, *, reference_reps=200, bootstrap_reps=50, seed=42) -> list[str]:
    """Reuse the canonical saved return panel; add no source loaders or trainer."""
    folder = Path(folder)
    returns = pd.read_parquet(folder/"monthly_local.parquet", columns=["fx_change_1m"]).fx_change_1m.unstack("currency")
    settings = json.loads((folder/"settings.json").read_text())
    tables, coassignment, metadata = analyze_structure(
        returns, pd.read_csv(folder/"memberships.csv"), fit_start=settings["fit_start"], fit_end=settings["fit_end"],
        reference_reps=reference_reps, bootstrap_reps=bootstrap_reps, seed=seed,
    )
    for name, frame in tables.items(): frame.to_csv(folder/f"{name}.csv", index=False)
    np.savez_compressed(folder/"bootstrap_coassignment.npz", **coassignment)
    (folder/"theory_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return OUTPUTS.copy()
