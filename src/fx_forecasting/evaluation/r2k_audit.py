"""Optional historical audit of R2K and scalar row-routing forecasts.

This experiment does not test joint multi-currency neural forecasts. The main
clustering/handoff workflow is independent of this optional scalar Ridge screen.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT
from fx_forecasting.data.rolling_features import load_selected_fx, make_rolling_features
from fx_forecasting.evaluation.grouping_screen import PREDICTORS, make_screen_panel, run_grouping_screen
from fx_forecasting.evaluation.rolling_clustering import (
    RollingExperiment, run_cross_sections, summarize_experiment,
)
from fx_forecasting.models.rolling_kmeans import RobustRollingKMeans


def produced_output_hashes(output_dir: Path, filenames: list[str]) -> dict[str, str]:
    """Hash only outputs explicitly produced by this invocation, not stale files."""
    return {name: hashlib.sha256((Path(output_dir) / name).read_bytes()).hexdigest()
            for name in sorted(set(filenames))}


def cluster_support(result: RollingExperiment, k: int, start: str = "2010-01") -> pd.DataFrame:
    rows = result.labels.loc[result.labels.period.ge(start)]
    counts = rows.groupby(["model", "period", "cluster"]).size().unstack(fill_value=0).reindex(columns=range(k), fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    output = pd.DataFrame({"min_size": counts.min(axis=1), "max_share": shares.max(axis=1),
                           "effective_clusters": np.exp(-(shares * np.log(shares.where(shares.gt(0)))).sum(axis=1)),
                           "n_clusters_lt3": counts.lt(3).sum(axis=1), "n_clusters_lt5": counts.lt(5).sum(axis=1)})
    return output.reset_index().assign(k=k)


def label_series(result: RollingExperiment, method: str) -> pd.Series:
    rows = result.labels.loc[result.labels.model.eq(method)].copy()
    rows["period"] = pd.PeriodIndex(rows.period, freq="M")
    return rows.set_index(["period", "currency"]).cluster.sort_index()


def compare_partitions(a: pd.Series, b: pd.Series, start: str) -> pd.DataFrame:
    common = a.index.intersection(b.index)
    a, b = a.reindex(common), b.reindex(common)
    rows = []
    for period in common.get_level_values("period").unique().sort_values():
        if period < pd.Period(start, "M"):
            continue
        x, y = a.xs(period), b.xs(period)
        rows.append({"period": str(period), "n_common": len(x),
                     "ari": adjusted_rand_score(x, y) if len(x) > 1 else np.nan})
    return pd.DataFrame(rows)


def geometry_audit(features: pd.DataFrame, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    shares, persistence = [], []
    for feature in features:
        values = features[feature].unstack("currency")
        values = values.loc[values.index >= pd.Period(start, "M")]
        centered = values.sub(values.mean(axis=1), axis=0).pow(2)
        share = centered.div(centered.sum(axis=1), axis=0)
        table = share.rename_axis(index="period", columns="currency").stack().rename("variance_share").reset_index()
        table["period"] = table.period.astype(str)
        table["feature"] = feature
        shares.append(table)
        for currency in values:
            # Keep calendar gaps in place; never compress time with dropna().
            x = values[currency]
            pairs = pd.concat([x, x.shift(1)], axis=1).dropna()
            rho = pairs.iloc[:, 0].corr(pairs.iloc[:, 1]) if len(pairs) >= 12 and pairs.std().gt(0).all() else np.nan
            persistence.append({"feature": feature, "currency": currency, "lag1_autocorrelation": rho,
                                "n_adjacent_pairs": len(pairs)})
    return pd.concat(shares, ignore_index=True), pd.DataFrame(persistence)


def stationary_counterexample() -> dict:
    """Reproducible counterexample to a universal unchanged-data guarantee.

    Do not add a no-op shortcut to the reference algorithm: snapping to data is
    precisely what can change the local solution even with an unchanged frame.
    """
    values = np.random.default_rng(2).normal(size=(12, 2))
    frame = pd.DataFrame(values, index=[f"a{i:02}" for i in range(12)], columns=["x", "y"])
    model = RobustRollingKMeans(3, 1, random_state=2, n_init=1)
    first, second = model.update(frame), model.update(frame)
    return {"seed": 2, "n_init": 1, "n_assets": 12, "k": 3,
            "identity_switches": int(first.labels.ne(second.labels).sum()),
            "partition_ari": float(adjusted_rand_score(first.labels, second.labels)),
            "inertia_first": first.inertia, "inertia_second": second.inertia,
            "data": frame.reset_index(names="asset").to_dict("records")}


def available_input_audit(ingested_dir: Path, cfg: dict) -> pd.DataFrame:
    """Inventory exact source matches, without claiming vintage availability."""
    obs = pd.read_parquet(Path(ingested_dir) / "observations.parquet")
    rows = []
    for key, dataset, lag, role in (
        ("fx", "WS_XRU", "0 months", "Current FX-only grouping/screen"),
        ("reer", "WS_EER", "2 months", "Candidate valuation context; not yet in screen"),
        ("policy", "WS_CBPOL", "1 month", "Optional differential; missing sources/gaps"),
        ("cpi", "WS_LONG_CPI", "2 months for derived inflation", "Optional differential; missing sources/gaps"),
    ):
        ids = {c["source_series"][key] for c in cfg["currencies"] if c["source_series"][key]}
        selected = obs.loc[obs.dataset.eq(dataset) & obs.series_id.isin(ids)]
        rows.append({"input": key, "currency_source_matches": selected.series_id.nunique(),
                     "roster_size": len(cfg["currencies"]), "source_rows": len(selected),
                     "nonnull_source_rows": int(selected.value.notna().sum()),
                     "assumed_lag": lag, "role": role})
    for name, dataset, source, lag in (
        ("USD NEER", "WS_EER", "M:N:B:US", "1 month"),
        ("USD credit", "WS_GLI", cfg["macro_availability"]["usd_credit_series"], "2 quarters"),
    ):
        selected = obs.loc[obs.dataset.eq(dataset) & obs.series_id.eq(source)]
        rows.append({"input": name, "currency_source_matches": None, "roster_size": len(cfg["currencies"]),
                     "source_rows": len(selected), "nonnull_source_rows": int(selected.value.notna().sum()),
                     "assumed_lag": lag, "role": "Common across currencies at a date; cannot separate cross section alone"})
    return pd.DataFrame(rows)


def run_methodology_audit(
    *, output_dir: Path = PROJECT_ROOT / "reports/clustering/r2kmeans/audit",
    ingested_dir: Path = INGESTED_DIR,
    sample_path: Path = PROJECT_ROOT / "config/research_sample.json",
    seed_count: int = 10,
) -> dict[str, pd.DataFrame | dict]:
    """Run an optional scalar-routing audit, not the joint forecasting handoff."""
    if seed_count < 2:
        raise ValueError("Use at least two initial seeds")
    output_dir = Path(output_dir)
    for protected in (Path(ingested_dir), PROJECT_ROOT / "data/raw"):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("Audit outputs must be outside source directories")
    fx, metadata, cfg = load_selected_fx(ingested_dir, sample_path)
    start, evaluation = "1995-01", "2010-01"
    raw_features = make_rolling_features(fx)
    features = raw_features.loc[raw_features.index.get_level_values("period") >= pd.Period(start, "M")]
    experiments, comparison, supports = {}, [], []
    grouped_labels = {}
    for k in (2, 3):
        result = run_cross_sections(features, n_clusters=k, extra_baselines=True)
        experiments[k] = result
        comparison.append(summarize_experiment(result, evaluation).assign(k=k, scaling="cross_sectional", lookback=12))
        supports.append(cluster_support(result, k, evaluation))
        for window in (1, 6, 12):
            grouped_labels[f"r2k_k{k}_w{window}"] = label_series(result, f"r2k_w{window}")
        grouped_labels[f"warm_k{k}"] = label_series(result, "kmeans_warm")
        grouped_labels[f"aligned_k{k}"] = label_series(result, "kmeans_aligned")

    # Fixed random currency groups control for merely giving the model more fits.
    rng = np.random.default_rng(2026)
    random_ids = dict(zip(sorted(fx.columns), rng.permutation(len(fx.columns)) % 2))
    base_index = grouped_labels["r2k_k2_w1"].index
    grouped_labels["random_static_k2"] = pd.Series(
        base_index.get_level_values("currency").map(random_ids).to_numpy(), index=base_index,
    )

    sensitivity_rows = []
    cases = [
        ("fixed_initial_scale", features, "initial_cross_section"),
        ("lookback_3m", make_rolling_features(fx, 3), "cross_sectional"),
        ("lookback_24m", make_rolling_features(fx, 24), "cross_sectional"),
        ("without_ARS_TRY_stress_only", features.loc[~features.index.get_level_values("currency").isin(["ARS", "TRY"])], "cross_sectional"),
        ("initialize_2005", features.loc[features.index.get_level_values("period") >= pd.Period("2005-01", "M")], "cross_sectional"),
    ]
    for name, candidate_features, scaling in cases:
        first = pd.Period("1996-01" if name == "lookback_24m" else ("2005-01" if name == "initialize_2005" else start), "M")
        candidate_features = candidate_features.loc[candidate_features.index.get_level_values("period") >= first]
        candidate = run_cross_sections(candidate_features, n_clusters=2, windows=(1,),
                                       baselines=False, scaling=scaling)
        if first == pd.Period(start, "M"):
            reference = experiments[2]
        else:
            # Match initialization date to avoid mixing lookback with start-date effects.
            reference_features = features if name == "initialize_2005" else features.loc[features.index.get_level_values("period") >= first]
            reference = run_cross_sections(reference_features, n_clusters=2, windows=(1,), baselines=False)
        agreement = compare_partitions(label_series(candidate, "r2k_w1"), label_series(reference, "r2k_w1"), evaluation)
        summary = summarize_experiment(candidate, evaluation).iloc[0].to_dict()
        sensitivity_rows.append({"case": name, "initialization": str(first), "scaling": scaling,
                                 "mean_ari_to_reference": agreement.ari.mean(), **summary})

    historical = features.loc[features.index.get_level_values("period") < pd.Period(evaluation, "M")]
    seed_rows = []
    reference = label_series(experiments[2], "r2k_w1")
    for n_init in (1, 10):
        for seed in range(seed_count):
            result = run_cross_sections(historical, n_clusters=2, windows=(1,), seed=seed,
                                        n_init=n_init, baselines=False)
            agreement = compare_partitions(label_series(result, "r2k_w1"), reference, start)
            row = summarize_experiment(result, start).iloc[0].to_dict()
            seed_rows.append({"seed": seed, "n_init": n_init, "mean_ari_to_reference": agreement.ari.mean(),
                              "ari_first12": agreement.ari.head(12).mean(),
                              "ari_last12": agreement.ari.tail(12).mean(), **row})

    shares, persistence = geometry_audit(features, evaluation)
    panel = make_screen_panel(fx)
    screen = run_grouping_screen(panel, grouped_labels, metadata, evaluation_start=evaluation)
    counterexample = stationary_counterexample()
    tables = {"partition_comparison": pd.concat(comparison, ignore_index=True),
              "cluster_support": pd.concat(supports, ignore_index=True),
              "sensitivity": pd.DataFrame(sensitivity_rows), "initialization_sensitivity": pd.DataFrame(seed_rows),
              "feature_variance_shares": shares, "feature_persistence": persistence,
              "forecast_summary": screen.summary, "forecast_currency_metrics": screen.currency_metrics,
              "forecast_fit_support": screen.fits, "available_inputs": available_input_audit(ingested_dir, cfg)}
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        produced.append(f"{name}.csv")
    screen.forecasts.to_parquet(output_dir / "forecasts.parquet", index=False)
    produced.append("forecasts.parquet")
    for k, experiment in experiments.items():
        experiment.labels.to_parquet(output_dir / f"labels_k{k}.parquet", index=False)
        produced.append(f"labels_k{k}.parquet")
    (output_dir / "stationary_counterexample.json").write_text(json.dumps(counterexample, indent=2) + "\n")
    produced.append("stationary_counterexample.json")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    code_paths = [Path(__file__), PROJECT_ROOT / "src/fx_forecasting/evaluation/grouping_screen.py",
                  PROJECT_ROOT / "src/fx_forecasting/evaluation/rolling_clustering.py",
                  PROJECT_ROOT / "src/fx_forecasting/data/rolling_features.py",
                  PROJECT_ROOT / "src/fx_forecasting/models/rolling_kmeans.py"]
    manifest = {"scope": "Optional historical scalar row-routing audit; does not evaluate joint multi-currency neural inputs or outputs; no untouched confirmatory holdout",
                "settings": {"start": start, "evaluation_start": evaluation, "end": cfg["monthly_window"]["end"],
                             "k": [2, 3], "windows": [1, 6, 12], "seed_count": seed_count,
                             "screen_features": list(PREDICTORS), "ridge_l2": 1.0,
                             "lookback_sensitivities": [3, 12, 24],
                             "scaling_sensitivities": ["cross_sectional", "initial_cross_section"],
                             "initialization_n_init": [1, 10], "reference_seed": 42,
                             "minimum_group_rows": 120, "minimum_group_months": 36,
                             "grouping_routing": "historical origin-time labels", "horizons": [1, 3, 12]},
                "source_sha256": {"observations.parquet": digest(Path(ingested_dir) / "observations.parquet"),
                                  "sample_config": digest(sample_path)},
                "implementation_sha256": {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in code_paths},
                "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scipy", "scikit-learn")}},
                "outputs": produced_output_hashes(output_dir, produced)}
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return {**tables, "stationary_counterexample": counterexample, "manifest": manifest}
