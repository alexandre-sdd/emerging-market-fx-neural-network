"""One reproducible comparison and joint-forecasting dataset workflow."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import pandas as pd
import numpy as np

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT
from fx_forecasting.data.panel import build_feature_panel, LOCAL_FEATURES, GLOBAL_FEATURES
from fx_forecasting.data.neural_dataset import (
    build_neural_dataset, eligibility_tables, frozen_groups, save_neural_dataset,
)
from fx_forecasting.models.currency_clustering import compare_clusterings

DEFAULT_OUTPUT = PROJECT_ROOT / "reports/clustering/handoff"
DEFAULT_SETTINGS = PROJECT_ROOT / "config/clustering_handoff.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_output_manifest(output_dir: Path, produced: list[str]) -> dict:
    """Hash an explicit list; pre-existing unrelated files never enter a run."""
    path = Path(output_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"] = {name: sha256(Path(output_dir) / name) for name in sorted(set(produced)) if name != "manifest.json"}
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def build_clustering_handoff(
    *, output_dir: Path = DEFAULT_OUTPUT, settings_path: Path = DEFAULT_SETTINGS,
    ingested_dir: Path = INGESTED_DIR,
    sample_path: Path = PROJECT_ROOT / "config/research_sample.json",
    bootstrap_reps: int | None = None, include_dtw: bool = True, write_report: bool = True,
) -> dict:
    """Read local BIS data, compare frozen groups, export once, and report.

    The scalar-routing Ridge audit is not imported or executed by this workflow.
    include_dtw=False and bootstrap_reps overrides support explicit smoke tests;
    the default deliverable runs DTW and all 50 block-bootstrap replications.
    """
    settings = json.loads(Path(settings_path).read_text())
    if bootstrap_reps is not None:
        settings["bootstrap_reps"] = bootstrap_reps
    output_dir = Path(output_dir)
    for protected in (Path(ingested_dir), PROJECT_ROOT / "data/raw"):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("Handoff outputs must be outside raw and ingested source directories")
    panel = build_feature_panel(ingested_dir=ingested_dir, sample_path=sample_path)
    comparison = compare_clusterings(
        panel.fx, fit_start=settings["fit_start"], fit_end=settings["fit_end"],
        k_values=tuple(settings["k_values"]), bootstrap_reps=settings["bootstrap_reps"],
        seed=settings["seed"], include_dtw=include_dtw,
    )
    axis = sorted(panel.fx.columns)
    groups = frozen_groups(comparison.memberships, axis)
    for definition in groups.values():
        score = comparison.scores.loc[comparison.scores.method.eq(definition["method"]) & comparison.scores.k.eq(definition["k"])]
        method_start = settings["fit_start"] if score.empty else str(score.iloc[0].fit_start)
        definition.update({"fit_start": method_start, "fit_end": settings["fit_end"],
                           "frozen_after_fit": True})
    dataset = build_neural_dataset(panel, groups, lookback=settings["lookback"],
                                    horizons=tuple(settings["horizons"]), fit_end=settings["fit_end"],
                                    validation_end=settings["validation_end"])
    dataset.metadata["grouping"] = {"fit_start": settings["fit_start"], "fit_end": settings["fit_end"],
                                     "default_method": settings["default_method"], "default_k": settings["default_k"],
                                     "interpretation": settings["interpretation"]}
    dataset.metadata["feature_timing"] = panel.config["macro_availability"]
    dataset.metadata["source_sha256"] = {"observations.parquet": sha256(Path(ingested_dir)/"observations.parquet"),
                                          "research_sample.json": sha256(sample_path)}
    details, sequence_counts = eligibility_tables(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = save_neural_dataset(dataset, output_dir)
    for name, frame in (("comparison.csv", comparison.scores), ("memberships.csv", comparison.memberships),
                         ("stability.csv", comparison.stability), ("sequence_counts.csv", sequence_counts)):
        frame.to_csv(output_dir/name, index=False)
        produced.append(name)
    np.savez_compressed(output_dir/"distances.npz", **{name: table.to_numpy() for name, table in comparison.distances.items()})
    produced.append("distances.npz")
    local = panel.local.join(panel.provenance).copy()
    local["fx_level"] = panel.fx.rename_axis(index="period", columns="currency").stack().reindex(local.index)
    local.to_parquet(output_dir/"monthly_local.parquet")
    panel.global_features.join(panel.global_provenance).to_parquet(output_dir/"monthly_global.parquet")
    details.to_parquet(output_dir/"eligibility.parquet", index=False)
    panel.metadata.to_csv(output_dir/"currency_metadata.csv")
    produced += ["monthly_local.parquet", "monthly_global.parquet", "eligibility.parquet", "currency_metadata.csv"]
    coverage = []
    for block, frame in (("local", panel.local), ("global", panel.global_features)):
        periods = frame.index.get_level_values("period") if isinstance(frame.index, pd.MultiIndex) else frame.index
        for split, mask in (("train", periods <= pd.Period(settings["fit_end"], "M")),
                            ("validation", (periods > pd.Period(settings["fit_end"], "M")) & (periods <= pd.Period(settings["validation_end"], "M"))),
                            ("test", periods > pd.Period(settings["validation_end"], "M"))):
            for feature in frame:
                coverage.append({"block": block, "split": split, "feature": feature, "cells": int(mask.sum()),
                                 "available_cells": int(frame.loc[mask, feature].notna().sum()),
                                 "available_fraction": float(frame.loc[mask, feature].notna().mean())})
    pd.DataFrame(coverage).to_csv(output_dir/"feature_coverage.csv", index=False)
    (output_dir/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    produced += ["feature_coverage.csv", "settings.json"]
    code = [PROJECT_ROOT/"src/fx_forecasting"/relative for relative in (
        "data/panel.py", "data/rolling_features.py", "data/neural_dataset.py", "models/currency_clustering.py",
        "models/rolling_kmeans.py", "evaluation/clustering_handoff.py", "evaluation/clustering_report.py",
        "evaluation/clustering_html.py", "evaluation/clustering_theory.py", "evaluation/clustering_evolution.py", "data/report_theme.py")]
    code += [PROJECT_ROOT/"tests"/name for name in ("test_r2k_paper_reference.py", "test_clustering_evolution.py")]
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow"] + (["tslearn"] if include_dtw else [])
    if write_report:
        packages.append("plotly")
    manifest = {"schema_version": 1, "purpose": "Fixed currency channels for joint within-cluster neural forecasting",
                "settings": settings, "include_dtw": include_dtw, "currency_order": axis,
                "array_shapes": {name: list(values.shape) for name, values in dataset.arrays.items()},
                "source_sha256": {**dataset.metadata["source_sha256"], "clustering_handoff.json": sha256(settings_path)},
                "implementation_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code if p.is_file()},
                "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in packages}},
                "local_features": list(LOCAL_FEATURES), "global_features": list(GLOBAL_FEATURES),
                "limitations": ["Revised historical data and retrospective regime exclusions",
                                "Candidate groups; no neural forecasting performance has been tested",
                                "Shared USD factor adjustment changes similarity geometry and can induce dependence",
                                "Sequence counts are forecast origins, not independent currency-by-horizon rows"], "outputs": {}}
    (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    update_output_manifest(output_dir, produced)
    if write_report:
        from fx_forecasting.evaluation.clustering_report import write_clustering_report
        produced += write_clustering_report(output_dir)
    return update_output_manifest(output_dir, produced)
