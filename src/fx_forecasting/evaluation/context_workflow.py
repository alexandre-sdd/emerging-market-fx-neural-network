"""Reproduce the BIS-only context prototype from the canonical source snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import importlib.metadata
import json
from pathlib import Path
import platform

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from fx_forecasting.config import PROJECT_ROOT, INGESTED_DIR
from fx_forecasting.data.currency_context import (
    rolling_pair_correlations, compact_correlations, correlation_regime_flags,
)
from fx_forecasting.data.panel import build_feature_panel, build_policy_divergence
from fx_forecasting.evaluation.clustering_handoff import sha256
from fx_forecasting.evaluation.context_screen import make_context_screen_table, run_context_screen
from fx_forecasting.models.currency_clustering import compare_clusterings

DEFAULT_SETTINGS = PROJECT_ROOT / "config/clustering_exogenous.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/clustering/context"


def _write_report(output_dir, settings, pairs, summary, fits):
    primary = summary.loc[summary.window.eq(settings["primary_window"])
                          & summary.return_view.eq(settings["primary_return_view"])].copy()
    primary["split"] = pd.Categorical(primary["split"], categories=["validation", "test"], ordered=True)
    primary = primary.sort_values(["split", "model"])
    scores = primary.set_index(["split", "model"]).equal_currency_rmse
    observation = (
        f"At the prespecified {settings['primary_window']}-month window, adding correlations changes validation "
        f"RMSE from {scores.loc[('validation', 'A_existing')]:.3f} to {scores.loc[('validation', 'B_correlation')]:.3f} "
        f"and test RMSE from {scores.loc[('test', 'A_existing')]:.3f} to {scores.loc[('test', 'B_correlation')]:.3f}. "
        f"The zero-change test RMSE is {scores.loc[('test', 'zero_change')]:.3f}. "
        "Assess both periods and the full sensitivity grid; a favorable test comparison alone does not establish a gain."
    )
    display = primary[["split", "model", "observations", "currencies", "equal_currency_rmse",
                       "equal_currency_mae", "currencies_beating_zero"]]
    lines = [
        "**BIS-only currency context: first forecast screen**", "",
        f"Primary: {settings['primary_window']}-month {settings['primary_return_view']} return correlations; "
        f"{settings['primary_horizon']}-month FX targets. Training ends {settings['fit_end']}; "
        f"validation ends {settings['validation_end']}.", "",
        "Each model uses the same currency-month target cells. RMSE and MAE below are means of "
        "per-currency errors, in percent log FX units. A lower error is better. Positive FX change means depreciation.", "",
        observation, "",
        "| Split | Model | Rows | Currencies | Mean currency RMSE | Mean currency MAE | Currencies beating zero |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in display.itertuples(index=False):
        lines.append(f"| {row.split} | {row.model} | {row.observations} | {row.currencies} | "
                     f"{row.equal_currency_rmse:.3f} | {row.equal_currency_mae:.3f} | {row.currencies_beating_zero} |")
    lines += ["", "A uses the existing canonical numeric inputs, carry/availability information and currency "
              "indicators. B adds three correlation summaries; C adds only BIS policy changes; D adds both. "
              "Arrangement and event features are still pending source/date verification. The own-FX Ridge "
              "uses only each row's four FX-history features and currency indicators, with shared coefficients.", "",
              "Regularization is selected separately for each feature set/window/view using equal-currency "
              "validation RMSE. Models and preprocessing then remain frozen at the training cutoff. Targets "
              "crossing a split endpoint are excluded. There is no monthly model refit.", "",
              "The screen uses current monthly rows, not the handoff's twelve-month joint neural sequences. "
              "Its sample counts and error levels should not be compared directly with those sequence experiments. "
              "New context variables are optional, with training-only median imputation and explicit missing indicators; "
              "the baseline's mandatory input requirements determine the common sample.", "",
              "Peer groups are adjusted-average K=3, fitted on 2011–2019. They may describe training history "
              "but cannot support an earlier-origin out-of-sample claim. High/low flags use per-currency "
              "training quartiles and are exported for later experiments; this screen uses continuous correlations.", "",
              "Revised BIS data, retrospective peg exclusions, overlapping targets and prior inspection of the "
              "evaluation periods limit interpretation. These are exploratory results, without a significance claim.", "",
              "All 6/9/12/18-month and raw/median-adjusted sensitivities are in `forecast_summary.csv`. "
              "The 60-month series is a descriptive reference. The HTML timeline contains no verified economic "
              "event annotations yet; use the source ledger to add them.", "",
              "Reproduce from the repository root: `.venv/bin/python scripts/run_currency_context.py`.", "",
              "Next: verify the arrangement ledger; join public dates to the timeline; test the arrangement "
              "block on the same sample; run noise/date sensitivities before the joint neural comparison.", ""]
    (output_dir / "findings.md").write_text("\n".join(lines))
    examples = [("DKK", "EUR"), ("CHF", "EUR"), ("CNY", "MYR"),
                ("AUD", "CAD"), ("BRL", "MXN"), ("EUR", "MAD")]
    figure = make_subplots(rows=3, cols=2, subplot_titles=[f"{a} / {b}" for a, b in examples],
                           shared_xaxes=True, vertical_spacing=.12)
    colors = {"raw": "#176c91", "median_adjusted": "#c46a35"}
    for position, (a, b) in enumerate(examples):
        left, right = sorted((a, b))
        selected = pairs.loc[pairs.currency_a.eq(left) & pairs.currency_b.eq(right)
                             & pairs.window.eq(settings["primary_window"])]
        for view in settings["return_views"]:
            history = selected.loc[selected.return_view.eq(view)]
            figure.add_trace(go.Scatter(
                x=history.period.astype(str).tolist(), y=history.correlation.tolist(),
                customdata=history.observed_pairs.tolist(), connectgaps=False,
                name=view, legendgroup=view, showlegend=position == 0, line=dict(color=colors[view], width=1.6),
                hovertemplate="%{x}<br>Correlation %{y:.3f}<br>Paired months %{customdata}<extra>%{fullData.name}</extra>",
            ), row=position//2+1, col=position%2+1)
    figure.update_layout(template="plotly_white", height=820, margin=dict(t=70, l=45, r=20, b=45),
                         legend=dict(orientation="h", y=1.09), font=dict(family="Arial"))
    figure.update_yaxes(range=[-1.05, 1.05], title_text="Correlation")
    figure.update_xaxes(type="date", tickformat="%Y", dtick="M60")
    timeline = figure.to_html(full_html=False, include_plotlyjs=False,
                              config={"responsive": True, "displaylogo": False})
    chosen = fits.loc[fits.window.eq(settings["primary_window"])
                      & fits.return_view.eq(settings["primary_return_view"])]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Currency context prototype</title>
<style>body{{margin:0;background:#f4f6f8;color:#203343;font:15px/1.6 Arial,sans-serif}}main{{max-width:1160px;margin:auto;padding:32px 22px}}
h1{{font-size:34px;line-height:1.2}}section{{background:white;border:1px solid #dde4e9;padding:22px;margin:20px 0;border-radius:8px}}
.note{{color:#5b6f7d}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #e3e9ed}}
th{{background:#edf3f6}}code{{background:#edf3f6;padding:2px 5px}}a{{color:#176c91}}</style><script>{get_plotlyjs()}</script></head>
<body><main><p class="note">39 currencies · exploratory historical validation · BIS-only inputs</p>
<h1>Do changing currency relationships help forecasts?</h1>
<p>Nine-month return correlations and observed policy divergence, added to the existing FX/macro inputs.</p>
<section><h2>First matched forecast screen</h2><p>Mean per-currency RMSE/MAE; lower is better. A: existing inputs. B: + correlations.
C: + policy changes only. D: + both. No arrangement or event features have been verified yet.</p>
<p>{html.escape(observation)}</p><div class="scroll">{display.to_html(index=False, border=0, float_format=lambda x: f'{x:.3f}')}</div>
<p class="note">All models score the same target cells. This pooled currency-month regression screen uses revised histories and previously inspected evaluation periods.
It does not establish joint neural gains or statistical significance.</p></section>
<section><h2>Monthly relationship timelines</h2><p>{settings['primary_window']} trailing monthly returns, updated monthly.
Raw and median-adjusted correlations measure different relationships. Gaps remain gaps. Economic annotations await source verification.</p>{timeline}</section>
<section><h2>Fitted model settings</h2><p>Training through {html.escape(settings['fit_end'])}; validation through {html.escape(settings['validation_end'])}.
Penalties selected on validation only. Numeric predictors use training-only imputation/scaling; missingness and fixed currency indicators are appended.</p>
<div class="scroll">{chosen[['model','selected_l2','training_rows','training_months','latest_training_endpoint']].to_html(index=False,border=0)}</div></section>
<section><h2>What to do next</h2><p>Complete the arrangement/event ledger, annotate episodes, and test dated arrangement features.
Then run noise/date checks and a compact joint neural comparison. The full window/view sensitivity grid is saved alongside this page.</p>
<p><a href="findings.md">Written findings</a> · <a href="forecast_summary.csv">All window/view results</a> ·
<a href="feature_coverage.csv">Feature availability</a> · <a href="manifest.json">Provenance</a></p>
<p><code>.venv/bin/python scripts/run_currency_context.py</code></p></section></main></body></html>"""
    (output_dir / "findings.html").write_text(page)


def build_context_prototype(*, output_dir: Path = DEFAULT_OUTPUT, settings_path: Path = DEFAULT_SETTINGS,
                            ingested_dir: Path = INGESTED_DIR,
                            sample_path: Path = PROJECT_ROOT / "config/research_sample.json") -> dict:
    settings = json.loads(Path(settings_path).read_text())
    if (settings["schema_version"] != 1 or settings["primary_window"] not in settings["correlation_windows"]
            or not set(settings["forecast_windows"]).issubset(settings["correlation_windows"])
            or settings["primary_window"] not in settings["forecast_windows"]
            or settings["primary_return_view"] not in settings["return_views"]):
        raise ValueError("Inconsistent context experiment settings")
    output_dir = Path(output_dir)
    for protected in (Path(ingested_dir), PROJECT_ROOT / "data/raw"):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("Outputs must be outside source data directories")
    print("Reading canonical BIS inputs and deriving observed policy changes...", flush=True)
    obs_path = Path(ingested_dir) / "observations.parquet"
    obs = pd.read_parquet(obs_path)
    panel = build_feature_panel(observations=obs, sample_path=sample_path)
    policy_local, policy_global = build_policy_divergence(panel, obs)
    print("Fitting fixed training-window peer groups...", flush=True)
    comparison = compare_clusterings(panel.fx, fit_start=settings["fit_start"], fit_end=settings["fit_end"],
                                    k_values=(settings["peer_k"],), bootstrap_reps=0,
                                    seed=settings["seed"], include_dtw=False)
    memberships = comparison.memberships.loc[comparison.memberships.method.eq(settings["peer_method"])].copy()
    if memberships.empty:
        raise ValueError("Requested peer grouping method is unavailable")
    peers = memberships.set_index("currency").cluster.to_dict()
    print("Computing all monthly pair relationships and window/view sensitivities...", flush=True)
    pairs = rolling_pair_correlations(panel.fx, windows=tuple(settings["correlation_windows"]),
                                      return_views=tuple(settings["return_views"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = []

    def parquet(name, frame, index=False):
        frame.to_parquet(output_dir / name, index=index)
        produced.append(name)

    def csv(name, frame):
        frame.to_csv(output_dir / name, index=False)
        produced.append(name)

    parquet("rolling_correlations.parquet", pairs)
    csv("peer_memberships.csv", memberships)
    aggregates = {key: [] for key in ("predictions", "validation_grid", "fits", "preprocessing", "summary", "currency_metrics")}
    primary_local = None
    for view in settings["return_views"]:
        for window in settings["forecast_windows"]:
            print(f"Matched forecast screen: {view}, {window} months...", flush=True)
            correlations, counts = compact_correlations(pairs, sorted(panel.fx.columns), peers,
                                                         window=window, return_view=view,
                                                         minimum_peers=settings["minimum_peers"])
            local = correlations.join(policy_local)
            if window == settings["primary_window"] and view == settings["primary_return_view"]:
                flags, thresholds = correlation_regime_flags(correlations, fit_end=settings["fit_end"],
                                                             minimum_months=settings["minimum_threshold_months"])
                primary_local = local.join(flags).join(counts)
                parquet("currency_context.parquet", primary_local, index=True)
                parquet("global_context.parquet", policy_global, index=True)
                csv("correlation_thresholds.csv", thresholds)
            table = make_context_screen_table(panel, local, policy_global, horizon=settings["primary_horizon"])
            result = run_context_screen(table, fit_end=settings["fit_end"], validation_end=settings["validation_end"],
                                         l2_grid=tuple(settings["ridge_l2_grid"]),
                                         minimum_training_months=settings["minimum_training_months"])
            for key in aggregates:
                frame = getattr(result, key).copy()
                frame["window"], frame["return_view"] = window, view
                aggregates[key].append(frame)
    tables = {key: pd.concat(frames, ignore_index=True) for key, frames in aggregates.items()}
    for name, key in (("forecast_summary.csv", "summary"), ("currency_metrics.csv", "currency_metrics"),
                      ("validation_grid.csv", "validation_grid"), ("model_fits.csv", "fits"),
                      ("preprocessing.csv", "preprocessing")):
        csv(name, tables[key])
    parquet("forecasts.parquet", tables["predictions"])
    coverage = []
    for block, frame in (("local", primary_local.drop(columns="peer_count")), ("global", policy_global)):
        periods = frame.index.get_level_values("period")
        for split, mask in (("train", periods <= settings["fit_end"]),
                            ("validation", (periods > settings["fit_end"]) & (periods <= settings["validation_end"])),
                            ("test", periods > settings["validation_end"])):
            for feature in frame:
                coverage.append({"block": block, "split": split, "feature": feature, "cells": int(mask.sum()),
                                 "available_cells": int(frame.loc[mask, feature].notna().sum()),
                                 "available_fraction": float(frame.loc[mask, feature].notna().mean())})
    coverage = pd.DataFrame(coverage)
    csv("feature_coverage.csv", coverage)
    _write_report(output_dir, settings, pairs, tables["summary"], tables["fits"])
    produced += ["findings.md", "findings.html"]
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    produced.append("settings.json")
    code_paths = [PROJECT_ROOT / relative for relative in (
        "src/fx_forecasting/data/panel.py", "src/fx_forecasting/data/rolling_features.py",
        "src/fx_forecasting/data/currency_context.py", "src/fx_forecasting/models/currency_clustering.py",
        "src/fx_forecasting/models/rolling_kmeans.py", "src/fx_forecasting/evaluation/grouping_screen.py",
        "src/fx_forecasting/evaluation/context_screen.py", "src/fx_forecasting/evaluation/context_workflow.py",
        "scripts/run_currency_context.py")]
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "currency_order": sorted(panel.fx.columns), "pair_rows": len(pairs),
        "source_sha256": {"observations.parquet": sha256(obs_path), "research_sample.json": sha256(sample_path),
                          "clustering_exogenous.json": sha256(settings_path)},
        "implementation_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code_paths},
        "versions": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                     for p in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "plotly")}},
        "feature_timing": {**panel.config["macro_availability"],
                           "policy_changes": "Configured policy lag; raw observed levels only, no carries",
                           "correlations": "Trailing contiguous monthly returns through origin t",
                           "thresholds": "Per-currency training quartiles; not an earlier-origin backtest",
                           "peers": f"{settings['peer_method']} K={settings['peer_k']}, fit {settings['fit_start']}–{settings['fit_end']}"},
        "limitations": settings["limitations"],
        "outputs": {name: sha256(output_dir/name) for name in sorted(produced)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
