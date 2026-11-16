"""Describe the CHF/EUR floor episode and a rolling-window counterexample.

Uses the existing context export; never removes the January shock from model
inputs or forecast evaluation. The leave-one-observation calculation is an
explicitly retrospective influence diagnostic with eight observations.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fx_forecasting.data.panel import build_feature_panel


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    context = ROOT / "reports/clustering/context"
    output = context / "chf_eur"
    manifest = json.loads((context / "manifest.json").read_text())
    sources = {"observations.parquet": ROOT / "data/ingested/observations.parquet",
               "research_sample.json": ROOT / "config/research_sample.json"}
    for name, path in sources.items():
        if sha256(path) != manifest["source_sha256"][name]:
            raise ValueError("The context export is stale relative to the current source snapshot")
    pair_path = context / "rolling_correlations.parquet"
    if sha256(pair_path) != manifest["outputs"][pair_path.name]:
        raise ValueError("Saved correlation output does not match its manifest")
    pairs = pd.read_parquet(pair_path, filters=[("currency_a", "=", "CHF"), ("currency_b", "=", "EUR")])
    panel = build_feature_panel()
    raw = 100*np.log(panel.fx).diff()
    adjusted = raw.sub(raw.median(axis=1), axis=0)
    # Verify saved nine-month values independently against direct sample correlation.
    primary = pairs.loc[pairs.window.eq(9)].pivot(index="period", columns="return_view", values="correlation")
    for view, values in (("raw", raw), ("median_adjusted", adjusted)):
        for period, estimate in primary[view].dropna().items():
            history = values.loc[period-8:period, ["CHF", "EUR"]]
            assert len(history) == 9 and history.notna().all().all()
            assert np.isclose(estimate, history.CHF.corr(history.EUR), atol=1e-9, rtol=1e-9)
    monthly = pd.DataFrame({"chf_per_eur": panel.fx.CHF/panel.fx.EUR,
                            "chf_usd_return": raw.CHF, "eur_usd_return": raw.EUR,
                            "cross_return": raw.CHF-raw.EUR}).join(primary.add_prefix("corr_9m_"))
    shock = pd.Period("2015-01", "M")
    monthly["january_2015_in_9m_window"] = (monthly.index-8 <= shock) & (monthly.index >= shock)
    for view, values in (("raw", raw), ("median_adjusted", adjusted)):
        diagnostic = []
        for period in monthly.index:
            history = values.loc[period-8:period, ["CHF", "EUR"]]
            if period-8 <= shock <= period and len(history) == 9 and history.notna().all().all():
                without = history.drop(shock)
                assert len(without) == 8
                diagnostic.append(without.CHF.corr(without.EUR))
            else:
                diagnostic.append(np.nan)
        monthly[f"diagnostic_without_january_{view}_8obs"] = diagnostic
    intervals = [
        ("Before floor", "2010-01", "2011-08"),
        ("Floor: all nine return months after introduction month", "2012-06", "2014-12"),
        ("Nine-month windows containing January 2015", "2015-01", "2015-09"),
        ("After January shock leaves nine-month window", "2015-10", "2016-12"),
        ("Counterexample: 2021, floor absent", "2021-01", "2021-12"),
    ]
    summaries = []
    for label, start, end in intervals:
        for view in ("raw", "median_adjusted"):
            values = monthly.loc[start:end, f"corr_9m_{view}"].dropna()
            summaries.append({"episode": label, "start": start, "end": end, "return_view": view,
                              "endpoints": len(values), "mean": values.mean(), "min": values.min(), "max": values.max()})
    summary = pd.DataFrame(summaries)
    exits = []
    for window in sorted(pairs.window.unique()):
        last, first = shock+int(window)-1, shock+int(window)
        for view in ("raw", "median_adjusted"):
            series = pairs.loc[pairs.window.eq(window) & pairs.return_view.eq(view)].set_index("period").correlation
            exits.append({"window": int(window), "return_view": view, "last_endpoint_with_january": str(last),
                          "first_endpoint_without_january": str(first), "correlation_before": series.loc[last],
                          "correlation_after": series.loc[first]})
    exits = pd.DataFrame(exits)
    output.mkdir(parents=True, exist_ok=True)
    export = monthly.copy()
    export.index = export.index.astype(str)
    export.index.name = "origin_month"
    export.to_csv(output / "monthly.csv")
    summary.to_csv(output / "episodes.csv", index=False)
    exits.to_csv(output / "window_exits.csv", index=False)
    dates = monthly.index.to_timestamp(how="end").normalize()
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.10,
                          subplot_titles=["CHF per EUR at month-end", "Trailing nine-month return correlations",
                                          "Influence of January 2015: retrospective eight-observation diagnostic"])
    figure.add_trace(go.Scatter(x=dates, y=monthly.chf_per_eur, name="CHF per EUR", line=dict(color="#176c91")), row=1, col=1)
    figure.add_trace(go.Scatter(x=["2011-09-06", "2015-01-15"], y=[1.2, 1.2], name="Announced floor",
                               line=dict(color="#a64d2c", dash="dash")), row=1, col=1)
    for view, color in (("raw", "#176c91"), ("median_adjusted", "#b76a2d")):
        figure.add_trace(go.Scatter(x=dates, y=monthly[f"corr_9m_{view}"], name=view,
                                   legendgroup=view, connectgaps=False, line=dict(color=color)), row=2, col=1)
    figure.add_trace(go.Scatter(x=dates, y=monthly.corr_9m_raw, name="Actual nine-month raw correlation",
                               connectgaps=False, line=dict(color="#176c91")), row=3, col=1)
    figure.add_trace(go.Scatter(x=dates, y=monthly.diagnostic_without_january_raw_8obs,
                               name="Same window, January omitted (8 observations)", connectgaps=False,
                               line=dict(color="#8a4f9a", dash="dot", width=3)), row=3, col=1)
    for row in (1, 2, 3):
        figure.add_vrect(x0="2011-09-06", x1="2015-01-15", fillcolor="#148574", opacity=.08,
                         line_width=0, row=row, col=1)
        for date in ("2011-09-06", "2015-01-15", "2015-10-31"):
            figure.add_vline(x=date, line_dash="dot", line_color="#8597a3", row=row, col=1)
    figure.update_yaxes(range=[-1.05, 1.05], row=2, col=1)
    figure.update_yaxes(range=[-1.05, 1.05], row=3, col=1)
    figure.update_xaxes(type="date", range=["2010-01-01", "2016-12-31"], tickformat="%Y", dtick="M12")
    figure.update_layout(template="plotly_white", height=1000, margin=dict(t=65, l=65, r=25, b=40),
                         legend=dict(orientation="h", y=-.08), font=dict(family="Arial"))
    chart = figure.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CHF/EUR: policy episode and window memory</title><style>body{{font:15px/1.6 Arial;color:#203343;background:#f4f6f8;margin:0}}
main{{max-width:1100px;margin:auto;padding:30px 20px}}section{{background:white;padding:22px;margin:18px 0;border:1px solid #dce4e8;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #dce4e8;text-align:left}}.scroll{{overflow-x:auto}}
code{{background:#edf3f6;padding:2px 5px}}</style><script>{get_plotlyjs()}</script></head><body><main>
<h1>CHF/EUR: a policy episode and a rolling-window effect</h1><p>Monthly BIS data; descriptive historical analysis. Shading marks the EUR floor period.</p>
<section><p><strong>1.</strong> During June 2012–December 2014, mean nine-month raw correlation is 0.945. A one-sided CHF 1.20 per EUR floor can limit relative moves when binding.</p>
<p><strong>2.</strong> Correlation changes from 0.983 in December 2014 to -0.335 in January 2015. The removal coincides with opposite-signed monthly CHF/USD and EUR/USD moves.</p>
<p><strong>3.</strong> The nine-month correlation rises from -0.065 in September 2015 to 0.862 in October as January leaves the window; the floor did not return. In 2021 the mean is 0.926 without the floor.</p>
<p>The primary correlation always retains January. The eight-observation line is an influence diagnostic, not a replacement feature or a forecast result.</p>{chart}</section>
<section><h2>Every selected episode, both return views</h2><div class="scroll">{summary.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')}</div>
<p>Means average overlapping rolling estimates; endpoints are not independent observations. Floor-period endpoints start in June 2012 so the nine return months exclude the September 2011 introduction month.</p></section>
<section><h2>Window memory depends on the chosen length</h2><div class="scroll">{exits.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')}</div>
<p>Exiting one observation and adding a new observation can move a correlation sharply without a new policy event. Median adjustment is a statistical reference, not an identified causal USD shock.</p></section>
<section><h2>Sources and reproduction</h2><p><a href="https://www.snb.ch/en/publications/communication/speeches/2011/ref_20110906_pmh">SNB introduction, 6 September 2011</a> ·
<a href="https://www.snb.ch/en/publications/communication/press-releases/2015/pre_20150115">SNB removal, 15 January 2015</a> ·
<a href="https://www.snb.ch/en/services-events/digital-services/glossary">SNB definition and policy dates</a>.</p>
<p>Original local data and calculations: <a href="monthly.csv">monthly evidence</a>, <a href="episodes.csv">episode summaries</a>, <a href="window_exits.csv">window exits</a>, <a href="manifest.json">hashes</a>.</p>
<p><code>.venv/bin/python scripts/analyze_chf_eur_episode.py</code></p></section></main></body></html>"""
    (output / "case_study.html").write_text(page)
    produced = ["monthly.csv", "episodes.csv", "window_exits.csv", "case_study.html"]
    provenance = {"schema_version": 1, "analysis_date": "2026-11-16", "primary_window": 9,
                  "source_sha256": {name: sha256(path) for name, path in sources.items()},
                  "correlation_export_sha256": sha256(pair_path), "script_sha256": sha256(Path(__file__)),
                  "limitations": ["Descriptive revised-history analysis", "Overlapping estimates are dependent",
                                  "Leaving January out is an eight-observation influence diagnostic only",
                                  "No forecast models, targets or production inputs are changed"],
                  "outputs": {name: sha256(output/name) for name in produced}}
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("Verified every saved nine-month CHF/EUR correlation against direct calculation.")
    print(f"Saved case study and evidence to {output}")


if __name__ == "__main__":
    main()
