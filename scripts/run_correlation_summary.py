"""Build an offline comparison of correlation clustering on returns and FX levels."""

from pathlib import Path
import html
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "clustering" / "correlation"
settings = json.loads((OUT / "settings.json").read_text())
selection = pd.read_csv(OUT / "model_selection.csv")
summary = pd.read_csv(OUT / "pool_summary.csv")
titles = {"returns": "Monthly returns", "levels": "Normalized FX levels"}
colors = {1: "#2B6CB0", 2: "#B77916"}
maps = {name: pd.read_csv(OUT / f"{name}_map.csv").set_index("currency") for name in titles}
histories = {name: pd.read_csv(OUT / f"{name}_histories.csv", index_col=0) for name in titles}


def style(fig, height=360, **layout):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="white",
                      font=dict(family="Arial, sans-serif", size=12, color="#26364A"),
                      margin=dict(l=48, r=18, t=32, b=42), **layout)
    fig.update_xaxes(gridcolor="#EBEFF4", zeroline=False)
    fig.update_yaxes(gridcolor="#EBEFF4", zeroline=False)
    return fig


scores = go.Figure()
for name, table in selection.groupby("representation", sort=False):
    scores.add_trace(go.Scatter(
        x=table.k.tolist(), y=table.silhouette.tolist(), mode="lines+markers", name=titles[name],
        line=dict(width=2, color={"returns": "#2B6CB0", "levels": "#168574"}[name]),
        customdata=table.cluster_sizes.tolist(),
        hovertemplate="K=%{x}<br>Silhouette: %{y:.3f}<br>Sizes: %{customdata}<extra>%{fullData.name}</extra>",
    ))
scores.add_vline(x=2, line_dash="dot", line_color="#AAB7C5")
style(scores, 280, xaxis_title="Number of clusters", yaxis_title="Silhouette",
      legend=dict(orientation="h", y=1.15, x=0))
scores.update_xaxes(dtick=1)

figures = {"scores": scores}
for name, members in maps.items():
    map_fig, paths = go.Figure(), go.Figure()
    for pool_id, group in members.groupby("pool_id", sort=True):
        label = f"Group {pool_id} · {len(group)} {'currency' if len(group) == 1 else 'currencies'}"
        small = len(group) <= 2
        map_fig.add_trace(go.Scatter(
            x=group.x.tolist(), y=group.y.tolist(), text=group.index.tolist(),
            mode="markers+text" if small else "markers", name=label, textposition="top center",
            marker=dict(size=10, color=colors[pool_id], line=dict(color="white", width=1)),
            customdata=group[["economy", "silhouette"]].to_numpy().tolist(),
            hovertemplate="<b>%{text} · %{customdata[0]}</b><br>Silhouette: %{customdata[1]:.3f}<extra>%{fullData.name}</extra>",
        ))
        for currency in group.index:
            paths.add_trace(go.Scatter(
                x=pd.to_datetime(histories[name].index).strftime("%Y-%m-%d").tolist(),
                y=histories[name][currency].tolist(), name=currency, mode="lines",
                opacity=.85 if small else .16, line=dict(color=colors[pool_id], width=1.7 if small else 1),
                showlegend=small,
                hovertemplate="%{fullData.name}<br>%{x|%b %Y}: %{y:.2f} SD<extra></extra>",
            ))
        if not small:
            paths.add_trace(go.Scatter(
                x=pd.to_datetime(histories[name].index).strftime("%Y-%m-%d").tolist(),
                y=histories[name][group.index].mean(axis=1).tolist(), mode="lines",
                line=dict(color=colors[pool_id], width=2.5), name="Large-group mean",
                hovertemplate="%{x|%b %Y}<br>Mean: %{y:.2f} SD<extra></extra>",
            ))
    legend = dict(orientation="h", y=1.13, x=0, font=dict(size=11))
    style(map_fig, 380, xaxis_title="MDS 1", yaxis_title="MDS 2", legend=legend)
    map_fig.update_yaxes(scaleanchor="x", scaleratio=1)
    style(paths, 380, yaxis_title="Standard deviations", legend=legend)
    paths.update_xaxes(dtick="M24", tickformat="%Y")
    figures[f"{name}_map"], figures[f"{name}_paths"] = map_fig, paths

membership = {}
for name, table in summary.groupby("representation"):
    membership[name] = "".join(
        f'<p><strong>Group {int(row.pool_id)} · {int(row.currencies)} {"currency" if row.currencies == 1 else "currencies"}</strong><br>'
        f'{html.escape(row.members)}</p>' for row in table.itertuples()
    )
best = selection.loc[selection.groupby("representation").silhouette.idxmax()].set_index("representation")
payload = "{" + ",".join(json.dumps(name) + ":" + fig.to_json() for name, fig in figures.items()) + "}"
page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Returns or normalized FX levels?</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#F4F6F8;color:#1B2D43;font:14px/1.6 Arial,sans-serif}
main{max-width:1120px;margin:auto;padding:32px 26px}h1{font-size:32px;letter-spacing:-.7px;margin:6px 0 8px;line-height:1.2}h2{font-size:20px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.1em;color:#6C7C8E}.subtitle{color:#607186;margin:0}
section{background:white;border:1px solid #E0E6EC;border-radius:8px;padding:20px 22px;margin-top:18px}.takeaway{border-left:3px solid #168574;background:#EAF3F1;padding:11px 15px;margin:18px 0 0}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.explain p{font-size:13px;margin:0;color:#4A5F75}.metric{font-size:25px;font-weight:700;color:#203D5B;margin:3px 0}.metric span{font-size:12px;color:#687A8E;font-weight:400}
.note{font-size:12px;color:#61758A;margin:12px 0 0}.explain strong{color:#213D57}.tabs{display:flex;gap:5px;background:#F0F3F6;padding:4px;border-radius:6px;width:max-content;margin-bottom:14px}
button{font:600 12px Arial,sans-serif;border:0;border-radius:4px;padding:8px 12px;background:transparent;color:#687B8F;cursor:pointer}button.active{background:white;color:#193D5A;box-shadow:0 1px 3px #162B4020}button:focus-visible{outline:2px solid #2B6CB0}
.plot{width:100%}.plot-label{font-size:12px;font-weight:700;margin-bottom:3px}.reading{background:#F2F6F9;padding:12px 15px;font-size:13px;margin-top:12px;border-radius:5px}
details{font-size:11px;color:#65768A;margin-top:12px}summary{cursor:pointer;font-weight:700}details p{margin:10px 0}.conclusion{background:#EEF3F8;border:0;font-size:13px;padding:15px 20px}
footer{font-size:10px;color:#78899A;margin:15px 0}footer a{color:#5D7892}
@media(max-width:720px){main{padding:20px 13px}h1{font-size:26px}.columns{grid-template-columns:1fr;gap:18px}section{padding:16px}}
</style><script>__PLOTLY__</script></head><body><main>
<div class="eyebrow">39 CURRENCIES · 108 MONTHS · JAN 2011–DEC 2019</div>
<h1>Returns or normalized FX levels?</h1><p class="subtitle">Same dates, currencies, correlation distance and average-linkage algorithm. No time warping.</p>
<div class="takeaway"><strong>Both representations are valid.</strong> Returns describe monthly co-movement; normalized levels describe historical paths. Neither split yet gives several substantial training pools.</div>
<section><h2>What are we comparing?</h2><div class="columns explain">
<div><h3>Monthly returns</h3><p>100 × log(Sₜ / Sₜ₋₁), standardized by currency. Asks: <strong>which currencies move together in the same months?</strong></p></div>
<div><h3>Normalized FX levels</h3><p>(Sₜ − mean S) / SD(S), within currency. Removes units and scale while retaining trends. Asks: <strong>which currencies follow similar paths?</strong></p></div></div>
<p class="note">Pearson correlation is unchanged by positive rescaling. Raw, rebased and z-scored levels have the same correlation matrix on this window. The substantive choice is levels versus returns.</p></section>
<section><h2>How many groups?</h2><div class="columns explain">
<div><h3>Returns: K=2</h3><div class="metric">__RETURN_SCORE__ <span>silhouette</span></div><p><strong>38 currencies + JPY alone.</strong> A broad shared pattern with one exception.</p></div>
<div><h3>Levels: K=2</h3><div class="metric">__LEVEL_SCORE__ <span>silhouette</span></div><p><strong>37 currencies + ISK and ILS.</strong> A small group follows a different historical path.</p></div></div>
<div id="scores" class="plot"></div><p class="note">K=2 has the highest silhouette in the inspected K=2–6 range. At K=3, both methods give 36 + 2 + 1. Higher level silhouette does not establish better forecasts; shared trends can inflate level correlations. Hierarchical clustering has no k-means inertia.</p></section>
<section><h2>Inspect the currencies together</h2><div class="tabs" role="tablist" aria-label="Representation">
<button id="returns-tab" class="active" role="tab" aria-selected="true">Monthly returns</button><button id="levels-tab" role="tab" aria-selected="false">Normalized FX levels</button></div>
<div class="columns"><div><div class="plot-label">Currency map</div><div id="map" class="plot"></div></div><div><div class="plot-label">Observed histories</div><div id="paths" class="plot"></div></div></div>
<div id="reading" class="reading"></div><p class="note">Blue = large group; amber = small group. Hover to identify a currency. Each MDS map approximates its own distance matrix; positions are not comparable across views. Histories use actual calendar dates.</p>
<details><summary>Currency memberships</summary><div id="membership"></div></details></section>
<section class="conclusion"><strong>For the forecasting project:</strong> use returns when studying common monthly shocks and levels when studying common paths. Keep both as candidate grouping designs, retain global pooling as the benchmark, and judge usefulness on later forecasts—not silhouette alone.</section>
<footer>Both use d = √[2(1−Pearson correlation)], average linkage, the same 39-currency sample and existing launch/peg masks. No FX imputation. Normalization and grouping end in December 2019. A singleton silhouette is zero by convention. <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html">Method</a></footer>
</main><script>
const specs=__FIGURES__,memberships=__MEMBERSHIP__;
const readings={returns:'JPY is the only singleton at K=2. At K=3, ARS and TRY also split off. Average linkage mostly identifies exceptions to broad co-movement; this does not yet justify several separate training models.',levels:'ISK and ILS separate in level-path space. Their membership differs from the return split: the normalization makes units comparable, but shared long-term trends still influence grouping. This is a useful sensitivity, not evidence of forecast gains.'};
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};
async function show(name){for(const n of ['returns','levels']){document.getElementById(n+'-tab').classList.toggle('active',n===name);document.getElementById(n+'-tab').setAttribute('aria-selected',String(n===name))}document.getElementById('reading').textContent=readings[name];document.getElementById('membership').innerHTML=memberships[name];await Promise.all(['map','paths'].map(id=>Plotly.react(id,specs[name+'_'+id].data,specs[name+'_'+id].layout,config)))}
document.getElementById('returns-tab').onclick=()=>show('returns');document.getElementById('levels-tab').onclick=()=>show('levels');
Plotly.newPlot('scores',specs.scores.data,specs.scores.layout,config);show('returns');
</script></body></html>"""
page = (page.replace("__PLOTLY__", get_plotlyjs())
        .replace("__RETURN_SCORE__", f"{best.loc['returns', 'silhouette']:.3f}")
        .replace("__LEVEL_SCORE__", f"{best.loc['levels', 'silhouette']:.3f}")
        .replace("__FIGURES__", payload)
        .replace("__MEMBERSHIP__", json.dumps(membership)))
destination = OUT / "correlation_comparison.html"
destination.write_text(page)
print(destination)
