"""Build a short, offline Plotly page from the clustering notebook's saved results."""

from pathlib import Path
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "clustering"
settings = json.loads((OUT / "manifest.json").read_text())
sample = json.loads((ROOT / "config" / "research_sample.json").read_text())
selection = pd.read_csv(OUT / "model_selection.csv").set_index("k")
runs = pd.read_csv(OUT / "grid_scores.csv")
members = pd.read_csv(OUT / "cluster_map.csv").set_index("currency")
k = int(settings["selected_k"])
start, end = settings["fit_start"], settings["fit_end"]
colors = {1: "#2B6CB0", 2: "#B77916", 3: "#168574"}
names = {1: "Mixed USD patterns", 2: "Weak residual group", 3: "EUR-centered patterns"}

# Reconstruct observed returns for interpretation; do not refit or change clusters.
obs = pd.read_parquet(ROOT / "data" / "ingested" / "observations.parquet")
sources = {c["source_series"]["fx"]: c["currency"] for c in sample["currencies"]}
rows = obs.loc[obs.dataset.eq("WS_XRU") & obs.frequency.eq("M") & obs.series_id.isin(sources)]
fx = rows.pivot(index="period", columns="series_id", values="value").rename(columns=sources)
fx.index = pd.PeriodIndex(fx.index, freq="M")
fx = fx.reindex(pd.period_range(**sample["monthly_window"], freq="M")).astype(float)
fx = fx.where(fx.gt(0))
for c in sample["currencies"]:
    fx.loc[fx.index < pd.Period(c["first_eligible_month"], "M"), c["currency"]] = np.nan
area_to_currency = {c["area"]: c["currency"] for c in sample["currencies"]}
for interval in sample["excluded_month_intervals"]:
    fx.loc[interval["start"]:interval["end"], area_to_currency[interval["area"]]] = np.nan
returns = (100 * np.log(fx).diff()).loc[start:end, members.index]
standardized = (returns - returns.mean()) / returns.std(ddof=0)

stats = {}
for pool_id, group in members.groupby("pool_id"):
    corr = returns[group.index].corr().to_numpy()
    stats[int(pool_id)] = {
        "n": len(group), "silhouette": float(group.silhouette.mean()),
        "correlation": float(np.median(corr[np.triu_indices(len(group), 1)])),
        "negative": int(group.silhouette.lt(0).sum()),
        "members": ", ".join(sorted(group.index)),
    }


def style(figure, height, **layout):
    figure.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Arial, sans-serif", size=12, color="#26364A"),
        margin=dict(l=48, r=20, t=22, b=44), hoverlabel=dict(font_size=12),
        paper_bgcolor="white", plot_bgcolor="white", **layout,
    )
    figure.update_xaxes(gridcolor="#EBEFF4", zeroline=False)
    figure.update_yaxes(gridcolor="#EBEFF4", zeroline=False)
    return figure


inertia = go.Figure(go.Scatter(
    x=selection.index.tolist(), y=selection.inertia.tolist(), mode="lines+markers",
    line=dict(color="#546E89", width=2), marker=dict(size=7),
    hovertemplate="K=%{x}<br>Median inertia: %{y:.2f}<extra></extra>",
))
inertia.add_trace(go.Scatter(x=[k], y=[selection.loc[k, "inertia"]], mode="markers",
                           marker=dict(size=13, color="#152C46"), hoverinfo="skip"))
inertia.add_vline(x=k, line_dash="dot", line_color="#AAB7C5")
style(inertia, 255, showlegend=False, xaxis_title="Number of clusters", yaxis_title="DTW inertia")
inertia.update_xaxes(dtick=1)

s = selection.loc[2:]
silhouette = go.Figure()
for seed, group in runs.loc[runs.k.gt(1)].groupby("seed"):
    silhouette.add_trace(go.Scatter(
        x=(group.k + {7: -.055, 42: 0, 99: .055}[int(seed)]).tolist(),
        y=group.silhouette.tolist(), mode="markers", marker=dict(size=5, color="#BCC6D2"),
        customdata=group.k.tolist(), hovertemplate=f"K=%{{customdata}} · seed {seed}<br>Silhouette: %{{y:.3f}}<extra></extra>",
    ))
silhouette.add_trace(go.Scatter(
    x=s.index.tolist(), y=s.silhouette.tolist(), mode="lines+markers",
    line=dict(color="#546E89", width=2), marker=dict(size=7),
    error_y=dict(type="data", array=(s.high-s.silhouette).tolist(),
                 arrayminus=(s.silhouette-s.low).tolist(), color="#BEC9D5", thickness=1),
    hovertemplate="K=%{x}<br>Median silhouette: %{y:.3f}<extra></extra>",
))
silhouette.add_trace(go.Scatter(x=[k], y=[s.loc[k, "silhouette"]], mode="markers",
                              marker=dict(size=13, color="#152C46"), hoverinfo="skip"))
silhouette.add_vline(x=k, line_dash="dot", line_color="#AAB7C5")
style(silhouette, 255, showlegend=False, xaxis_title="Number of clusters", yaxis_title="DTW silhouette")
silhouette.update_xaxes(dtick=1, range=[1.7, 6.3])

cluster_map = go.Figure()
patterns = go.Figure()
individual_indices = []
dates = returns.index.to_timestamp().strftime("%Y-%m-%d").tolist()
for pool_id, group in members.groupby("pool_id", sort=True):
    label = f"{pool_id} · {names[pool_id]} ({len(group)})"
    cluster_map.add_trace(go.Scatter(
        x=group.mds_1.tolist(), y=group.mds_2.tolist(), text=group.index.tolist(),
        customdata=group[["economy", "silhouette", "monthly_volatility_pct"]].to_numpy().tolist(),
        mode="markers", name=label,
        marker=dict(size=11, color=colors[pool_id], line=dict(width=1, color="white")),
        textposition="top center", textfont=dict(size=10),
        hovertemplate="<b>%{text} · %{customdata[0]}</b><br>Silhouette: %{customdata[1]:.3f}<br>Monthly volatility: %{customdata[2]:.2f}%<extra>%{fullData.name}</extra>",
    ))
    for currency in group.index:
        individual_indices.append(len(patterns.data))
        patterns.add_trace(go.Scatter(
            x=dates, y=standardized[currency].tolist(), mode="lines", name=currency,
            visible=False, showlegend=False, line=dict(color=colors[pool_id], width=1), opacity=.22,
            hovertemplate=f"{currency}<br>%{{x|%b %Y}}: %{{y:.2f}} SD<extra></extra>",
        ))
    patterns.add_trace(go.Scatter(
        x=dates, y=standardized[group.index].mean(axis=1).tolist(), name=label, mode="lines",
        line=dict(color=colors[pool_id], width=2.5),
        hovertemplate="%{x|%b %Y}<br>Group mean: %{y:.2f} SD<extra>%{fullData.name}</extra>",
    ))
legend = dict(orientation="h", x=0, y=1.08, font=dict(size=11), itemclick="toggle", itemdoubleclick="toggleothers")
style(cluster_map, 430, legend=legend, xaxis_title="MDS 1", yaxis_title="MDS 2")
cluster_map.update_yaxes(scaleanchor="x", scaleratio=1)
style(patterns, 430, legend=legend, yaxis_title="Standardized monthly return")
patterns.update_xaxes(rangeslider_visible=False, dtick="M24", tickformat="%Y")

descriptions = {
    1: f"AUD is representative, but JPY and TRY also belong. Shared USD moves do not make a distinct pool: {stats[1]['negative']} of 22 currencies have negative silhouettes.",
    2: "ARS, CNY and GBP form a weak group. GBP has silhouette −0.205; Argentina’s monthly volatility is 6.10%, versus China’s 1.06%. Shape similarity does not imply equal risk.",
    3: "The clearest group. DKK, BAM and MKD each correlate above 0.95 with EUR returns in this window. The pattern is consistent with common EUR exposure; all 14 silhouettes are positive.",
}
cards = "".join(
    f'<article class="cluster-card" style="--cluster:{colors[pool]}">'
    f'<div class="cluster-id">CLUSTER {pool} · {value["n"]} CURRENCIES</div>'
    f'<h3>{names[pool]}</h3>'
    f'<div class="cluster-metrics">Silhouette <strong>{value["silhouette"]:.3f}</strong>'
    f'<span>Median ρ <strong>{value["correlation"]:.2f}</strong></span></div>'
    f'<p>{descriptions[pool]}</p><div class="tickers">{html.escape(value["members"])}</div></article>'
    for pool, value in stats.items()
)
improvement = 100 * (1-selection.loc[3, "inertia"] / selection.loc[2, "inertia"])
next_improvement = 100 * (1-selection.loc[4, "inertia"] / selection.loc[3, "inertia"])
figures = {"inertia": inertia, "silhouette": silhouette, "map": cluster_map, "patterns": patterns}
figure_json = "{" + ",".join(json.dumps(name) + ":" + fig.to_json() for name, fig in figures.items()) + "}"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Currency clusters for pooled training</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#F4F6F8;color:#1B2D43;font:14px/1.55 Arial,sans-serif}
main{max-width:1130px;margin:auto;padding:36px 28px 28px}header{margin-bottom:22px}
.eyebrow{font-size:11px;letter-spacing:.12em;color:#64758A;font-weight:700}h1{font-size:32px;letter-spacing:-.7px;line-height:1.15;margin:8px 0 10px}
.subtitle{color:#5B6B7C;margin:0}h2{font-size:20px;letter-spacing:-.3px;margin:0}h3{font-size:17px;margin:7px 0 12px}
.takeaway{padding:12px 16px;border-left:3px solid #168574;background:#EAF3F1;margin-top:18px}
section{background:white;border:1px solid #E0E6EC;border-radius:9px;margin-top:16px;padding:20px 22px}
.section-head{display:flex;align-items:baseline;gap:12px;margin-bottom:8px}.number{color:#8A99A9;font-size:12px;font-weight:700}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:22px}.plot-label{font-size:12px;font-weight:700;margin:8px 0 0}.plot-caption{font-size:11px;color:#66768A;margin:0}
.reason{border-top:1px solid #EBEFF3;padding-top:12px;margin:14px 0 0;color:#42566D}.reason strong{color:#20364E}
.controls{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin:12px 0 0}
.tabs{display:flex;gap:4px;padding:3px;background:#F0F3F6;border-radius:6px}.tab{border:0;background:transparent;padding:7px 12px;font:600 12px Arial;color:#65758A;border-radius:4px;cursor:pointer}.tab.active{background:white;color:#173650;box-shadow:0 1px 3px #162B4020}
label{font-size:12px;color:#53667B;cursor:pointer}input{vertical-align:middle;margin:0 5px 0 0;accent-color:#2B6CB0}
.view[hidden],label[hidden]{display:none}.plot{width:100%}.figure-note{font-size:11px;color:#697A8D;margin:5px 0 0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:15px}.cluster-card{border-top:3px solid var(--cluster);padding-top:12px;min-width:0}
.cluster-id{font-size:10px;letter-spacing:.08em;color:var(--cluster);font-weight:700}.cluster-metrics{font-size:11px;color:#5B6E82;display:flex;gap:16px}.cluster-metrics strong{color:#20374D}
.cluster-card p{font-size:12px;line-height:1.65;color:#4A5E73;margin:12px 0}.tickers{font-size:10px;line-height:1.8;color:#718093;border-top:1px solid #EDF0F4;padding-top:8px}
.conclusion{background:#EEF3F8;border:0;padding:15px 20px;font-size:13px}.conclusion strong{color:#193953}
footer{font-size:10px;line-height:1.7;color:#758497;margin-top:15px}footer a{color:#5D7892}button:focus-visible,input:focus-visible{outline:2px solid #2B6CB0;outline-offset:3px}
@media(max-width:720px){main{padding:22px 14px}h1{font-size:26px}section{padding:16px}.plots,.cards{grid-template-columns:1fr}.plots{gap:10px}.cards{gap:22px}.cluster-card{padding-top:10px}.controls{align-items:flex-start}.cluster-card p{margin:9px 0}}
@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}.controls,.modebar{display:none}footer{font-size:9px}}
</style><script>__PLOTLY__</script></head><body><main>
<header><div class="eyebrow">MONTHLY BIS FX RETURNS · __WINDOW__</div>
<h1>Currency clusters for pooled training</h1>
<p class="subtitle">39 currencies · standardized log returns · DTW allows shifts of up to two months</p>
<div class="takeaway"><strong>Three groups are a working choice.</strong> The EUR-centered group is the strongest candidate for pooling; the other two are weakly separated.</div></header>
<section><div class="section-head"><span class="number">01</span><h2>Why three clusters?</h2></div>
<div class="plots"><div><div class="plot-label">Compactness</div><div id="inertia" class="plot"></div><p class="plot-caption">Lower inertia means tighter groups. Extra clusters add flexibility.</p></div>
<div><div class="plot-label">Separation</div><div id="silhouette" class="plot"></div><p class="plot-caption">Higher is better. Line = median; bars and dots = three seeds.</p></div></div>
<p class="reason"><strong>K=2 → 3:</strong> inertia falls __IMPROVEMENT__%. <strong>K=3 → 4:</strong> only __NEXT__%. Median silhouette peaks at <strong>0.125</strong>, barely above K=2’s <strong>0.122</strong>. At K≥4, some fits contain just one or two currencies. <strong>The overlapping seed ranges make K=3 tentative.</strong></p></section>
<section><div class="section-head"><span class="number">02</span><h2>See the groups together</h2></div>
<div class="controls"><div class="tabs" role="tablist" aria-label="Cluster view"><button id="map-tab" class="tab active" role="tab" aria-selected="true" aria-controls="map-view">Currency map</button><button id="patterns-tab" class="tab" role="tab" aria-selected="false" aria-controls="patterns-view">Return patterns</button></div>
<label id="labels-control"><input id="labels" type="checkbox">Currency labels</label><label id="individual-control" hidden><input id="individual" type="checkbox">Show individual currencies</label></div>
<div id="map-view" class="view" role="tabpanel" aria-labelledby="map-tab"><div id="map" class="plot"></div><p class="figure-note">Hover for currency details; click a legend group to filter. MDS is a 2D approximation of the full DTW distances.</p></div>
<div id="patterns-view" class="view" role="tabpanel" aria-labelledby="patterns-tab" hidden><div id="patterns" class="plot"></div><p class="figure-note">Lines are group means of standardized monthly returns at their actual dates. These are calendar averages, not DTW barycenters.</p></div></section>
<section><div class="section-head"><span class="number">03</span><h2>What do the clusters mean?</h2></div><div class="cards">__CARDS__</div>
<p class="figure-note">ρ = median pairwise Pearson correlation within each group, using the observed monthly returns. Silhouettes describe the selected K=3 fit.</p></section>
<section class="conclusion"><strong>Next test:</strong> compare global pooling with the same model plus a EUR-group indicator on observations after 2019. Clustering alone does not establish forecast improvement.</section>
<footer>Method: 108 months per currency; returns centered and scaled within currency; DTW radius 2; three seeds × three initializations; selected fit uses seed 42 and the lowest K=3 inertia. Revised BIS history, with the notebook’s launch and peg masks. <a href="https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html">DTW method</a> · <a href="https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.silhouette_score.html">Silhouette</a></footer>
</main><script>
const specs=__FIGURES__, individualIndices=__INDICES__;
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d'],toImageButtonOptions:{format:'png',scale:2}};
const drawn=new Set();
async function draw(name){if(!drawn.has(name)){await Plotly.newPlot(name,specs[name].data,specs[name].layout,config);drawn.add(name)}else{Plotly.Plots.resize(name)}}
async function view(name){for(const v of ['map','patterns']){document.getElementById(v+'-view').hidden=v!==name;document.getElementById(v+'-tab').classList.toggle('active',v===name);document.getElementById(v+'-tab').setAttribute('aria-selected',String(v===name))}document.getElementById('labels-control').hidden=name!=='map';document.getElementById('individual-control').hidden=name!=='patterns';await draw(name)}
document.getElementById('map-tab').onclick=()=>view('map');
document.getElementById('patterns-tab').onclick=()=>view('patterns');
document.getElementById('labels').onchange=e=>Plotly.restyle('map',{mode:e.target.checked?'markers+text':'markers'});
document.getElementById('individual').onchange=e=>Plotly.restyle('patterns',{visible:e.target.checked},individualIndices);
Promise.all([draw('inertia'),draw('silhouette'),draw('map')]);
</script></body></html>"""

page = (PAGE.replace("__PLOTLY__", get_plotlyjs())
        .replace("__WINDOW__", f"{start} to {end}")
        .replace("__IMPROVEMENT__", f"{improvement:.0f}")
        .replace("__NEXT__", f"{next_improvement:.0f}")
        .replace("__CARDS__", cards)
        .replace("__FIGURES__", figure_json)
        .replace("__INDICES__", json.dumps(individual_indices)))
destination = OUT / "cluster_summary.html"
destination.write_text(page)
print(destination)
