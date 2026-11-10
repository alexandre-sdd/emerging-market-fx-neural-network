"""A short, offline findings brief backed by the saved clustering evidence."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fx_forecasting.data.report_theme import CSS
from fx_forecasting.evaluation.clustering_report import METHOD_NAMES


def _payload(folder: Path) -> dict:
    scores = pd.read_csv(folder/"comparison.csv")
    counts = pd.read_csv(folder/"sequence_counts.csv")
    stability = pd.read_csv(folder/"stability.csv")
    metadata = json.loads((folder/"dataset_metadata.json").read_text())
    currencies = metadata["currencies"]
    country_names = pd.read_csv(folder/"currency_metadata.csv").set_index("currency").loc[currencies, "economy"].to_dict()
    panel = pd.read_parquet(folder/"monthly_local.parquet", columns=["fx_change_1m", "inflation_diff"])
    history = panel.fx_change_1m.unstack("currency").loc["2011-01":"2019-12", currencies]
    raw = history.corr()
    adjusted = history.sub(history.median(axis=1), axis=0).corr()
    inflation = panel.inflation_diff.unstack("currency").loc["2011-01":"2019-12", currencies]
    missing_inflation = inflation.columns[inflation.isna().all()].tolist()
    quality = pd.read_csv(folder/"group_quality.csv").merge(pd.read_csv(folder/"group_recovery.csv"), on=["method", "k", "cluster"], how="left")
    groups = json.loads((folder/"groups.json").read_text())
    # One short-report sample definition; the full eligibility cube stays in CSV/PDF.
    train_counts = counts.loc[counts.view.eq("macro") & counts.horizon.eq(12) & counts.split.eq("train")]
    quality = quality.merge(train_counts[["method", "k", "cluster", "usable_origins"]], on=["method", "k", "cluster"], how="left")
    for key, definition in groups.items():
        if key == "all:k1":
            continue
        for group in definition["clusters"]:
            members = group["currencies"]
            outside = [c for c in currencies if c not in members]
            within = adjusted.loc[members, members].to_numpy()[np.triu_indices(len(members), 1)]
            row = quality.method.eq(definition["method"]) & quality.k.eq(definition["k"]) & quality.cluster.eq(group["cluster"])
            quality.loc[row, "adjusted_gap"] = within.mean() - adjusted.loc[members, outside].to_numpy().mean() if len(within) else np.nan
    quality["raw_gap"] = quality.within_raw_correlation - quality.outside_raw_correlation
    with np.load(folder/"bootstrap_coassignment.npz", allow_pickle=False) as archive:
        coassignment = {key: np.round(archive[key].astype(float), 3).tolist() for key in archive.files}
    score_columns = ["method", "k", "sizes", "silhouette", "within_correlation", "between_correlation",
                     "subwindow_ari", "bootstrap_mean_ari", "fit_start", "bootstrap_reps", "n_singletons"]
    bootstrap = {}
    for (method, k), frame in stability.loc[stability.kind.eq("bootstrap")].groupby(["method", "k"]):
        bootstrap[f"{method}:k{k}"] = frame.ari.round(5).tolist()
    available = [m for m in METHOD_NAMES if m in scores.method.unique()]
    return {
        "names": {m: METHOD_NAMES[m] for m in available},
        "groups": groups,
        "countryNames": country_names,
        "currencies": currencies, "correlation": raw.round(4).to_numpy().tolist(),
        "scoreColumns": score_columns,
        "scores": json.loads(scores[score_columns].to_json(orient="values", double_precision=6)),
        "bootstrap": bootstrap, "adjustedCorrelation": adjusted.round(4).to_numpy().tolist(),
        "coassignment": coassignment, "inflation": {"complete_currencies": int(inflation.notna().all().sum()),
            "currencies": len(currencies), "source_absent": missing_inflation},
        "quality": json.loads(quality.to_json(orient="records", double_precision=5)),
        "evolution": _evolution_payload(folder),
    }


def _evolution_payload(folder: Path) -> dict | None:
    path = folder/"evolution_metadata.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    with np.load(folder/"evolution_labels.npz", allow_pickle=False) as archive:
        metadata["labels"] = {key: archive[key].tolist() for key in archive.files}
    metadata["summary"] = json.loads(pd.read_csv(folder/"evolution_summary.csv").to_json(orient="records"))
    return metadata


def _evolution_html(data: dict) -> str:
    e = data.get("evolution")
    if not e:
        return ''
    methods = ''.join(f'<option value="{m}"'+(' selected' if m=='r2k_w6' else '')+f'>{escape(n)}</option>' for m,n in e['names'].items())
    currencies = ''.join(f'<option value="{c}"'+(' selected' if c=='EUR' else '')+f'>{c} · {escape(data["countryNames"][c])}</option>' for c in e['currencies'])
    default = next(x for x in e['summary'] if x['method']=='r2k_w6' and x['k']==3)
    current = next(x for x in e['summary'] if x['method']=='r2k_w1' and x['k']==3)
    return f"""
<section id="evolution"><div class="eyebrow">01 / TA REVIEW · OBSERVED HISTORICAL EVOLUTION</div><h2>Which currencies stay together through time?</h2>
<p>R2K now runs continuously through the available monthly history. The five-year view counts how often a currency pair shares a group across <strong>60 actual monthly states</strong>. It uses no bootstrap and no artificial block joins. Monthly states begin {e['initialization']}; the 39-currency axis stays fixed, with gaps explicitly masked.</p>
<div class="insights"><article><h3>Monthly continuity is not long-run persistence</h3><p>For R2K K=3, W=6, mean ARI is <strong>{default['mean_ari_1m']:.3f} versus the preceding month</strong>, but <strong>{default['mean_ari_12m']:.3f} versus twelve months earlier</strong>. Summaries cover {e['summary_start']}–{e['summary_end']}, starting when all 39 currencies can support a 60-month return window.</p></article><article><h3>Longer memory does not guarantee fewer switches</h3><p>At K=3, monthly ID-switch rates average <strong>{current['mean_switch_rate']:.1%} for W=1</strong> and <strong>{default['mean_switch_rate']:.1%} for W=6</strong>. These are observed membership changes, not forecast errors. W changes classification against recent centroids; it does not change their fitted path.</p></article></div>
<div class="controls"><label><span>Method</span><select id="evolution-method">{methods}</select></label><label><span>Groups K</span><select id="evolution-k">{''.join(f'<option value="{k}"'+(' selected' if k==3 else '')+f'>{k}</option>' for k in e['k_values'])}</select></label><label><span>Five-year window endpoint</span><input id="evolution-end" type="range" min="59" max="{len(e['dates'])-1}" value="{len(e['dates'])-1}" step="1"></label></div>
<p id="evolution-window" class="explorer-title" aria-live="polite"></p><p id="evolution-stats" class="selection-note"></p>
<div class="plot-panel"><h3>Observed co-clustering during these five years</h3><div id="evolution-matrix" class="chart" role="img" aria-label="Five-year observed pair co-clustering frequencies"></div><p class="caption">White = never together; red = always together, conditional on joint eligibility. Hover shows the numerator and denominator. This is a historical fraction, not a probability of future membership. The currency order stays fixed as the window moves.</p></div>
<div class="controls"><label><span>Inspect a currency’s companions</span><select id="evolution-currency">{currencies}</select></label></div>
<div class="table-scroll"><table class="data-table"><thead><tr><th>Companion</th><th>Country / economy</th><th>Months together</th><th>Jointly eligible months</th><th>Fraction</th></tr></thead><tbody id="evolution-companions"></tbody></table></div>
<details><summary>Monthly groups at this window’s endpoint</summary><div id="evolution-groups" class="group-grid"></div><p class="caption">R2K IDs follow its centroid paths. Correlation-refit IDs use maximum-overlap matching only for display. Pair frequencies do not depend on numeric group labels.</p></details>
<div class="plot-panel"><h3>Monthly membership history</h3><div id="evolution-timeline" class="chart" role="img" aria-label="Chronological currency membership history"></div><p class="caption">The outlined interval is the selected five-year window. Blank cells are ineligible observations. Colors identify groups within this method/K, not fixed economic types across different settings.</p></div>
<details><summary>Three different windows, and the mathematical definition</summary><p><strong>Feature window:</strong> R2K uses 12 observed monthly returns to describe annualized FX change and volatility. <strong>Centroid memory W:</strong> current observations are labeled against the latest W monthly centroid states; the paper recommends roughly 6–12 increments. <strong>Reporting window:</strong> 60 monthly labels form the five-year co-clustering frequency.</p><p>For currencies i,j and endpoint t: <strong>frequency = eligible months with label(i) = label(j) / months when both labels are observed</strong>, over t−59,…,t. A missing pair never counts as a disagreement. Early windows can have fewer than 60 usable observations; their denominators remain visible.</p><p>The correlation alternatives genuinely refit on trailing 60-month return histories at each endpoint, requiring complete returns per included currency. R2K instead follows the paper’s continuous monthly update. These methods organize different objects: shared FX movements versus relative return/volatility states.</p><p>Adjacent five-year windows overlap by 59 months, so apparent smoothness is partly mechanical. Inspect 12- and 60-month partition agreement, coverage and group sizes. High co-clustering can also arise from one oversized group. A five-year restart is an initialization sensitivity, not the paper’s main update rule.</p></details>
<p><strong>Scope:</strong> this is a descriptive study including post-2019 observations. It does not establish forecast improvement or provide an untouched forecasting holdout. Dynamic labels do not change the exported neural tensor’s currency identities.</p>
<div class="links"><a class="download" href="evolution_metrics.csv">Monthly diagnostics ↓</a><a class="download" href="evolution_pairs.csv">Pair frequencies over the stated summary span ↓</a><a class="download" href="evolution_initialization.csv">Initialization / restart audit ↓</a></div></section>
<section id="paper-audit"><div class="eyebrow">02 / PAPER IMPLEMENTATION AUDIT</div><h2>Check the update rule before interpreting stability.</h2>
<p>An independent NumPy implementation of Algorithm 2 reproduces <strong>135 subsequent updates</strong> of the production model across K=2–4 and W=1, 6, 12. It independently computes seed selection, Lloyd means and historical-bank classification. Existing regression checks cover ties, duplicate locations, chronology, missing assets, convergence and fixed scaling.</p>
<details><summary>What was verified, and what the paper does not guarantee</summary><div class="table-scroll"><table class="data-table"><thead><tr><th>Paper rule</th><th>Implementation / audit</th></tr></thead><tbody>
<tr><td>Initialize once; update with prior centroids</td><td>First cross section uses seeded k-means++; later seeds snap previous centers to distinct current feature locations.</td></tr><tr><td>Small clusters choose first</td><td>Previous current-Lloyd counts determine selection priority. Sorting priority does not rename historical cluster IDs.</td></tr><tr><td>Rolling classification</td><td>Minimum distance over the latest W centroid states; no averaging of centers or voting over labels. Current state is included.</td></tr><tr><td>Chronology and normalization</td><td>Features use trailing observed returns. Current cross-sectional z-scores are an explicit project choice; old centers retain their original normalized coordinates.</td></tr><tr><td>Convergence and degeneracy</td><td>A genuine fixed point at the iteration limit is accepted. Failed or collapsed updates do not commit model state.</td></tr><tr><td>Limits and interpretation</td><td>W does not guarantee lower turnover. A reproducible unchanged-data counterexample changes the local solution after snapping; the paper’s universal invariance statement is too strong.</td></tr></tbody></table></div><p>The paper is the algorithmic reference, not an executable oracle. No author-maintained reference implementation was identified in the search. Deterministic tie rules, duplicate-location handling, initial solver and cross-sectional scaling are documented implementation choices. Passing a finite set of independent replays is evidence of fidelity, not a proof for all inputs.</p><p>R2K’s “robust” refers to update/label stability; its squared-distance loss still permits outlier groups. Preserve singleton findings and inspect eligibility, economic meaning and oversized groups. Frozen December 2019 snapshots are only one point on the paths above.</p><p><a href="https://ssrn.com/abstract=4729393">Holmes et al., supplied 6 March 2024 version</a>: Algorithm 2, p.4; centroid memory §5.2, p.5; selection order §5.3, pp.6–7. The complete audit, numerical results and limitations are in the PDF.</p></details></section>
"""


EXTRA_CSS = """
:root{--teal:#176b80}header{padding:46px max(5vw,24px) 38px}header h1{font-size:clamp(32px,4.4vw,58px);max-width:970px;letter-spacing:-1.5px}header .lede{font-size:19px;max-width:880px}main{max-width:1240px}section{padding:38px 0}h2{font-size:29px}nav{gap:24px}p{max-width:980px}
.chart{border:0;margin:16px 0;min-height:330px}.plot-panel{background:white;border:1px solid var(--line);padding:20px;border-radius:8px}.controls{margin:16px 0;gap:15px}.controls label{display:block}.controls label>span{display:block;margin-bottom:5px;color:var(--ink)}select{max-width:100%}button{cursor:pointer;border:1px solid #b7c8d0;background:white;color:var(--ink);border-radius:5px;padding:9px 13px;font:600 13px Arial,sans-serif}button:hover{border-color:var(--teal);color:var(--teal)}:focus-visible{outline:3px solid #ddaa52;outline-offset:3px}
.muted{color:var(--muted);font-size:13px}.group-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}.group-card{background:white;border:1px solid var(--line);border-top:3px solid var(--teal);border-radius:6px;padding:18px}.group-card h3{margin:0 0 5px}.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}.chip{font:12px/1.5 ui-monospace,monospace;background:#edf3f5;padding:3px 7px;border-radius:4px}.explorer-title{font-weight:700;font-size:15px}.selection-note{border-left:3px solid var(--teal);padding-left:13px;font-size:14px;max-width:1000px}.badge{font-size:11px;font-weight:700;letter-spacing:.6px;background:#e8f2f4;color:var(--teal);border-radius:20px;padding:5px 9px}
.country-names{padding:10px 12px;margin:14px 0 0;background:#f8fafb}.country-names summary{font-size:13px}.country-list{display:grid;grid-template-columns:3.5em minmax(0,1fr);gap:6px 12px;margin:12px 0 0;font-size:13px}.country-list dt{font:600 13px ui-monospace,monospace}.country-list dd{margin:0}
.math-explainer{max-width:980px;padding:12px 16px}.math-explainer summary{font-size:14px}.math-scroll{overflow-x:auto;padding:12px;background:#f5f8fa;border-radius:4px;margin:12px 0}.math-scroll math{font-size:1.05rem}.math-explainer ol{font-size:14px;padding-left:22px}.math-explainer li{margin:6px 0}
.data-table td{font-variant-numeric:tabular-nums}.data-table caption{caption-side:top;text-align:left;padding:12px;font-size:14px;font-weight:700;color:var(--ink)}.quality-table{min-width:770px}.quality-table small{white-space:nowrap}.data-table tr.selected-row td,.data-table tr.selected-row th{background:#e8f2f4;font-weight:700}.matrix-table th,.matrix-table td{text-align:center}.matrix-table td{font-size:15px;font-weight:700;padding:13px}
details{border:1px solid var(--line);border-radius:7px;background:white;padding:16px 20px}summary{font-size:16px;color:var(--ink)}details p{font-size:14px}.skip-link{position:absolute;left:-9999px}.skip-link:focus{left:12px;top:10px;background:white;padding:10px;z-index:40}.error{padding:16px;background:#fff0e9;color:#8b3825}footer{letter-spacing:.3px;font-size:12px}
@media(max-width:700px){main{padding:0 16px}.group-grid{grid-template-columns:1fr}nav{gap:20px;padding:12px 16px}section{padding:30px 0}header h1{letter-spacing:-1px}.plot-panel{padding:12px}.controls{padding:12px;align-items:stretch}.controls label{flex:1 1 130px}.controls select{width:100%;font-size:13px}.chart{overflow:hidden}summary{font-size:15px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{nav,.controls,button,.modebar{display:none!important}header{background:white;color:var(--ink);padding:20px 0}header .eyebrow,header .meta{color:var(--muted)}main{max-width:none}.plot-panel,details{break-inside:avoid}section{padding:20px 0}.chart{break-inside:avoid}.group-grid{grid-template-columns:1fr 1fr}}
"""


MEAN_ARI_MATH = '<div class="math-scroll"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Mean block ARI equals one divided by B times the sum of ARI b from b equals one to B."><mrow><mover accent="true"><mi>ARI</mi><mo>¯</mo></mover><mo>=</mo><mfrac><mn>1</mn><mi>B</mi></mfrac><munderover><mo>∑</mo><mrow><mi>b</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><msub><mi>ARI</mi><mi>b</mi></msub></mrow></math></div>'


RUNTIME = r"""
'use strict';
const D=JSON.parse(document.getElementById('clustering-data').textContent);
const byId=id=>document.getElementById(id), fmt=(n,d=3)=>n==null?'Not evaluated':Number(n).toFixed(d);
const htmlEscape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const scores=D.scores.map(row=>Object.fromEntries(D.scoreColumns.map((key,i)=>[key,row[i]])));
const letter=n=>String.fromCharCode(65+Number(n));
const colors=['#176b80','#bd7b35','#67866b','#8b70a6','#bc6570','#67849b'];
// Explicit stops avoid the opposite RdBu ordering in Plotly.js and Python.
const correlationColors=[[0,'#2166ac'],[.25,'#92c5de'],[.5,'#f7f7f7'],[.75,'#f4a582'],[1,'#b2182b']];
const frequencyColors=[[0,'#f7f7f7'],[.5,'#f4a582'],[1,'#b2182b']];
const config={responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'],toImageButtonOptions:{format:'png',scale:2}};
const base={paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'Arial, sans-serif',size:12,color:'#203343'},margin:{l:65,r:25,t:20,b:55},hoverlabel:{bgcolor:'white'},showlegend:false};
const visible=new Set(), plotted=new Set();
function state(){return {method:byId('method').value,k:Number(byId('groups-k').value)};}
function definition(){const s=state();return D.groups[`${s.method}:k${s.k}`];}
function score(method,k){return scores.find(x=>x.method===method&&x.k===k);}
function draw(id,data,layout){if(!visible.has(id))return;Plotly.react(id,data,{...base,...layout},config).then(()=>plotted.add(id)).catch(e=>{byId(id).classList.add('error');byId(id).textContent='Unable to draw this chart: '+e.message;});}
function matrixCellColor(value,frequency){
 const palette=frequency?frequencyColors:correlationColors,x=Math.max(0,Math.min(1,frequency?value:(value+1)/2));
 const right=palette.findIndex(p=>p[0]>=x),hi=Math.max(1,right),a=palette[hi-1],b=palette[hi];
 const mix=(x-a[0])/(b[0]-a[0]),rgb=[1,3,5].map(pos=>Math.round(parseInt(a[1].slice(pos,pos+2),16)*(1-mix)+parseInt(b[1].slice(pos,pos+2),16)*mix));
 const linear=rgb.map(c=>{c/=255;return c<=.04045?c/12.92:((c+.055)/1.055)**2.4;});
 const luminance=.2126*linear[0]+.7152*linear[1]+.0722*linear[2];
 return `background:rgb(${rgb.join(',')});color:${luminance<.179?'#fff':'#000'}`;
}
function matrixNumbers(matrix,groups,frequency){
 const indices=groups.map(g=>g.currencies.map(c=>D.currencies.indexOf(c)));
 const labels=groups.map(g=>`${letter(g.cluster)} · ${g.currencies.length} currencies`);
 let html='<caption>Mean pairwise '+(frequency?'coassignment frequencies':'correlations')+'</caption><thead><tr><th scope="col">Group</th>'+labels.map(x=>`<th scope="col">${x}</th>`).join('')+'</tr></thead><tbody>';
 groups.forEach((g,i)=>{html+=`<tr><th scope="row">${labels[i]}</th>`;groups.forEach((h,j)=>{
  const values=[];indices[i].forEach((a,ia)=>indices[j].forEach((b,ib)=>{if(i!==j||ia<ib)values.push(matrix[a][b]);}));
  if(!values.length){html+='<td title="No within-group pair exists for a singleton">N/A</td>';return;}
  const mean=values.reduce((a,b)=>a+b,0)/values.length;
  html+=`<td style="${matrixCellColor(mean,frequency)}" title="Mean over ${values.length} distinct currency pairs">${!frequency&&mean>0?'+':''}${mean.toFixed(3)}</td>`;
 });html+='</tr>';});byId('block-matrix').innerHTML=html+'</tbody>';
 const pairs=[];matrix.forEach((row,i)=>row.forEach((value,j)=>{if(i<j)pairs.push(value);}));pairs.sort((a,b)=>a-b);
 const quantile=q=>{const position=(pairs.length-1)*q,lo=Math.floor(position),hi=Math.ceil(position);return pairs[lo]+(pairs[hi]-pairs[lo])*(position-lo);};
 byId('pair-summary').textContent=frequency?`Across ${pairs.length} distinct pairs, median coassignment is ${quantile(.5).toFixed(3)}.`:`Across ${pairs.length} distinct pairs: median r = ${quantile(.5).toFixed(3)}; middle 50% = ${quantile(.25).toFixed(3)} to ${quantile(.75).toFixed(3)}; ${pairs.filter(x=>x>0).length} positive and ${pairs.filter(x=>x<0).length} negative. The fixed −1 to +1 scale is unchanged.`;
}
function heatmap(){
 const s=state(),g=definition(),key=`${s.method}:k${s.k}`,view=byId('matrix-view').value;
 let matrix=view==='adjusted'?D.adjustedCorrelation:view==='coassignment'?D.coassignment[key]:D.correlation;
 const available=!!matrix,isFrequency=view==='coassignment'&&available;
 if(!matrix)matrix=D.correlation;
 matrixNumbers(matrix,g.clusters,isFrequency);
 const label=isFrequency?'Bootstrap coassignment frequency':view==='adjusted'?'Median-adjusted return correlation':'Original return correlation';
 byId('matrix-title').textContent=label+', ordered by selected groups';
 byId('matrix-caption').textContent=available?(isFrequency?'Coassignment uses 0 (white) to 1 (red), matching the positive half of the correlation palette. Frequency across 50 twelve-month time-block resamples is a sensitivity measure, not a probability of true membership.':'Fixed correlation scale: −1 blue · 0 white · +1 red, for every method and K. Same training calendar, 2011–2019. Hover for exact values.'): 'Group-specific resampling was computed for the four correlation methods only. Showing original returns with the fixed scale: −1 blue · 0 white · +1 red.';
 const order=g.clusters.flatMap(x=>x.currencies),indices=order.map(x=>D.currencies.indexOf(x));
 const z=indices.map(i=>indices.map(j=>matrix[i][j]));
 let offset=0;const shapes=[];g.clusters.slice(0,-1).forEach(group=>{offset+=group.currencies.length;for(const axis of ['x','y'])shapes.push(axis==='x'?{type:'line',x0:offset-.5,x1:offset-.5,y0:-.5,y1:38.5}:{type:'line',y0:offset-.5,y1:offset-.5,x0:-.5,x1:38.5});});
 draw('correlation-chart',[{type:'heatmap',z,x:order,y:order,zmin:isFrequency?0:-1,zmax:1,zauto:false,colorscale:isFrequency?frequencyColors:correlationColors,reversescale:false,xgap:.3,ygap:.3,colorbar:{thickness:22,len:.9,outlinewidth:1,outlinecolor:'#dce4e8',ticks:'outside',tickmode:'array',tickvals:isFrequency?[0,.25,.5,.75,1]:[-1,-.5,0,.5,1],ticktext:isFrequency?['0','0.25','0.50','0.75','1']:['−1','−0.5','0','+0.5','+1'],tickfont:{size:13,color:'#203343'},title:{text:isFrequency?'Frequency':'Correlation',side:'top',font:{size:13}}},hovertemplate:'%{y} × %{x}<br>'+label+': %{z:.3f}<extra></extra>'}],{
 height:Math.max(390,Math.min(690,byId('correlation-chart').clientWidth*.84)),margin:{l:52,r:95,t:18,b:60},
 xaxis:{type:'category',tickangle:-90,tickfont:{size:9},constrain:'domain',fixedrange:true},
 yaxis:{type:'category',autorange:'reversed',tickfont:{size:9},fixedrange:true},
 shapes:shapes.map(x=>({...x,line:{color:'#263d49',width:1.4}}))});
}
function evidence(){
 const s=state(),refs=byId('references').checked;
 const methods=Object.keys(D.names).filter(m=>D.bootstrap[`${m}:k${s.k}`]&&(refs||m!=='levels_average'));
 const data=methods.map(m=>({type:'box',orientation:'h',x:D.bootstrap[`${m}:k${s.k}`],name:D.names[m],
 marker:{color:m===s.method?'#176b80':'#b9cbd3'},line:{color:m===s.method?'#176b80':'#708a98'},boxpoints:false,showlegend:false,
 hovertemplate:D.names[m]+'<br>Block ARI: %{x:.3f}<extra></extra>'}));
 data.push({type:'scatter',mode:'markers',x:methods.map(m=>score(m,s.k).subwindow_ari),y:methods.map(m=>D.names[m]),
 marker:{symbol:'diamond-open',color:'#a16a24',size:10,line:{width:2}},name:'Subwindow agreement',showlegend:true,
 hovertemplate:'%{y}<br>Subwindow ARI: %{x:.3f}<extra></extra>'});
 draw('stability-chart',data,{height:Math.max(340,methods.length*52+60),margin:{l:135,r:22,t:12,b:62},
 xaxis:{title:{text:'Membership agreement (ARI)'},range:[-.12,1.04],zeroline:true,gridcolor:'#edf1f3',fixedrange:true},
 yaxis:{autorange:'reversed',automargin:true,fixedrange:true},legend:{orientation:'h',x:0,y:-.22},boxmode:'group'});
 const rows=scores.filter(x=>x.k===s.k&&(refs||!x.method.startsWith('r2k')&&x.method!=='levels_average'));
 byId('evidence-body').innerHTML=rows.map(x=>`<tr><td>${D.names[x.method]}${x.method===s.method?' <strong>•</strong>':''}</td><td>${JSON.parse(x.sizes).join(' / ')}</td><td>${fmt(x.subwindow_ari)}</td><td>${fmt(x.bootstrap_mean_ari)}</td><td>${fmt(x.silhouette)}</td></tr>`).join('');
 byId('evidence-caption').textContent=`K=${s.k}. Boxes show 50 twelve-month block resamples (middle half and whiskers; outliers hidden). Diamonds compare the overlapping 2011–2015 and 2015–2019 subwindows. R2K is a snapshot reference, without a bootstrap estimate.`;
}
function kEvidence(){
 const s=state(),rows=scores.filter(x=>x.method===s.method).sort((a,b)=>a.k-b.k);
 const custom=rows.map(x=>[JSON.parse(x.sizes).join(' / '),x.n_singletons]);
 const data=[['silhouette','Silhouette','#176b80','y'],['bootstrap_mean_ari','Mean block ARI','#bd7b35','y2']].map(([key,name,color,axis])=>({
  type:'scatter',mode:'lines+markers',x:rows.map(x=>x.k),y:rows.map(x=>x[key]),name,yaxis:axis,customdata:custom,
  line:{color,width:2.5},marker:{color,size:rows.map(x=>x.k===s.k?12:7)},
  hovertemplate:'K=%{x}<br>'+name+': %{y:.3f}<br>Sizes: %{customdata[0]}<br>Singletons: %{customdata[1]}<extra></extra>'}));
 draw('k-chart',data,{height:365,margin:{l:65,r:20,t:15,b:55},
  xaxis:{title:{text:'Number of groups K'},dtick:1,range:[1.8,6.2],fixedrange:true,anchor:'y2'},
  yaxis:{title:{text:'Silhouette'},domain:[.56,1],range:[-.05,1],gridcolor:'#edf1f3',fixedrange:true},
  yaxis2:{title:{text:'Block ARI'},domain:[0,.42],range:[-.1,1],gridcolor:'#edf1f3',fixedrange:true}});
 byId('k-caption').textContent=`${D.names[s.method]} only. Silhouette uses this method’s distance space; ARI measures membership recovery. Both panels retain the same scales across methods. Missing block estimates are left blank. Highlighted points mark K=${s.k}.`;
 byId('k-body').innerHTML=rows.map(x=>`<tr${x.k===s.k?' class="selected-row"':''}><th scope="row">${x.k}</th><td>${JSON.parse(x.sizes).join(' / ')}</td><td>${fmt(x.silhouette)}</td><td>${fmt(x.bootstrap_mean_ari)}</td><td>${x.n_singletons}</td></tr>`).join('');
 const values=(D.bootstrap[`${s.method}:k${s.k}`]||[]).slice().sort((a,b)=>a-b);
 const quantile=q=>{const p=(values.length-1)*q,lo=Math.floor(p),hi=Math.ceil(p);return values[lo]+(values[hi]-values[lo])*(p-lo);};
 byId('mean-block-ari-results').textContent=values.length?`${D.names[s.method]}, K=${s.k}: B=${values.length} resamples; mean ${fmt(score(s.method,s.k).bootstrap_mean_ari)}; median ${fmt(quantile(.5))}; 10th percentile ${fmt(quantile(.1))}; range ${fmt(values[0])}–${fmt(values[values.length-1])}.`:`${D.names[s.method]}, K=${s.k}: mean block ARI was not evaluated. Missing estimates are not zero agreement.`;
}
function currencyCards(groups){return groups.map(group=>`<article class="group-card" style="border-top-color:${colors[group.cluster]}"><h3>Group ${letter(group.cluster)} <span class="badge">${group.currencies.length} currencies</span></h3><div class="muted">${group.representative?'Medoid '+group.representative+' · dataset ID':'Monthly state · persistent ID'} ${group.cluster}</div><div class="chips">${group.currencies.map(c=>`<span class="chip">${c}</span>`).join('')}</div><details class="country-names"><summary>Country names</summary><dl class="country-list">${group.currencies.map(c=>`<dt>${c}</dt><dd>${htmlEscape(D.countryNames[c])}</dd>`).join('')}</dl></details></article>`).join('');}
function summary(){
 const s=state(),g=definition(),v=score(s.method,s.k),signed=n=>n==null?'N/A':(n>0?'+':'')+fmt(n);
 byId('selection-label').textContent=`${D.names[s.method]} · K=${s.k}`;
 byId('groups-cards').innerHTML=currencyCards(g.clusters);
 byId('quality-body').innerHTML=g.clusters.map(group=>{
  const q=D.quality.find(x=>x.method===s.method&&x.k===s.k&&x.cluster===group.cluster);
  return `<tr><th scope="row">${letter(group.cluster)} / ${group.representative}</th><td>${q.within_raw_correlation==null?'N/A':fmt(q.within_raw_correlation)}</td><td title="Within ${fmt(q.within_raw_correlation)} minus outside ${fmt(q.outside_raw_correlation)}">${signed(q.raw_gap)}</td><td>${signed(q.adjusted_gap)}</td><td title="${q.negative_silhouettes} of ${q.n_currencies} currencies have negative silhouette">${fmt(q.mean_silhouette)} <small>(${q.negative_silhouettes})</small></td><td>${q.mean_jaccard==null?'Not evaluated':fmt(q.mean_jaccard)}</td><td>${q.usable_origins}</td></tr>`;
 }).join('');
 byId('selection-note').textContent=`Fitted ${g.fit_start}–${g.fit_end}; frozen memberships. Overall silhouette ${fmt(v.silhouette)}; mean block ARI ${fmt(v.bootstrap_mean_ari)}. Group letters are specific to this method and K.`;
 byId('method-note').textContent=s.method.startsWith('r2k')?'R2K groups recent annual FX-change/volatility states. Its complete state history starts July 2011; the frozen December 2019 snapshot describes regimes, rather than synchronous co-movement.':s.method==='levels_average'?'Level correlation compares persistent FX paths. Trends can dominate: its large silhouette is not comparable with return-space silhouettes.':s.method==='dtw'?'DTW k-means compares standardized return shapes with up to two months of alignment. This differs from synchronous co-movement.':s.method.startsWith('adjusted')?'Subtract the monthly cross-currency median, then standardize each currency history. The grouping uses residuals; forecasts still target original FX changes.':'Standardized nominal return histories compare same-month FX/USD movements, including common dollar variation.';
}
function evolution(){
 if(!D.evolution)return;
 const e=D.evolution,method=byId('evolution-method').value,k=Number(byId('evolution-k').value),end=Number(byId('evolution-end').value),start=end-59;
 const path=e.labels[`${method}:k${k}`],cs=e.currencies,n=cs.length,den=Array.from({length:n},()=>Array(n).fill(0)),num=den.map(r=>r.slice());
 for(let t=start;t<=end;t++)for(let i=0;i<n;i++)if(path[t][i]>=0)for(let j=0;j<n;j++)if(path[t][j]>=0){den[i][j]++;num[i][j]+=path[t][i]===path[t][j]?1:0;}
 const frequency=num.map((row,i)=>row.map((v,j)=>den[i][j]?v/den[i][j]:null)),pairs=[];
 for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(den[i][j])pairs.push(frequency[i][j]);pairs.sort((a,b)=>a-b);
 const groups=Array.from({length:k},(_,cluster)=>({cluster,currencies:cs.filter((c,i)=>path[end][i]===cluster)})).filter(g=>g.currencies.length);
 byId('evolution-window').textContent=`${e.names[method]} · K=${k} · ${e.dates[start]}–${e.dates[end]} (60 calendar months)`;
 const median=pairs.length?(pairs[Math.floor((pairs.length-1)/2)]+pairs[Math.ceil((pairs.length-1)/2)])/2:null;
 byId('evolution-stats').textContent=`Endpoint group sizes: ${groups.map(g=>g.currencies.length).join(' / ')}; ${path[end].filter(x=>x>=0).length}/39 eligible currencies. Median observed pair frequency: ${fmt(median)}. High frequencies may reflect a large catch-all group; inspect membership sizes as well.`;
 byId('evolution-groups').innerHTML=currencyCards(groups);
 draw('evolution-matrix',[{type:'heatmap',z:frequency,x:cs,y:cs,zmin:0,zmax:1,zauto:false,colorscale:frequencyColors,
 customdata:num.map((row,i)=>row.map((v,j)=>[v,den[i][j]])),colorbar:{title:{text:'Fraction together'},tickvals:[0,.25,.5,.75,1],thickness:20},
 hovertemplate:'%{y} × %{x}<br>Together %{customdata[0]} / %{customdata[1]} eligible months<br>Fraction %{z:.3f}<extra></extra>'}],
 {height:Math.max(400,Math.min(650,byId('evolution-matrix').clientWidth*.8)),margin:{l:52,r:100,t:12,b:60},xaxis:{tickangle:-90,tickfont:{size:9},fixedrange:true},yaxis:{autorange:'reversed',tickfont:{size:9},fixedrange:true}});
 const focus=cs.indexOf(byId('evolution-currency').value);
 const companions=cs.map((c,j)=>({c,j,f:frequency[focus][j],n:den[focus][j],same:num[focus][j]})).filter(x=>x.j!==focus&&x.n).sort((a,b)=>b.f-a.f||b.n-a.n||a.c.localeCompare(b.c)).slice(0,10);
 byId('evolution-companions').innerHTML=companions.map(x=>`<tr><td>${x.c}</td><td>${htmlEscape(D.countryNames[x.c])}</td><td>${x.same}</td><td>${x.n}</td><td>${fmt(x.f)}</td></tr>`).join('')||'<tr><td colspan="5">No jointly eligible observations in this window.</td></tr>';
 const timeline=cs.map((c,i)=>path.map(row=>row[i]<0?null:row[i]));
 const palette=colors.slice(0,k).flatMap((c,i)=>[[i/k,c],[(i+1)/k,c]]);
 draw('evolution-timeline',[{type:'heatmap',z:timeline,x:e.dates,y:cs,zmin:-.5,zmax:k-.5,zauto:false,colorscale:palette,
 colorbar:{title:{text:'Group ID'},tickvals:Array.from({length:k},(_,i)=>i),ticktext:Array.from({length:k},(_,i)=>letter(i)),thickness:20},
 hovertemplate:'%{y} · %{x}<br>Group ID %{z}<extra></extra>'}],{height:650,margin:{l:52,r:90,t:10,b:50},
 xaxis:{type:'date',fixedrange:true},yaxis:{autorange:'reversed',tickfont:{size:9},fixedrange:true},
 shapes:[{type:'rect',xref:'x',yref:'paper',x0:e.dates[start],x1:e.dates[end],y0:0,y1:1,fillcolor:'rgba(255,255,255,.18)',line:{color:'#203343',width:2}}]});
}
function update(){summary();heatmap();kEvidence();evidence();}
if(D.evolution){['evolution-method','evolution-k','evolution-currency'].forEach(id=>byId(id).addEventListener('change',evolution));byId('evolution-end').addEventListener('input',evolution);byId('show-fixed-reference').addEventListener('click',()=>{byId('fixed-reference').open=true;});}
['method','groups-k','references','matrix-view'].forEach(id=>byId(id).addEventListener('change',update));
function select(method,k){byId('method').value=method;byId('groups-k').value=String(k);update();}
byId('reset').addEventListener('click',()=>{byId('matrix-view').value='original';byId('references').checked=false;select('adjusted_average',2);});
byId('try-k3').addEventListener('click',()=>select('adjusted_average',3));
byId('try-ward4').addEventListener('click',()=>select('adjusted_ward',4));
const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){visible.add(entry.target.id);if(entry.target.id==='correlation-chart')heatmap();else if(entry.target.id==='k-chart')kEvidence();else if(entry.target.id.startsWith('evolution-'))evolution();else evidence();observer.unobserve(entry.target);}}},{rootMargin:'200px'});
['correlation-chart','k-chart','stability-chart',...(D.evolution?['evolution-matrix','evolution-timeline']:[])].forEach(id=>observer.observe(byId(id)));
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{heatmap();evolution();for(const id of plotted)Plotly.Plots.resize(id);},180);});
update();evolution();

"""


def write_clustering_html(output_dir: Path) -> list[str]:
    """Write the findings brief with compact explanations; full detail is in PDF/MD."""
    from plotly.offline import get_plotlyjs

    folder = Path(output_dir)
    data = _payload(folder)
    baseline_bootstrap = np.asarray(data["bootstrap"].get("adjusted_average:k2", []))
    stability_results = (
        f"<p><strong>Saved K=2 baseline, {len(baseline_bootstrap)} repetitions:</strong> "
        f"mean {baseline_bootstrap.mean():.3f}; median {np.median(baseline_bootstrap):.3f}; "
        f"10th percentile {np.quantile(baseline_bootstrap, .1):.3f}; "
        f"range {baseline_bootstrap.min():.3f}–{baseline_bootstrap.max():.3f}. "
        "The average hides some fragile resamples.</p>"
        if len(baseline_bootstrap) else "<p>Block-resample agreement was not evaluated for the saved baseline.</p>"
    )
    methods = ''.join(f'<option value="{escape(m)}"'+(' selected' if m == 'adjusted_average' else '')+
                      f'>{escape(name)}</option>' for m, name in data['names'].items())
    evolution_content = _evolution_html(data)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    content = f"""
<a class="skip-link" href="#findings">Skip to findings</a>
<header><div class="eyebrow">COLUMBIA · DEEP LEARNING · FINDINGS BRIEF</div>
<h1>Currency clusters<br>through time</h1>
<p class="lede">Five-year co-clustering histories, a paper-level R2K audit, and the implications for joint FX forecasting.</p>
<div class="meta">3–5 minute read · Technical business stakeholders &amp; deep learning TA · 10 November 2026</div></header>
<nav aria-label="Report sections"><a href="#evolution">Five-year evolution</a><a href="#paper-audit">R2K audit</a><a id="show-fixed-reference" href="#fixed-reference">Fixed-group handoff</a><a href="clustering_report.pdf">Detailed PDF</a></nav>
<main>
{evolution_content}
<details id="fixed-reference"><summary>Earlier fixed-group comparison and neural handoff</summary>
<section id="findings"><div class="eyebrow">01 / THE RESULT</div><h2>Stable groups, uneven separation.</h2>
<p>We compared eight approaches at K=2–6 on 39 currencies, using 2011–2019 training data. The saved baseline is median-adjusted return correlation with average linkage, K=2: <strong>25 and 14 currencies</strong>. Its mean block-resample agreement is <strong>0.784</strong>, but silhouette is only <strong>0.175</strong>. Membership is more convincing than sharp separation.</p>
<details id="block-agreement-math" class="math-explainer"><summary>How is block-resample agreement calculated?</summary>
<p><strong>Question:</strong> would the groups survive a different mix of historical episodes?</p>
<ol><li>Draw nine rolling 12-month blocks with replacement from the 108 training months. Blocks can repeat or be omitted; boundary blocks wrap from December 2019 to January 2011. All currencies use the same month draws.</li><li>Refit the same method and K on each resampled history, including its return adjustment and scaling. Compare the new partition with the original using the adjusted Rand index (ARI).</li></ol>
<p>Blocks retain consecutive-month relationships inside each stretch; joins are artificial. For the 39 currencies, there are <strong>P = 39 × 38 / 2 = 741 pairs</strong>. Let <strong>U</strong> count pairs grouped together originally, <strong>V</strong> pairs grouped together after resampling, and <strong>S</strong> pairs grouped together in both. With these group sizes fixed, the expected shared count under random assignment is <strong>E</strong>:</p>
<div class="math-scroll"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="E equals U times V divided by P. ARI equals S minus E divided by the quantity U plus V over two minus E."><mrow><mi>E</mi><mo>=</mo><mfrac><mrow><mi>U</mi><mi>V</mi></mrow><mi>P</mi></mfrac><mo>,</mo><mspace width="1em"/><mi>ARI</mi><mo>=</mo><mfrac><mrow><mi>S</mi><mo>−</mo><mi>E</mi></mrow><mrow><mfrac><mrow><mi>U</mi><mo>+</mo><mi>V</mi></mrow><mn>2</mn></mfrac><mo>−</mo><mi>E</mi></mrow></mfrac></mrow></math></div>
<p><strong>ARI = 1:</strong> identical memberships, even if group labels swap. <strong>Around 0:</strong> chance-level agreement. Negative values indicate less agreement than chance. Average the scores over B resamples:</p>
{MEAN_ARI_MATH}
{stability_results}
<p>This is sensitivity to the sampled training history, not a percentage of correctly classified currencies, a confidence interval or forecasting accuracy. A repeatable group can still be internally heterogeneous.</p></details>
<div class="insights"><article><h3>The Europe-heavy group is coherent</h3><p>Its 14 currencies have original-return correlation <strong>0.750 within</strong> versus <strong>0.442 outside</strong>. Group recovery (Jaccard) averages <strong>0.857</strong>.</p></article><article><h3>The broad group is a weak complement</h3><p>Its 25 currencies have <strong>0.448 within</strong> versus <strong>0.442 outside</strong>: a gap of just <strong>0.006</strong>, despite Jaccard recovery of <strong>0.912</strong>.</p></article></div>
<p><strong>Project implication:</strong> these are usable candidate groups for joint neural forecasting. Stable membership does not establish useful predictive sharing; restricting a network to its group may discard cross-group signals.</p>
<div class="links"><a class="download" href="clustering_report.pdf">Detailed PDF: methods &amp; reproducibility ↗</a><a class="download" href="comparison.csv" download>All results (CSV) ↓</a></div>
</section>
<section id="groups"><div class="eyebrow">02 / CURRENCIES &amp; SHARED MOVEMENT</div><h2>Read the numbers alongside the heatmaps.</h2>
<div class="controls"><label for="method"><span>Method</span><select id="method">{methods}</select></label><label for="groups-k"><span>Groups K</span><select id="groups-k">{''.join(f'<option value="{k}">{k}</option>' for k in range(2,7))}</select></label><button id="reset" type="button">Reset to saved baseline</button><button id="try-k3" type="button">Compare adjusted K=3</button></div>
<div id="selection-label" class="explorer-title" aria-live="polite"></div><p id="method-note" class="selection-note"></p>
<div id="groups-cards" class="group-grid"></div>
<div class="table-scroll"><table class="data-table quality-table"><caption>Group quality and usable history for this selection</caption><thead><tr><th>Group / medoid</th><th>Within r</th><th>Original Δr</th><th>Adjusted Δr</th><th>Silhouette (negative members)</th><th>Jaccard</th><th>Train origins</th></tr></thead><tbody id="quality-body"></tbody></table></div>
<p class="caption"><strong>Δr = mean correlation within the group − mean correlation to all other currencies.</strong> Within r uses original returns. Adjusted Δr uses median-adjusted returns. Jaccard measures group recovery after resampling; higher is more repeatable. Train origins use 12-month targets and macro + credit inputs. A singleton has no within-pair correlation (N/A).</p>
<p id="selection-note" class="muted" aria-live="polite"></p>
<div class="plot-panel"><h3 id="matrix-title">Original return correlations</h3><label for="matrix-view">Show <select id="matrix-view"><option value="original">Original return correlation</option><option value="adjusted">Median-adjusted correlation</option><option value="coassignment">Bootstrap coassignment</option></select></label>
<p id="pair-summary" class="selection-note" aria-live="polite"></p><div class="table-scroll"><table id="block-matrix" class="data-table matrix-table" aria-label="Numeric group-level pairwise averages"></table></div>
<p class="caption">Diagonal cells exclude self-correlations; off-diagonal cells average pairs across groups. Colors match the full heatmap. Lines mark the selected groups.</p>
<div id="correlation-chart" class="chart" role="img" aria-label="Interactive currency-pair correlation or bootstrap coassignment matrix"></div><p class="caption" id="matrix-caption"></p></div>
<details><summary>How to read the three views, and what about inflation?</summary><p><strong>Original returns</strong> answer whether currencies move together against USD. Across all 741 pairs, median correlation is 0.490; 734 are positive. <strong>Median-adjusted returns</strong> show what remains after subtracting the common monthly median; their median pair correlation is −0.017. This changes the question, rather than improving the original correlations. <strong>Coassignment</strong> shows how often a pair stays together in the 50 block resamples.</p><p>Neither correlation view controls for inflation. Lagged inflation differentials and REER gaps are neural inputs, not main grouping features. The annual inflation input has complete training-window coverage for {data['inflation']['complete_currencies']}/39 currencies; sources are absent for {', '.join(data['inflation']['source_absent'])}. Global USD/credit inputs are common to every currency.</p></details>
</section>
<section id="evidence"><div class="eyebrow">03 / WHY NOT MORE GROUPS?</div><h2>K=2 is a baseline, not an optimum.</h2>
<p>At K=3, adjusted average linkage splits the broad group into 14 and 11 currencies, leaving the Europe-heavy group unchanged. Silhouette falls from <strong>0.175 to 0.166</strong> and mean block ARI from <strong>0.784 to 0.673</strong>. The 14-currency subgroup gains usable history: <strong>180 versus 79</strong> training origins at 12 months. That is a data-coverage benefit, not stronger statistical separation.</p>
<div class="plot-panel"><h3>K=2–6 in the selected method</h3><div id="k-chart" class="chart" role="img" aria-label="Silhouette and membership stability by number of groups"></div><p id="k-caption" class="caption"></p>
<div class="table-scroll"><table class="data-table"><thead><tr><th>K</th><th>Group sizes</th><th>Silhouette</th><th>Mean block ARI</th><th>Singletons</th></tr></thead><tbody id="k-body"></tbody></table></div>
<details id="mean-block-ari-math" class="math-explainer"><summary>What does mean block ARI mean?</summary>
<p><strong>This is the same quantity called “block-resample agreement” above.</strong> Fix a method and K. In resample b, refit the grouping on synchronized 12-month blocks, then compare its entire partition with the original training fit. That comparison gives ARI<sub>b</sub>; average over B resamples:</p>
{MEAN_ARI_MATH}
<p><strong>B</strong> is the number of resamples, and <strong>b</strong> identifies one repetition. Each ARI is chance-adjusted: 1 means identical grouping, around 0 means chance-level agreement, and negative values indicate less agreement than chance. Group labels can swap without changing the score.</p>
<p id="mean-block-ari-results" class="selection-note" aria-live="polite"></p>
<p>Higher mean ARI means more repeatable membership under this resampling scheme. It measures the <strong>whole partition</strong>; Jaccard recovery measures an individual group. The mean can hide fragile repetitions and does not measure group coherence or forecasting accuracy. The numbers above follow your selected method and K.</p></details></div>
<p>More groups can expose individual currencies instead of useful multivariate blocks. This occurs with adjusted average linkage at K=4–6. Ward K=4 avoids singletons, yet its six-currency group has silhouette <strong>−0.039</strong> and Jaccard <strong>0.279</strong>. <button id="try-ward4" type="button">Inspect Ward K=4</button></p>
<details id="robustness"><summary>Robustness: compare resampling and subwindow agreement</summary><p>Fifty twelve-month time-block resamples preserve short-run dependence. ARI compares each resulting partition with the training fit; diamonds compare the overlapping 2011–2015 and 2015–2019 fits. Stable assignments can still form a weak group.</p><label><input type="checkbox" id="references"> Include level and R2K references in the table</label>
<div id="stability-chart" class="chart" role="img" aria-label="Block-bootstrap stability distributions and subwindow agreement"></div><p id="evidence-caption" class="caption"></p>
<div class="table-scroll"><table class="data-table"><thead><tr><th>Method</th><th>Group sizes</th><th>Window ARI</th><th>Block ARI</th><th>Silhouette</th></tr></thead><tbody id="evidence-body"></tbody></table></div><p class="caption">Silhouettes are comparable within their representation and distance space. Large values for FX levels or moment states do not establish superior forecasting groups.</p></details>
<details id="methodology"><summary>Methodology: what was clustered and what the paper adds</summary><p>The objects are currencies, represented by 108 monthly FX changes. Original and median-adjusted histories use average linkage or Ward. DTW k-means allows up to two months of alignment. Level correlation and frozen Robust Rolling K-Means (R2K) moment states are interpretation references. All methods retain the same 39-currency roster; R2K’s complete state history begins July 2011 after its 12-month warm-up.</p><p>R2K groups annual FX change and volatility; both K=2 snapshots isolate ARS. The paper’s robustness concerns initialization and label continuity, not a trimmed loss that removes outliers. Genuine extremes remain in the panel. A singleton is a finding to explain, not sufficient grounds to delete a currency.</p><p>Silhouette measures within-versus-nearest-other distance. Jaccard is intersection/union with the best matching resampled group. ARI is chance-adjusted partition agreement. These diagnostics describe training-data structure; none measures neural forecast improvement. Formulas, the factor-reference analysis, exact timing rules and all method/K tables are in the PDF.</p></details>
</section>
<section id="next"><div class="eyebrow">04 / WHAT THIS MEANS FOR OUR PROJECT</div><h2>Test the grouping restriction before adopting it.</h2>
<p><strong>Prioritize the all-currency model and adjusted K=3 as the first forecasting comparison; keep K=2 as the stability control.</strong> The loader’s saved default remains K=2. Compare matched origins, targets, features and model capacity; then test the extra K=3 history separately. Keep the candidate budget small: the 12-month validation horizon has only 24 matured origins.</p>
<p>The handoff already provides fixed channels, 12-month inputs, 1/3/12-month targets and masks. Validation is 2020–2022; test is 2023–August 2026. Teammates still need to train the networks and compare per-currency forecast errors. Revised data, retrospective regime exclusions and overlapping samples limit inference.</p>
<div class="links"><a class="download" href="clustering_report.pdf">Full technical reference ↗</a><a class="download" href="clustering_report.md">Markdown source ↗</a><a class="download" href="groups.json">Memberships ↗</a><a class="download" href="sequence_counts.csv" download>All eligibility counts ↓</a><a class="download" href="manifest.json">Sources &amp; hashes ↗</a></div>
</section>
</details>
<noscript><aside>Enable JavaScript to explore memberships and charts. The findings above and the linked PDF/CSV results remain available.</aside></noscript>
</main><footer>Clustering diagnostics describe shared movement and stability. Neural forecasting improvement remains untested.</footer>
"""
    # One embedded Plotly bundle; no CDNs, webfonts, fetch calls or server required.
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Currency groups | findings brief</title><style>'+CSS+EXTRA_CSS+
                '</style></head><body>'+content+'<script type="application/json" id="clustering-data">'+payload+
                '</script><script>'+get_plotlyjs()+'</script><script>'+RUNTIME+'</script></body></html>')
    (folder/"clustering_report.html").write_text(document)
    return ["clustering_report.html"]
