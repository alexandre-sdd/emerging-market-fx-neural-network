"""Present the earlier BIS quality evidence using the shared report design."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from fx_forecasting.config import PROCESSED_DIR, PROJECT_ROOT
from fx_forecasting.data.brief_comparison import render_brief_comparison
from fx_forecasting.data.report_theme import render_document

NAMES={"fx":"Bilateral FX", "reer":"Real effective FX", "neer":"Nominal effective FX",
       "policy":"Policy rates", "cpi":"Consumer prices", "usd_credit":"USD credit"}


def chart_spec(fig,title,height=440):
    fig.update_layout(title=dict(text=title,font=dict(size=17)),height=height,
        template="plotly_white",font=dict(family="Arial, sans-serif",color="#203343"),
        paper_bgcolor="white",plot_bgcolor="white",colorway=["#147d92","#c35d38","#765ba7","#32875d","#b48c2e","#4273b4"],
        margin=dict(l=75,r=30,t=75,b=70),legend=dict(orientation="h",y=-.22))
    return json.loads(pio.to_json(fig))


def build_quality_report(destination_dir: Path, source_dir: Path | None = None):
    """Read frozen quality tables; do not rerun or modify the earlier analysis."""
    source_dir=Path(source_dir or PROJECT_ROOT/"reports"/"eda")
    destination_dir=Path(destination_dir)
    assets=destination_dir/"quality"
    assets.mkdir(parents=True,exist_ok=True)
    keys=("overview","missing_by_series","missing_gap_dates","raw_audit","nonpositive_values",
          "currency_overlap","shared_currency_codes","largest_fx_changes")
    tables={key:pd.read_csv(source_dir/f"{key}.csv") for key in keys}
    summary=json.loads((source_dir/"summary.json").read_text())
    choices=json.loads((PROJECT_ROOT/"config"/"research_sample.json").read_text())
    if choices["evidence"]["source_sha256"] != {key:value["sha256"] for key,value in summary["sources"].items()}:
        raise ValueError("Research choices and saved quality evidence refer to different source snapshots")
    for source in (PROJECT_ROOT/"config"/"research_sample.json",
                   PROJECT_ROOT/"docs"/"data"/"sample_and_treatment.md",
                   PROJECT_ROOT/"docs"/"project_brief.md"):
        if source.suffix==".md":
            portable=source.read_text(encoding="utf-8").replace(
                "](../../config/research_sample.json)","](research_sample.json)").replace(
                "](data/sample_and_treatment.md)","](sample_and_treatment.md)")
            (assets/source.name).write_text(portable,encoding="utf-8")
        else:
            shutil.copyfile(source,assets/source.name)
    for key in keys:
        shutil.copyfile(source_dir/f"{key}.csv",assets/f"{key}.csv")
    shutil.copyfile(source_dir/"summary.json",assets/"summary.json")
    overview=tables["overview"].set_index("dataset")
    profiles=tables["missing_by_series"]
    gaps=tables["missing_gap_dates"]
    overlap=tables["currency_overlap"]
    audit=tables["raw_audit"]
    duplicate_rows=int(audit.raw_exact_duplicate_rows.sum())
    duplicate_ids=int(audit.raw_duplicate_series_rows.sum())
    invalid=int(audit.invalid_numeric_tokens.sum())
    total=int(overview.observed.sum())
    internal=int(overview.internal.sum())
    policy=overview.loc["policy"]
    policy_boundary=100*(policy.leading+policy.trailing)/policy.missing
    dates=', '.join(sorted({m["downloaded_at_utc"][:10] for m in summary["sources"].values()}))
    shared_currencies=int(tables["shared_currency_codes"].currency.nunique())
    figures={};banks={"availability":{}}

    fig=go.Figure()
    matrices={}
    for key in NAMES:
        data=pd.read_csv(PROCESSED_DIR/f"eda_{key}.csv",index_col=0)
        data.index=pd.PeriodIndex(data.index,freq="Q" if key=="usd_credit" else "M")
        if int(data.notna().to_numpy().sum()) != int(overview.loc[key,"observed"]):
            raise ValueError(f"Saved {key} matrix differs from the earlier quality counts; refresh the initial review explicitly")
        matrices[key]=data
        if key!="usd_credit":
            fig.add_scatter(x=data.index.to_timestamp(),y=data.notna().sum(axis=1),name=NAMES[key],mode="lines")
    fig.update_yaxes(title="Series observed in month")
    figures["quality-coverage"]=chart_spec(fig,"Observed series through time")
    fig=go.Figure()
    for column,label in (("leading","Before history"),("internal","Inside history"),("trailing","After history")):
        fig.add_bar(y=[NAMES[k] for k in overview.index],x=100*overview[column]/overview.expected,
                    orientation="h",name=label)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="% of expected series-period cells")
    figures["quality-missing"]=chart_spec(fig,"Where the missing values occur")
    for key,data in matrices.items():
        # The map is a readable REER-covered subset of FX, not the 39-currency roster.
        ids=profiles.loc[profiles.dataset.eq(key)].sort_values("economy")
        if key=="fx":ids=ids.loc[ids.area.isin(overlap.loc[overlap.has_reer,"area"])]
        labels=ids.economy.tolist()
        fig=go.Figure(go.Heatmap(z=data[ids.series].isna().T.astype(int),
            x=data.index.to_timestamp(),y=labels,zmin=0,zmax=1,showscale=False,
            customdata=np.tile(data.index.astype(str),(len(ids),1)),
            colorscale=[[0,"#e2f0f1"],[.499,"#e2f0f1"],[.5,"#c35d38"],[1,"#c35d38"]],
            hovertemplate="%{y}<br>%{customdata}<br>Missing flag: %{z} (0 observed; 1 missing)<extra></extra>"))
        title=f"{NAMES[key]} · {len(ids)} series"
        if key=="fx":title+=" with broad REER overlap"
        banks["availability"][key]=chart_spec(fig,title,height=max(430,165+len(ids)*17))
    focus=gaps.loc[gaps.gap_type.eq("internal") & gaps.dataset.isin(["policy","cpi"])].copy()
    fig=go.Figure()
    for key in ("policy","cpi"):
        sub=focus.loc[focus.dataset.eq(key)]
        start=pd.PeriodIndex(sub.start,freq="M").to_timestamp()
        end=(pd.PeriodIndex(sub.end,freq="M")+1).to_timestamp()
        duration=(end-start).total_seconds()*1000
        fig.add_bar(y=sub.economy,x=duration,base=start,orientation="h",name=NAMES[key],
            customdata=sub[["start","end","periods"]],
            hovertemplate="%{y}<br>%{customdata[0]} → %{customdata[1]}<br>%{customdata[2]} missing months<extra></extra>")
    fig.update_xaxes(type="date")
    fig.update_layout(barmode="overlay")
    figures["quality-gaps"]=chart_spec(fig,"Internal macro gaps: exact calendar intervals")
    fig=go.Figure(go.Bar(x=[NAMES[k] for k in ("reer","neer","policy","cpi","usd_credit")],
        y=[int(overlap[f"has_{k}"].sum()) for k in ("reer","neer","policy","cpi","usd_credit")],
        text=[int(overlap[f"has_{k}"].sum()) for k in ("reer","neer","policy","cpi","usd_credit")],textposition="outside"))
    fig.update_yaxes(title="FX reference areas with a matching source",range=[0,75])
    figures["quality-overlap"]=chart_spec(fig,"Reference-area overlap across data sources")
    moves=tables["largest_fx_changes"].head(12).sort_values("abs_change_pct")
    fig=go.Figure(go.Bar(y=moves.economy+" · "+moves.period,x=moves.abs_change_pct,orientation="h",
        customdata=moves[["currency","monthly_change_pct"]],
        hovertemplate="%{y}<br>%{customdata[0]}<br>Simple monthly change: %{customdata[1]:,.2f}%<extra></extra>"))
    fig.update_xaxes(type="log",title="Absolute simple monthly change (%) · log axis")
    figures["quality-breaks"]=chart_spec(fig,"Largest observed source-universe FX moves",height=560)

    def table(frame,columns=None,limit=None):
        frame=frame if columns is None else frame[columns]
        frame=frame if limit is None else frame.head(limit)
        frame=frame.rename(columns=lambda c:c.replace("_"," ").capitalize())
        return '<div class="table-scroll">'+frame.to_html(index=False,border=0,classes="data-table",
            float_format=lambda n:f"{n:,.2f}",na_rep="—",escape=True)+'</div>'
    def plot(key,bank=None,selection=None):
        attrs=f'data-bank="{bank}" data-selection="{selection}"' if bank else f'data-figure="{key}"'
        return f'<div class="chart" id="{key}" {attrs}></div>'
    def download(key,label=None):
        return f'<a class="download" href="quality/{key}.csv" download>{html.escape(label or key.replace("_"," "))} ↗ CSV</a>'
    source_defs=pd.DataFrame([
        ["FX","192 areas / 147 currency codes","Monthly · end of month","LCU per USD; rising = depreciation"],
        ["REER / NEER","64 economies each","Monthly · period average","Broad indices; 2020 = 100; rising = appreciation"],
        ["Policy","49 national / euro-area histories","Monthly · end of month","Percent per annum; zero / negative rates can be valid"],
        ["CPI","63 economies","Monthly","Index; 2010 = 100; use inflation changes, not level comparisons"],
        ["USD credit","Outside-US aggregate + 12 countries","Quarterly · 2000Q1–2026Q1 observed","Millions of USD; total credit to non-banks"],
    ],columns=["Source","Reviewed universe","Timing","Definition and units"])
    priorities=pd.DataFrame([
        ["Currency roster","192 FX areas represent 147 currencies; a 39-currency design is selected locally.","Reconcile the intended roster and 35-versus-34 policy count with the brief.","#brief-comparison"],
        ["Policy rates",f"{int(policy.internal)} internal missing months; Japan accounts for 116. Most other missingness is at history boundaries.","Apply the selected short-carry flags and explicit long-gap treatment; avoid national euro splicing.","#brief-comparison"],
        ["US CPI","October 2025 is absent.","Check propagation into YoY inflation and lagged differentials shared by every country.","data_eda.html#credit-section"],
        ["Credit timing","2026Q2 is missing; 2026Q1 is the latest observed quarter.","Use an explicit two-quarter lag and compare credit/no-credit models on common samples.","data_eda.html#credit-section"],
        ["Large FX moves","Extreme changes are present and untrimmed.","Verify source breaks and denomination/regime changes within the selected sample.","data_eda.html#behavior"],
        ["Real-time availability","Source dates describe reference periods; exports are revised histories.","Validate release timing and fit transformations inside each training window.","data_eda.html#method"],
    ],columns=["Decision","Evidence","Implication","EDA section"])
    decisions='<div class="table-scroll"><table class="data-table decision-table"><thead><tr><th>Decision</th><th>Evidence</th><th>What it means for forecasting</th><th>Follow through</th></tr></thead><tbody>'
    for row in priorities.itertuples(index=False,name=None):
        label="Discussion ↗" if row[3].startswith("#") else "EDA ↗"
        decisions+='<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in row[:3])+f'<td><a href="{row[3]}">{label}</a></td></tr>'
    decisions+='</tbody></table></div>'
    opts=''.join(f'<option value="{key}" {"selected" if key=="policy" else ""}>{name}</option>' for key,name in NAMES.items())
    internal_rows=profiles.loc[profiles.internal_missing.gt(0)].sort_values("internal_missing",ascending=False)
    content=f'''
<header><div class="eyebrow">COLUMBIA · DEEP LEARNING · DATA QUALITY</div>
<h1>Emerging-market FX.<br><span>Is the data ready?</span></h1>
<p class="lede">The earlier BIS quality review, organised around the checks that determine whether a forecasting experiment is interpretable.</p>
<div class="meta">January 1994–August 2026 · BIS snapshot {html.escape(dates)} · earlier evidence preserved · interactive Plotly · works offline</div>
<div class="report-switch"><a href="data_quality.html" aria-current="page">01 / Data quality</a><a href="data_eda.html">02 / Data EDA</a><a href="research_choices.html">03 / Research choices</a></div></header>
<nav aria-label="Report sections"><a href="#findings">Findings</a><a href="#brief-comparison">Brief vs data</a><a href="#coverage">Coverage &amp; gaps</a><a href="#audit">Source integrity</a><a href="#mapping">Currency mapping</a><a href="#breaks">Large moves</a><a href="#decisions">Modeling decisions</a><a href="#sources">Sources</a><a href="data_eda.html">Data EDA ↗</a></nav>
<main><section id="findings"><div class="eyebrow">01 / READINESS</div><h2>The exports parse cleanly; study choices need reconciliation.</h2>
<p>The five BIS exports provide <b>{total:,}</b> observed values across <b>{int(overview.series.sum())}</b> selected series. This review covers the downloaded universe, rather than the planned 39-currency model sample. Missing data are left explicit; no imputation, clipping, peg exclusions or currency stitching were performed.</p>
<div class="kpis"><div><strong>{duplicate_rows}</strong><span>duplicate raw rows; {duplicate_ids} duplicate IDs</span></div><div><strong>{policy.missing_pct:.2f}%</strong><span>policy-rate cells missing¹</span></div><div><strong>{internal}</strong><span>internal missing cells across all sources</span></div><div><strong>2026Q1</strong><span>latest observed USD-credit quarter</span></div></div>
<div class="insights"><article><h3>Policy coverage is the main constraint</h3><p><b>{policy_boundary:.1f}%</b> of missing policy cells occur before or after a history, so the headline missing rate is largely a coverage issue. Japan’s <b>116 internal gaps</b> remain a separate concern for rate-differential features. <a href="#coverage">Inspect dates ↗</a></p></article>
<article><h3>One US CPI gap affects the panel</h3><p><b>October 2025</b> is missing. It affects inflation changes that use that level and, after lagging, multiple countries’ domestic-minus-US predictors. Complete FX coverage does not guarantee complete macro features. <a href="data_eda.html#credit-section">See feature availability ↗</a></p></article>
<article><h3>Economies are not distinct currencies</h3><p>The FX export spans <b>192 reference areas</b> and <b>147 currency codes</b>. Shared currency codes and national EUR backcasts require explicit representation choices. Source overlap cannot establish the final 39-currency roster. <a href="#mapping">Inspect mapping ↗</a></p></article>
<article><h3>Dates do not establish availability</h3><p>Credit is quarterly, macro series have different collection conventions, and the exports contain revised histories. A correct forecasting experiment still needs release rules and lags. <a href="#decisions">Review decisions ↗</a></p></article></div>
<p class="caption">¹ 4,504 missing policy cells / 19,208 expected cells; denominators include late starts and discontinued histories. Findings are preserved from the 8 October review, with a new presentation.</p>
<p><a href="#brief-comparison"><b>Jump to the closing discussion: executive instructions versus findings →</b></a></p></section>

<section id="coverage"><span id="3.-Missing-data-and-date-coverage"></span><div class="eyebrow">02 / COVERAGE &amp; MISSINGNESS</div><h2>Separate unavailable history from gaps inside it.</h2>
<p>Each monthly series is assessed on the same 392-month calendar. Quarterly credit is assessed on 106 completed quarters, 2000Q1–2026Q2. Late starts, internal gaps and missing tails answer different questions; a trailing gap can reflect discontinued coverage or publication delay.</p>
<div class="two-col">{plot('quality-coverage')}{plot('quality-missing')}</div>
{table(tables['overview'],['dataset','series','observed','missing','missing_pct','internal','complete_series','last'])}
<p>REER and NEER are complete for all 64 economies. Policy rates contain 164 internal missing months; CPI contains one. All 13 credit series are complete through 2026Q1, then have an absent 2026Q2 cell. No observation is inferred from the expected calendar.</p>
<div class="controls"><label>Source <select id="quality-component">{opts}</select></label><span>Teal = observed · orange = missing · hover for exact month and economy</span></div>
{plot('availability-map','availability','policy')}
<p class="caption">The FX map shows the 64 reference areas with broad REER coverage for readability; coverage totals above still use all 192 FX areas. Other maps show the full selected source universe. These maps do not choose a model sample.</p>
{plot('quality-gaps')}<h3>Internal gaps with the greatest implications</h3>
{table(internal_rows,['dataset','economy','internal_missing','longest_internal_gap'])}
<details><summary>Exact internal gap intervals</summary>{table(gaps.loc[gaps.gap_type.eq('internal')],['dataset','economy','start','end','periods'])}</details>
<div class="links">{download('overview')}{download('missing_by_series')}{download('missing_gap_dates')}</div></section>

<section id="audit"><span id="2.-Duplicate-and-parsing-checks"></span><div class="eyebrow">03 / SOURCE INTEGRITY</div><h2>No duplicate records or invalid numeric tokens were found.</h2>
<p>Audits cover all raw frequencies, before the project’s monthly and quarterly selections. They found <b>{duplicate_rows}</b> duplicate complete rows, <b>{duplicate_ids}</b> repeated series identifiers and <b>{invalid}</b> unexpected numeric tokens. Blank cells and BIS NaN markers are recognised as missing. Unique source series and time columns establish unique selected series-period keys.</p>
{table(audit,['dataset','raw_series_rows','raw_exact_duplicate_rows','raw_duplicate_series_rows','raw_blank_series_ids','invalid_numeric_tokens'])}
<p>Selected FX, effective-rate, CPI and credit values contain no nonpositive observations, supporting the log transformations used in the EDA. Policy rates are excluded from this positivity check because zero and negative rates can be valid.</p>
{table(tables['nonpositive_values'])}<aside>Passing these checks establishes parsing and key integrity. It does not validate economic regimes, historical currency definitions, publication timing or real-time availability.</aside>
<div class="links">{download('raw_audit')}{download('nonpositive_values')}</div></section>

<section id="mapping"><span id="4.-Currency-and-economy-overlaps"></span><div class="eyebrow">04 / UNITS &amp; MAPPING</div><h2>Match definitions before joining country panels.</h2>
{table(source_defs)}{plot('quality-overlap')}
<p>These are reference-area matches, rather than complete-history matches. Country USD credit exists for 12 areas; the outside-US aggregate is a separate global variable. A matching economy code does not establish a continuous, timely feature history. There are <b>{shared_currencies}</b> currency codes used by multiple FX reference areas.</p>
<aside>The EDA uses 42 distinct non-USD currencies with REER coverage, retains euro-area EUR from January 1999 and omits national EUR backcasts. The separate primary design selects 39 and records groups and peg exclusions. Its exact correspondence with the executives’ intended roster remains unconfirmed. <a href="#brief-comparison">See the scope comparison and first decisions ↗</a></aside>
<details><summary>Economy-to-currency mapping and source overlap</summary>{table(overlap,['area','economy','currency','has_reer','has_neer','has_policy','has_cpi','has_usd_credit'])}</details>
<details><summary>Currency codes shared across reference areas</summary>{table(tables['shared_currency_codes'],['area','economy','currency'])}</details>
<details><summary>Source ranges and first/last observations</summary>{table(profiles,['dataset','economy','first_in_sample','last_in_sample','min','median','max','units'])}</details>
<div class="links">{download('currency_overlap')}{download('shared_currency_codes')}</div></section>

<section id="breaks"><span id="6.-Large-FX-changes-to-review"></span><div class="eyebrow">05 / LARGE-MOVE FLAGS</div><h2>Extreme values need interpretation before treatment.</h2>
<p>The earlier review retained the largest monthly simple FX changes in the full 192-area universe. Some are enormous; a clean parser does not explain them. Currency denomination changes, regime transitions and source breaks require case-specific checks. These are flags, not proven errors or recommendations to trim observations.</p>
{plot('quality-breaks')}{table(tables['largest_fx_changes'],['period','economy','currency','monthly_change_pct'],10)}
<p class="caption">This earlier table uses simple percentage changes: 100 × (S_t / S_(t−1) − 1). The EDA uses log changes and a smaller provisional currency universe, so rankings and magnitudes differ. Missing months are never forward-filled.</p>
<div class="links">{download('largest_fx_changes')}<a class="download" href="data_eda.html#behavior">Tails and large moves in the exploratory sample ↗</a></div></section>

<section id="decisions"><div class="eyebrow">06 / WHAT MUST BE RESOLVED</div><h2>Define the panel before interpreting model gains.</h2>
{decisions}<div class="verdict"><strong>Readiness verdict:</strong> the exports support a reproducible descriptive study. Sample and treatment choices are documented; reconciling their correspondence with the executive brief and applying them in production are the next steps. Revised histories and assumed lags support pseudo-out-of-sample analysis, rather than a real-time vintage claim. <a href="#brief-comparison">Review what must be settled first →</a></div></section>

<section id="sources"><div class="eyebrow">07 / PROVENANCE</div><h2>One source snapshot, two complementary reports.</h2>
<p>This page reads the saved quality tables and descriptive matrices from the earlier BIS review. Restyling does not rerun that analysis or alter its values. The EDA report adds exploratory transformations, future targets, per-currency associations and benchmark diagnostics on a documented provisional sample.</p>
<p><b>Rebuild both pages:</b> <code>python scripts/run_interactive_eda.py</code>. To refresh the underlying initial quality analysis explicitly, run <code>python scripts/run_eda.py</code> first. Raw exports remain unchanged.</p>
<div class="links"><a class="download" href="quality/summary.json" download>Earlier selections &amp; source hashes ↗ JSON</a><a class="download" href="data_eda.html">Forecast-oriented data EDA →</a></div>
<p class="sources">Definitions: <a href="https://data.bis.org/bulkdownload">BIS bulk exports</a> · <a href="https://data.bis.org/topics/XRU">FX</a> · <a href="https://data.bis.org/topics/EER">EER</a> · <a href="https://data.bis.org/topics/CBPOL">Policy rates</a> · <a href="https://data.bis.org/topics/CPI">CPI</a> · <a href="https://data.bis.org/topics/GLI">Global liquidity</a>. The BIS broad USD NEER index is distinct from ICE DXY.</p></section>
{render_brief_comparison(overview,overlap,choices)}
</main><footer>EMERGING-MARKET FX NEURAL NETWORK · DATA QUALITY · EARLIER BIS SNAPSHOT</footer>'''
    path=destination_dir/"data_quality.html"
    path.write_text(render_document("Emerging-market FX · Data quality",content,figures,banks,get_plotlyjs()),encoding="utf-8")
    return path
