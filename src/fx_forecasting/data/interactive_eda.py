"""Forecast-oriented descriptive analysis, separate from the modeling pipeline.

Targets and timing conventions here are exploratory definitions, not a final
model-ready panel. No data are filled, clipped, or selected by correlation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from fx_forecasting.config import PROJECT_ROOT, RAW_DIR
from fx_forecasting.data.eda import (
    END, START, area_values, missing_profile, sample_values,
)
from fx_forecasting.data.loaders import load_project_datasets
from fx_forecasting.data.interactive_report import render_html
from fx_forecasting.data.quality_report import build_quality_report

PAPER_URL = "https://www.wu.ac.at/fileadmin/wu/d/ri/isk/VSFX_2021/Papers/Rapach_SmartCarry_07-02-2021.pdf"
FEATURES = {
    "fx_1m": "Latest FX move",
    "momentum_3m": "3m FX momentum",
    "volatility_12m": "12m FX volatility",
    "reer_gap": "REER gap (past 60m)",
    "rate_diff": "Policy differential (lag 1m)",
    "inflation_diff": "Inflation differential (lag 2m)",
    "usd_move": "Broad USD index move",
    "credit_growth": "USD credit YoY (lag 2Q)",
    "global_vol": "Global FX risk proxy",
    "rate_x_risk": "Policy differential × FX risk",
    "inflation_x_risk": "Inflation differential × FX risk",
}
WINDOWS = {"train": "Before 2010 (targets realised by Dec 2009)",
           "later": "Forecast origins from Jan 2010", "full": "Full available history"}
COLORS = ["#147d92", "#c35d38", "#765ba7", "#32875d", "#b48c2e", "#4273b4"]


def exploratory_sample(datasets, samples):
    """One currency per REER-covered reference area; EUR represented by XM."""
    meta = datasets["fx"].metadata
    eligible = meta.loc[meta.REF_AREA.isin(datasets["reer"].metadata.REF_AREA)].copy()
    eligible["keep"] = ~eligible.CURRENCY.eq("USD") & (
        ~eligible.CURRENCY.eq("EUR") | eligible.REF_AREA.eq("XM"))
    eligible["reason"] = np.where(eligible.CURRENCY.eq("USD"), "USD numeraire",
        np.where(eligible.CURRENCY.eq("EUR") & ~eligible.REF_AREA.eq("XM"),
                 "EUR represented by euro area; national backcasts omitted", "Exploratory comparison"))
    selected = eligible.loc[eligible.keep].sort_values("Reference area")
    if selected.CURRENCY.duplicated().any():
        raise ValueError("Explicit representative mapping needed for repeated currencies")
    fx = area_values(datasets["fx"], samples["fx"])[selected.REF_AREA].copy()
    # A synthetic pre-launch euro history is not a traded EUR time series.
    if "XM" in fx:
        fx.loc[fx.index < pd.Period("1999-01", "M"), "XM"] = np.nan
    return selected.set_index("REF_AREA", drop=False), eligible, fx


def make_features(fx, macro):
    logfx = np.log(fx.where(fx > 0))
    returns = 100 * logfx.diff()
    reer = np.log(macro["reer"].where(macro["reer"] > 0))
    inflation = 100 * np.log(macro["cpi"].where(macro["cpi"] > 0)).diff(12)
    rates = macro["policy"].sub(macro["policy"]["US"], axis=0).shift(1)
    infdiff = inflation.sub(inflation["US"], axis=0).shift(2)
    # A cross-currency median prevents one idiosyncratic extreme move from
    # defining global stress. This monthly proxy differs from the paper's
    # daily-data global volatility measure.
    global_vol = returns.rolling(12, min_periods=12).std().median(axis=1)
    credit_yoy = 100 * np.log(macro["usd_credit"]["3P"].where(macro["usd_credit"]["3P"] > 0)).diff(4)
    lookup = fx.index.asfreq("Q") - 2
    credit_monthly = pd.Series(credit_yoy.reindex(lookup).to_numpy(), index=fx.index)

    def broadcast(series):
        return pd.DataFrame({a: series for a in fx}, index=fx.index)

    features = {
        "fx_1m": returns,
        "momentum_3m": 100 * logfx.diff(3),
        "volatility_12m": returns.rolling(12, min_periods=12).std(),
        "reer_gap": 100 * (reer - reer.shift(1).rolling(60, min_periods=60).mean()),
        "rate_diff": rates,
        "inflation_diff": infdiff,
        "usd_move": broadcast(100 * np.log(macro["neer"]["US"]).diff()),
        "credit_growth": broadcast(credit_monthly),
        "global_vol": broadcast(global_vol),
        "rate_x_risk": rates.mul(global_vol, axis=0),
        "inflation_x_risk": infdiff.mul(global_vol, axis=0),
    }
    features = {k: v.reindex(index=fx.index, columns=fx.columns) for k, v in features.items()}
    targets = {h: 100 * (logfx.shift(-h) - logfx) for h in (1, 3, 12)}
    return returns, features, targets, global_vol, credit_monthly


def window_mask(index, horizon, window):
    if window == "train":
        return index + horizon <= pd.Period("2009-12", "M")
    if window == "later":
        return index >= pd.Period("2010-01", "M")
    return np.ones(len(index), dtype=bool)


def safe_corr(x, y, method="spearman", minimum=36):
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < minimum or pair.iloc[:, 0].std() < 1e-8 or pair.iloc[:, 1].std() < 1e-8:
        return np.nan, n
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method), n


def correlation_table(features, targets):
    rows = []
    for h, target in targets.items():
        for window in WINDOWS:
            mask = window_mask(target.index, h, window)
            for method in ("pearson", "spearman"):
                for feature, value in features.items():
                    for area in target:
                        rho, n = safe_corr(value.loc[mask, area], target.loc[mask, area], method)
                        rows.append(dict(horizon=h, window=window, method=method,
                                         feature=feature, area=area, correlation=rho, n=n))
    return pd.DataFrame(rows)


def currency_metrics(returns, meta):
    rows = []
    for window in ("full", "later"):
        r = returns.loc[window_mask(returns.index, 1, window)]
        for area in r:
            s = r[area].dropna()
            if s.empty:
                continue
            rho, _ = safe_corr(r[area], r[area].shift(1), "pearson")
            rows.append(dict(area=area, economy=meta.loc[area, "Reference area"],
                currency=meta.loc[area, "CURRENCY"], window=window, n=len(s),
                annual_log_drift=12*s.mean(), annual_volatility=np.sqrt(12)*s.std(),
                median=s.median(), skewness=s.skew(), excess_kurtosis=s.kurt(),
                lag1_autocorrelation=rho, p01=s.quantile(.01), p99=s.quantile(.99),
                max_abs_month=s.abs().max(), tail_gt10_pct=100*s.abs().gt(10).mean(),
                near_zero_pct=100*s.abs().lt(.01).mean()))
    return pd.DataFrame(rows)


def random_walk_metrics(targets, meta):
    """Zero-change error scale and an expanding historical-mean comparator.

    At t, y[t-h] is the most recent h-month label that has realised. The mean
    is learned only from those available labels, separately for each currency.
    """
    rows, paths = [], {}
    for h, y in targets.items():
        forecast = y.shift(h).expanding(min_periods=60).mean()
        mask = window_mask(y.index, h, "later")
        for a in y:
            pair = pd.DataFrame({"actual": y.loc[mask, a], "mean": forecast.loc[mask, a]}).dropna()
            if len(pair) < 36:
                continue
            actual, pred = pair.actual, pair["mean"]
            rw = actual.pow(2).mean()
            mse = (actual-pred).pow(2).mean()
            directional = actual.abs().gt(1e-8) & pred.abs().gt(1e-8)
            rows.append(dict(horizon=h, area=a, economy=meta.loc[a,"Reference area"],
                n=len(pair), first_origin=str(pair.index.min()), last_origin=str(pair.index.max()),
                rw_rmse=np.sqrt(rw), rw_mae=actual.abs().mean(), mean_rmse=np.sqrt(mse),
                mean_r2_os=100*(1-mse/rw) if rw > 1e-16 else np.nan,
                mean_direction_pct=100*(np.sign(actual[directional]) == np.sign(pred[directional])).mean(),
                directional_n=int(directional.sum())))
            paths[(h,a)] = (actual.pow(2) - (actual-pred).pow(2)).cumsum()
    return pd.DataFrame(rows), paths


def regime_correlations(features, targets, risk):
    threshold = risk.loc[:"2009-12"].quantile(.75)
    rows=[]
    for h, target in targets.items():
        for regime in ("normal", "high risk"):
            mask = (target.index >= pd.Period("2010-01", "M")) & risk.notna().to_numpy()
            mask &= (risk.gt(threshold) if regime == "high risk" else risk.le(threshold)).to_numpy()
            for k in ("rate_diff", "inflation_diff", "reer_gap", "credit_growth"):
                for a in target:
                    rho,n = safe_corr(features[k].loc[mask,a],target.loc[mask,a])
                    rows.append(dict(horizon=h,regime=regime,feature=k,area=a,correlation=rho,n=n))
    return pd.DataFrame(rows), float(threshold)


def styled(fig, title, ytitle=None, height=420):
    fig.update_layout(title=dict(text=title, font=dict(size=17)), height=height,
        template="plotly_white", font=dict(family="Arial, sans-serif", color="#203343"),
        paper_bgcolor="white", plot_bgcolor="white", colorway=COLORS,
        margin=dict(l=65,r=28,t=65,b=60), legend=dict(orientation="h",y=-.2),
        hovermode="closest")
    if ytitle:
        fig.update_yaxes(title=ytitle)
    return json.loads(pio.to_json(fig))


def build_report(output_dir: Path | None = None) -> Path:
    report_root = Path(output_dir) if output_dir else PROJECT_ROOT / "reports" / "eda"
    output = report_root / "interactive"
    output.mkdir(parents=True, exist_ok=True)
    earlier_reports = PROJECT_ROOT / "reports" / "eda"
    build_quality_report(report_root,source_dir=earlier_reports)
    datasets = load_project_datasets()
    if any(d.audit["invalid_numeric_tokens"] or d.audit["raw_duplicate_series_rows"] for d in datasets.values()):
        raise ValueError("Resolve invalid tokens or duplicate source series before analysis")
    samples = {k:sample_values(k,v) for k,v in datasets.items()}
    profiles = pd.concat([missing_profile(k,v,samples[k]) for k,v in datasets.items()],ignore_index=True)
    macro = {k:area_values(datasets[k],v) for k,v in samples.items() if k != "fx"}
    meta, selection, fx = exploratory_sample(datasets,samples)
    returns, features, targets, risk, credit_monthly = make_features(fx,macro)
    metrics = currency_metrics(returns,meta)
    corr = correlation_table(features,targets)
    baseline, baseline_paths = random_walk_metrics(targets,meta)
    regimes, threshold = regime_correlations(features,targets,risk)
    overview = profiles.groupby("dataset",sort=False).agg(series=("series","size"),
        observed=("observed","sum"),expected=("expected_periods","sum"),
        leading=("leading_missing","sum"),internal=("internal_missing","sum"),
        trailing=("trailing_missing","sum"),absent=("all_missing","sum"))
    overview["missing_pct"] = 100*(1-overview.observed/overview.expected)
    availability = pd.DataFrame({k:v.notna().sum(axis=1) for k,v in features.items()})
    complete_core = pd.concat({k:features[k].stack(future_stack=True) for k in
        ("fx_1m","momentum_3m","volatility_12m","reer_gap","rate_diff","inflation_diff","usd_move")},axis=1)
    core_counts = complete_core.notna().all(axis=1).groupby(level=0).sum()
    credit_core_counts = (complete_core.notna().all(axis=1) &
        features["credit_growth"].stack(future_stack=True).notna()).groupby(level=0).sum()
    extreme = returns.stack().rename("monthly_log_change").reset_index()
    extreme.columns=["period","area","monthly_log_change"]
    extreme["abs_change"] = extreme.monthly_log_change.abs()
    extreme["economy"] = extreme.area.map(meta["Reference area"])
    extreme = extreme.nlargest(20,"abs_change")
    # Rolling near-flat histories identify periods worth reviewing, not peg dates.
    flat = returns.abs().lt(.01).where(returns.notna()).rolling(60,min_periods=60).mean()*100
    flags = pd.DataFrame({"max_60m_near_zero_pct":flat.max(),
                          "annual_volatility":metrics.query("window == 'later'").set_index("area").annual_volatility})
    flags["review_flag"] = flags.max_60m_near_zero_pct.ge(95)

    figures, banks = make_figures(samples,macro,meta,fx,returns,features,targets,
        profiles,metrics,corr,baseline,baseline_paths,regimes,risk,threshold,core_counts,credit_core_counts,flat)
    tables = {"coverage":overview.reset_index(),"sample_mapping":selection.reset_index(),
        "missingness":profiles,"currency_metrics":metrics,"predictive_correlations":corr,
        "regime_correlations":regimes,"baseline_metrics":baseline,"extreme_moves":extreme,
        "peg_review_flags":flags.reset_index(),"feature_availability":availability,
        "core_feature_counts":pd.DataFrame({"core":core_counts,"core_and_credit":credit_core_counts})}
    for name, frame in tables.items():
        frame.to_csv(output/f"{name}.csv", index=name in {"feature_availability","core_feature_counts"})
    # Export descriptive panel with explicit period + area keys, not a modeling artifact.
    panel = pd.concat({**features,**{f"future_{h}m":v for h,v in targets.items()}},axis=1)
    panel.columns.names=["variable","area"]
    panel.stack(level="area",future_stack=True).rename_axis(index=["period","area"]).to_csv(output/"exploratory_panel.csv")
    manifest_path = RAW_DIR/"download_manifest.json"
    source_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source_hashes = {name:hashlib.sha256((RAW_DIR/f"{name}.csv").read_bytes()).hexdigest()
                     for name in {d.name for d in datasets.values()}}
    for name, checksum in source_hashes.items():
        if name in source_manifest and checksum != source_manifest[name].get("sha256"):
            raise ValueError(f"Source hash differs from recorded download: {name}")
    source = {"monthly_window":[START,END],"paper_url":PAPER_URL,
        "paper_version":"July 2, 2021 working paper", "source_hashes":source_hashes,
        "downloads":source_manifest,"plotly_version":__import__("plotly").__version__,
        "provisional_currencies":len(meta),"risk_threshold_pre2010_p75":threshold,
        "credit_alignment":"quarter containing origin minus two quarters",
        "correlation_min_pairs":36,"baseline_min_realised_labels":60}
    (output/"analysis_manifest.json").write_text(json.dumps(source,indent=2)+"\n")
    page = render_html(figures,banks,tables,meta,source,FEATURES,WINDOWS,get_plotlyjs())
    path=report_root/"data_eda.html"
    path.write_text(page,encoding="utf-8")
    write_findings(output,tables,source)
    return path


def make_figures(samples,macro,meta,fx,r,features,targets,profiles,metrics,corr,
                 baseline,baseline_paths,regimes,risk,threshold,core,credit_core,flat):
    figures, banks = {}, {"currency":{},"correlation":{},"baseline":{}}
    dates = fx.index.to_timestamp()
    labels = {a:f"{meta.loc[a,'CURRENCY']} · {a}" for a in fx}
    fig=go.Figure()
    for k in ("fx","reer","neer","policy","cpi"):
        fig.add_scatter(x=dates,y=samples[k].notna().sum(axis=1),name=k.upper(),mode="lines")
    figures["coverage"]=styled(fig,"Available monthly series in the source universe","Observed series")
    fig=go.Figure()
    totals=profiles.groupby("dataset").expected_periods.sum()
    for col,label in (("leading_missing","Before history"),("internal_missing","Internal gap"),
                      ("trailing_missing","After history"),("all_missing","All absent")):
        values=100*profiles.groupby("dataset")[col].sum()/totals
        fig.add_bar(y=values.index,x=values,orientation="h",name=label)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="% of expected series-period cells")
    figures["missing"]=styled(fig,"Missingness includes late starts and discontinued histories")

    for a in fx:
        fig=make_subplots(rows=2,cols=2,subplot_titles=("FX path · Jan 2010 = 100", "Monthly log changes",
            "Rolling 12m volatility (annualised)", "Monthly return autocorrelation"),vertical_spacing=.18)
        recent=fx.loc["2010-01":,a];base=recent.loc[pd.Period("2010-01","M")]
        fig.add_scatter(x=recent.index.to_timestamp(),y=100*recent/base,name="FX index",mode="lines",row=1,col=1)
        fig.update_yaxes(type="log",row=1,col=1,title="Index · log axis")
        fig.add_histogram(x=r[a].dropna(),nbinsx=45,name="All months",row=1,col=2)
        fig.update_xaxes(title="100 × Δlog(LCU/USD)",row=1,col=2)
        fig.add_scatter(x=dates,y=r[a].rolling(12,min_periods=12).std()*np.sqrt(12),
                        mode="lines",name="Volatility",row=2,col=1)
        fig.update_yaxes(title="Annualised log %",row=2,col=1)
        ac=[safe_corr(r[a],r[a].shift(lag),"pearson")[0] for lag in range(1,13)]
        fig.add_bar(x=list(range(1,13)),y=ac,name="Autocorrelation",row=2,col=2)
        fig.update_yaxes(range=[-1,1],row=2,col=2)
        fig.update_xaxes(title="Lag in calendar months",row=2,col=2)
        fig.update_layout(showlegend=False)
        banks["currency"][a]=styled(fig,f"{meta.loc[a,'Reference area']} · {meta.loc[a,'CURRENCY']}",height=660)

    recent=metrics.query("window == 'later'").set_index("area").loc[fx.columns]
    fig=go.Figure(go.Scatter(x=recent.annual_volatility,y=recent.annual_log_drift,mode="markers+text",
        text=[meta.loc[a,"CURRENCY"] for a in fx],textposition="top center",
        marker=dict(size=10,color=recent.excess_kurtosis,colorscale="YlOrRd",colorbar=dict(title="Excess<br>kurtosis")),
        customdata=np.column_stack([recent.economy,recent.n,recent.p01,recent.p99]),
        hovertemplate="%{customdata[0]}<br>Vol: %{x:.2f}<br>Drift: %{y:.2f}<br>N=%{customdata[1]}<br>1% / 99%: %{customdata[2]:.2f} / %{customdata[3]:.2f}<extra></extra>"))
    fig.update_xaxes(title="Annualised volatility (log %)")
    figures["dispersion"]=styled(fig,"Currency heterogeneity · monthly observations from Jan 2010","Annualised mean log change (%)",520)
    tails=recent.sort_values("excess_kurtosis",ascending=False).head(15)
    fig=go.Figure(go.Bar(x=[labels[a] for a in tails.index],y=tails.excess_kurtosis,
        customdata=np.column_stack([tails.tail_gt10_pct,tails.max_abs_month]),
        hovertemplate="%{x}<br>Excess kurtosis: %{y:.2f}<br>|move| > 10 log%: %{customdata[0]:.1f}%<br>Largest |move|: %{customdata[1]:.2f}<extra></extra>"))
    figures["tails"]=styled(fig,"Largest excess kurtosis · from Jan 2010","Excess kurtosis")

    later=r.loc["2010-01":]
    co=later.corr(min_periods=60)
    co.loc[later.std().lt(1e-8),:]=np.nan;co.loc[:,later.std().lt(1e-8)]=np.nan
    observed=later.notna().astype(int);counts=observed.T.dot(observed)
    fig=go.Figure(go.Heatmap(z=co,x=[labels[a] for a in fx],y=[labels[a] for a in fx],
        zmin=-1,zmax=1,colorscale="RdBu",reversescale=True,customdata=counts,
        hovertemplate="%{y} / %{x}<br>Pearson r=%{z:.3f}<br>Paired months=%{customdata}<extra></extra>"))
    figures["comovement"]=styled(fig,"Contemporaneous return correlation · from Jan 2010 (min 60 pairs)",height=770)
    candidates=later.columns[(later.notna().sum()>=160)&(later.std()>.05)]
    balanced=later[candidates].dropna()
    if len(balanced)<60 or len(candidates)<3:
        raise ValueError("Insufficient balanced history for the descriptive PCA")
    standard=(balanced-balanced.mean())/balanced.std()
    pca=PCA().fit(standard)
    pc=pca.explained_variance_ratio_*100
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Explained return variation", "First-component loadings"))
    fig.add_bar(x=[f"PC{i+1}" for i in range(min(10,len(pc)))],y=pc[:10],name="Explained %",row=1,col=1)
    load=pd.Series(pca.components_[0],index=candidates).sort_values()
    fig.add_bar(x=[labels[a] for a in load.index],y=load,name="PC1 loading",row=1,col=2)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Explained %",row=1,col=1)
    figures["pca"]=styled(fig,f"Common variation · {len(candidates)} currencies, {len(balanced)} complete months",height=500)
    figures["pca"]["layout"]["meta"]={"pc1_variance_pct":float(pc[0]),"n_currencies":len(candidates),"n_months":len(balanced)}

    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Global risk proxy (monthly log %)",
        "Median 36m correlation with the broad USD index"),vertical_spacing=.15)
    fig.add_scatter(x=dates,y=risk,name="Median rolling 12m volatility",mode="lines",row=1,col=1)
    fig.add_scatter(x=dates,y=np.repeat(threshold,len(dates)),name="Pre-2010 75th percentile",mode="lines",
        line=dict(dash="dot",color=COLORS[1]),row=1,col=1)
    dollar=100*np.log(macro["neer"]["US"]).diff()
    rolling=pd.DataFrame({a:r[a].rolling(36,min_periods=36).corr(dollar) for a in fx})
    fig.add_scatter(x=dates,y=rolling.median(axis=1),name="Median correlation",mode="lines",row=2,col=1)
    fig.update_yaxes(range=[-1,1],row=2,col=1)
    figures["risk"]=styled(fig,"Global conditions change over time",height=580)

    for h in targets:
        for window in WINDOWS:
            for method in ("spearman","pearson"):
                subset=corr.query("horizon == @h and window == @window and method == @method")
                z=subset.pivot(index="area",columns="feature",values="correlation").reindex(index=fx.columns,columns=FEATURES)
                n=subset.pivot(index="area",columns="feature",values="n").reindex_like(z)
                fig=go.Figure(go.Heatmap(z=z,x=list(FEATURES.values()),y=[labels[a] for a in fx],
                    zmin=-.6,zmax=.6,colorscale="RdBu",reversescale=True,customdata=n,
                    hovertemplate="%{y}<br>%{x}<br>Correlation=%{z:.3f}<br>Paired origins=%{customdata}<extra></extra>"))
                banks["correlation"][f"{h}|{window}|{method}"]=styled(fig,
                    f"Predictors at t vs future {h}m FX change · {method.title()}",height=820)
    summary=corr.query("horizon == 1 and method == 'spearman'").copy()
    summary["absolute"]=summary.correlation.abs()
    med=summary.groupby(["window","feature"]).absolute.median().unstack("window").reindex(FEATURES)
    fig=go.Figure()
    for window in ("train","later"):
        fig.add_bar(x=list(FEATURES.values()),y=med[window],name="Before 2010" if window=="train" else "From 2010")
    fig.update_layout(barmode="group")
    figures["stability"]=styled(fig,"Typical signal strength changes · median |Spearman ρ| across currencies","Median |ρ|",500)
    fig=go.Figure()
    for regime in ("normal","high risk"):
        s=regimes.query("horizon == 1 and regime == @regime").groupby("feature").correlation.median()
        fig.add_bar(x=[FEATURES[k] for k in s.index],y=s,name=regime.title())
    fig.update_layout(barmode="group")
    figures["regimes"]=styled(fig,"Stress-conditioned next-month relationships · from Jan 2010","Median signed Spearman ρ")

    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Outside-US credit stock (trillions USD)",
        "Year-on-year log growth (%)"),vertical_spacing=.15)
    q=macro["usd_credit"]["3P"];qd=q.index.to_timestamp(how="end")
    fig.add_scatter(x=qd,y=q/1_000_000,name="Quarterly stock",mode="lines+markers",row=1,col=1)
    fig.add_scatter(x=qd,y=100*np.log(q).diff(4),name="Quarterly YoY growth",mode="lines+markers",row=2,col=1)
    fig.add_scatter(x=dates,y=features["credit_growth"].iloc[:,0],name="Monthly predictor · lag 2Q",mode="lines",row=2,col=1)
    figures["credit"]=styled(fig,"Dollar credit: growth distinguishes changes from a trending stock",height=560)
    fig=go.Figure()
    fig.add_scatter(x=dates,y=core,name="Core features complete",mode="lines")
    fig.add_scatter(x=dates,y=credit_core,name="Core + lagged credit complete",mode="lines")
    for h,y in targets.items():
        eligible=complete_mask(features)&y.notna()
        fig.add_scatter(x=dates,y=eligible.sum(axis=1),name=f"Core + {h}m label",mode="lines",line=dict(dash="dot"))
    figures["usable"]=styled(fig,"Feature warm-up, macro gaps and horizon endpoints constrain the panel","Eligible currencies")
    fig=go.Figure(go.Heatmap(z=flat.T,x=dates,y=[labels[a] for a in fx],zmin=0,zmax=100,
        colorscale="YlOrRd",colorbar=dict(title="Near-zero<br>months (%)"),
        hovertemplate="%{y}<br>Window ending %{x|%Y-%m}<br>60m near-zero share=%{z:.1f}%<extra></extra>"))
    figures["pegs"]=styled(fig,"Near-flat FX periods · |monthly log change| < 0.01% over 60 months",height=750)

    for h in targets:
        s=baseline.query("horizon == @h").sort_values("rw_rmse",ascending=False)
        fig=make_subplots(rows=1,cols=2,subplot_titles=("Random-walk RMSE (log percentage points)",
            "Expanding historical mean: R² vs random walk (%)"))
        fig.add_bar(x=[labels[a] for a in s.area],y=s.rw_rmse,customdata=s.n,name="RW RMSE",
            hovertemplate="%{x}<br>RMSE=%{y:.2f}<br>Matched origins=%{customdata}<extra></extra>",row=1,col=1)
        fig.add_bar(x=[labels[a] for a in s.area],y=s.mean_r2_os,name="Mean R² OS",row=1,col=2)
        fig.update_layout(showlegend=False)
        banks["baseline"][str(h)]=styled(fig,f"{h}m horizon · forecast origins from Jan 2010 · matched labels",height=500)
    fig=go.Figure()
    for a in ("BR","MX","ZA","TR","JP","XM"):
        if (1,a) in baseline_paths:
            p=baseline_paths[(1,a)]
            fig.add_scatter(x=p.index.to_timestamp(),y=p,name=labels[a],mode="lines")
    figures["loss"]=styled(fig,"Cumulative squared-error saving: historical mean vs zero-change · 1m",
        "Σ(error² RW − error² mean)")
    return figures,banks


def complete_mask(features):
    masks=[features[k].notna() for k in ("fx_1m","momentum_3m","volatility_12m",
           "reer_gap","rate_diff","inflation_diff","usd_move")]
    result=masks[0].copy()
    for mask in masks[1:]:
        result &= mask
    return result


def write_findings(output,tables,source):
    recent=tables["currency_metrics"].query("window == 'later'")
    co=tables["predictive_correlations"].query("horizon == 1 and method == 'spearman' and window == 'train'").copy()
    ranking=co.assign(absolute=co.correlation.abs()).groupby("feature").absolute.median().sort_values(ascending=False)
    b=tables["baseline_metrics"].query("horizon == 1")
    lines=["# Forecast-oriented exploratory analysis", "",
        "Open `../data_eda.html` for the self-contained interactive Plotly report; `../data_quality.html` preserves the earlier quality review. Reproduce the EDA with `python scripts/run_interactive_eda.py`.", "",
        f"- Provisional sample: {source['provisional_currencies']} distinct non-USD currencies with broad REER coverage; this is not the agreed 39-currency roster.",
        f"- Median annualised FX volatility from Jan 2010: {recent.annual_volatility.median():.2f} log%; median absolute lag-1 correlation: {recent.lag1_autocorrelation.abs().median():.3f}.",
        f"- Strongest typical pre-2010 next-month Spearman association: {FEATURES[ranking.index[0]]}, median absolute correlation {ranking.iloc[0]:.3f} (pairwise available samples).",
        f"- Expanding historical-mean forecast beats zero-change MSPE for {int(b.mean_r2_os.gt(0).sum())}/{int(b.mean_r2_os.notna().sum())} evaluable currencies at 1m; median R² OS = {b.mean_r2_os.median():.2f}%.",
        "- Correlations are descriptive. Longer targets overlap, global predictors repeat across currencies, historical vintages are revised, and pegs remain in this exploratory sample.", "",
        "## Definitions", "",
        "FX target = 100 × [log(S at t+h) − log(S at t)], S = local currency units per USD. Positive means depreciation. Annualised volatility = monthly standard deviation × sqrt(12); annualised log drift = monthly mean × 12.", "",
        "Policy differentials use a one-month lag; CPI YoY log-inflation differentials use a two-month lag. REER gap compares the current log index with the previous 60 calendar months' mean. Credit YoY growth uses the quarter containing t minus two quarters, with no interpolation. These are exploratory availability assumptions, not validated release-date histories.", "",
        "Before-2010 correlations include only targets realised by December 2009. Expanding mean forecasts use only realised labels, with at least 60 observations. Zero-change directional accuracy is undefined; the mean-forecast directional metric omits zero target/forecast ties and reports its own denominator.", "",
        "## Reference and scope", "",
        f"[Filippou, Rapach, Taylor and Zhou, July 2, 2021 working paper]({PAPER_URL}). This report extends the research question to the BIS exploratory universe; it does not reproduce the paper's data or 70-predictor specification.", "",
        "CSV tables expose all sample mappings, sample sizes, tails, horizon metrics, and correlations. `analysis_manifest.json` records source hashes, snapshot metadata and timing conventions. No trained neural-network results or statistical significance claims are reported.", ""]
    (output/"README.md").write_text("\n".join(lines),encoding="utf-8")
