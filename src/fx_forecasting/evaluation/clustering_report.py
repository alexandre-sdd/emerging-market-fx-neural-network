"""Detailed Markdown/PDF reference and the accompanying short HTML brief."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_NAMES = {
    "returns_average": "Returns / average",
    "returns_ward": "Returns / Ward",
    "adjusted_average": "Adjusted / average",
    "adjusted_ward": "Adjusted / Ward",
    "dtw": "Constrained DTW",
    "levels_average": "Levels / average",
    "r2k_w1": "R2K (1-month window)",
    "r2k_w6": "R2K (6-month window)",
}
COLORS = ["#176b80", "#be792d", "#64846a"]


def _cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "-" if not np.isfinite(value) else f"{value:.3f}"
    return str(value)


def _table(frame: pd.DataFrame) -> str:
    def cell(value):
        return _cell(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def _frame(rows, columns, widths=None):
    frame = pd.DataFrame(rows, columns=columns)
    if widths is not None:
        frame.attrs["widths"] = widths
    return frame


def _figures(folder: Path, scores: pd.DataFrame, stability: pd.DataFrame,
             counts: pd.DataFrame, groups: dict) -> list[str]:
    """Retain the notebook's four figure names; use audience-facing labels."""
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "text.color": "#203d4b", "axes.titleweight": "bold"})
    produced = []
    key = "adjusted_average:k2"
    order = [c for g in groups[key]["clusters"] for c in g["currencies"]]
    raw = pd.read_parquet(folder/"monthly_local.parquet")
    returns = raw.fx_change_1m.unstack("currency").loc["2011-01":"2019-12", order]
    fig, ax = plt.subplots(figsize=(8, 6.4), constrained_layout=True)
    mesh = ax.imshow(returns.corr(), vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    boundary = len(groups[key]["clusters"][0]["currencies"])-.5
    ax.axhline(boundary, color="#203d4b", lw=1)
    ax.axvline(boundary, color="#203d4b", lw=1)
    ax.set_title("Same-month FX movements within the proposed groups", pad=12)
    fig.colorbar(mesh, ax=ax, shrink=.8, label="Return correlation, 2011-2019")
    name = "group_correlations.png"
    fig.savefig(folder/name, dpi=180); plt.close(fig); produced.append(name)

    methods = [m for m in ("returns_average", "returns_ward", "adjusted_average", "adjusted_ward", "dtw")
               if m in scores.method.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    for ax, k in zip(axes, (2, 3)):
        values = [stability.loc[stability.method.eq(m) & stability.k.eq(k)
                                & stability.kind.eq("bootstrap"), "ari"].to_numpy() for m in methods]
        boxes = ax.boxplot(values, showfliers=False, patch_artist=True)
        for patch, method in zip(boxes["boxes"], methods):
            patch.set_facecolor(COLORS[0] if method == "adjusted_average" else "#c4d4da")
        ax.set_xticks(np.arange(1, len(methods)+1),
                      [METHOD_NAMES[m].replace(" / ", "\n").replace("Constrained ", "") for m in methods])
        ax.set(title=f"{k} groups", ylabel="Membership agreement (ARI)", ylim=(-.1, 1.05))
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=.2)
    name = "cluster_stability.png"
    fig.savefig(folder/name, dpi=180); plt.close(fig); produced.append(name)

    chosen = scores.loc[scores.k.eq(2)].set_index("method")
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    bottom = np.zeros(len(chosen))
    for j in range(2):
        sizes = [json.loads(s)[j] for s in chosen.sizes]
        ax.barh([METHOD_NAMES[m] for m in chosen.index], sizes, left=bottom,
                color=COLORS[j], label=f"Group {'AB'[j]}")
        for i, size in enumerate(sizes):
            ax.text(bottom[i]+size/2, i, str(size), ha="center", va="center", color="white", fontsize=9)
        bottom += sizes
    ax.invert_yaxis(); ax.set(xlabel="Number of currencies", title="Two-group solutions have very different practical sizes")
    ax.legend(loc="lower right")
    name = "group_sizes.png"
    fig.savefig(folder/name, dpi=180); plt.close(fig); produced.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for ax, group in zip(axes, groups[key]["clusters"]):
        selected = counts.loc[counts.method.eq("adjusted_average") & counts.k.eq(2)
                              & counts.cluster.eq(group["cluster"]) & counts.view.eq("macro")]
        x = np.arange(3)
        for i, split in enumerate(("train", "validation", "test")):
            values = selected.loc[selected.split.eq(split)].set_index("horizon").reindex([1, 3, 12]).usable_origins
            ax.bar(x+(i-1)*.24, values, width=.24, label=split.capitalize(), color=COLORS[i])
        ax.set(xticks=x, xticklabels=[1, 3, 12], xlabel="Forecast horizon (months)", ylabel="Usable monthly origins",
               title=f"Group {'AB'[group['cluster']]}: {len(group['currencies'])} currencies")
    axes[0].legend(fontsize=9)
    name = "sequence_counts.png"
    fig.savefig(folder/name, dpi=180); plt.close(fig); produced.append(name)
    return produced


def _temporal_blocks(folder: Path, *, render_figures: bool = True) -> tuple[list, list[str]]:
    """Observed evolution and a paper-level implementation audit, ahead of frozen groups."""
    from fx_forecasting.evaluation.clustering_evolution import pair_frequencies, NAMES
    from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
    meta = json.loads((folder/'evolution_metadata.json').read_text())
    summary = pd.read_csv(folder/'evolution_summary.csv')
    audit = pd.read_csv(folder/'evolution_initialization.csv')
    dates, currencies = meta['dates'], meta['currencies']
    with np.load(folder/'evolution_labels.npz', allow_pickle=False) as archive:
        path = archive['r2k_w6:k3']
    unavailable = ', '.join(c for c,v in zip(currencies,path[-1]) if v < 0)
    num, den = pair_frequencies(path[-60:])
    frequency = np.divide(num[-1], den[-1], out=np.full_like(num[-1], np.nan, dtype=float), where=den[-1]>0)
    if render_figures:
        fig, ax = plt.subplots(figsize=(8, 6.4), constrained_layout=True)
        image = ax.imshow(frequency, vmin=0, vmax=1, cmap=LinearSegmentedColormap.from_list('coassignment',['#f7f7f7','#f4a582','#b2182b']), interpolation='nearest')
        ax.set_xticks(range(len(currencies)), currencies, rotation=90, fontsize=7)
        ax.set_yticks(range(len(currencies)), currencies, fontsize=7)
        ax.set_title(f'R2K K=3, W=6: {dates[-60]} to {dates[-1]}')
        fig.colorbar(image, ax=ax, shrink=.8, label='Together / jointly eligible months')
        fig.savefig(folder/'evolution_coclustering.png', dpi=180);plt.close(fig)
        cmap = ListedColormap(['#176b80','#bd7b35','#67866b']);cmap.set_bad('#eeeeee')
        fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        image = ax.imshow(np.ma.masked_less(path.T, 0), aspect='auto', interpolation='nearest',
                          cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
        ax.set_yticks(range(len(currencies)), currencies, fontsize=7)
        ticks=list(range(0,len(dates),48));ax.set_xticks(ticks,[dates[t] for t in ticks],rotation=30,fontsize=8)
        ax.axvline(len(dates)-60-.5,color='black',lw=1)
        ax.set_title('Continuous R2K K=3, W=6: monthly labels (grey = unavailable)')
        fig.colorbar(image,ax=ax,ticks=[0,1,2],shrink=.7,label='Persistent cluster ID')
        fig.savefig(folder/'evolution_memberships.png',dpi=180);plt.close(fig)
    chosen=summary.loc[summary.k.eq(3)].set_index('method').reindex(NAMES).reset_index()
    chosen['method']=chosen.method.map(NAMES)
    chosen=chosen[['method','mean_ari_1m','mean_ari_12m','mean_ari_60m','mean_largest_share','mean_effective_groups']]
    chosen.columns=['K=3 method','ARI 1m','ARI 12m','ARI 60m','Largest share','Effective K']
    chosen.attrs['widths']=[.31,.12,.13,.13,.15,.16]
    pairs=[]
    for c,d in [('EUR','DKK'),('BRL','MXN'),('CNY','SGD'),('ARS','TRY')]:
        i,j=currencies.index(c),currencies.index(d)
        pairs.append([f'{c} / {d}',int(num[-1,i,j]),int(den[-1,i,j]),float(frequency[i,j])])
    current=summary.loc[summary.method.eq('r2k_w6') & summary.k.eq(3)].iloc[0]
    seeds=audit.loc[audit.kind.eq('initial_seed')].groupby(['n_init','seed']).ari_to_continuous.agg(['mean','min']).reset_index()
    seeds.columns=['First-fit starts','Seed','Mean ARI','Minimum ARI']
    restarts=audit.loc[audit.kind.eq('restart_60m')]
    return [
        ('h','Temporal review. What the TA asked us to establish'),
        ('callout','**Priority: observe how currency relationships evolve.** Run R2K continuously, then count pair co-clustering in genuine trailing five-year windows. The 60-month analysis window is separate from the 12-month feature lookback and the centroid memory W.'),
        ('p',f'Monthly R2K states begin {meta["initialization"]}; the eligible roster can change within the fixed 39-currency axis. The comparison below covers {meta["summary_start"]}-{meta["summary_end"]}: the first endpoint at which all currencies can support a complete 60-month return history, through the available endpoint. {dates[-1]} has {int((path[-1]>=0).sum())} eligible currencies; missing: {unavailable or "none"}. Lagged ARIs use jointly observed currencies; each monthly denominator is exported.'),
        ('table',chosen),
        ('p',f'At K=3, W=6, month-to-month ARI averages {current.mean_ari_1m:.3f}, but twelve-month agreement is {current.mean_ari_12m:.3f} and sixty-month agreement is {current.mean_ari_60m:.3f}. This distinguishes short-run continuity from persistent economic groupings. The largest R2K group averages {current.mean_largest_share:.1%} of eligible currencies. High pair co-clustering can reflect a broad group rather than specific affinity.'),
        ('page',None),('h','Temporal review. Latest five-year pair relationships'),
        ('sub',f'Actual pair frequencies, {dates[-60]}-{dates[-1]} (R2K K=3, W=6)'),
        ('equation',r'p_{ij,t}=\dfrac{\sum_{u=t-59}^{t} I_{ij,u}\,\mathbf{1}\{g_{i,u}=g_{j,u}\}}{\sum_{u=t-59}^{t} I_{ij,u}}'),
        ('table',_frame(pairs,['Currency pair','Together months','Eligible months','Fraction'],[.31,.23,.23,.23])),
        ('p','I marks joint eligibility, so absence never becomes a disagreement. These are observed historical fractions, not probabilities of future membership. Adjacent reporting windows overlap by 59 months; their smoothness is partly mechanical. Correlation alternatives refit genuinely trailing 60-month return histories, whereas R2K updates cross-sectional return/volatility states. Their temporal objectives differ.'),
        ('figure',('evolution_coclustering.png',f'Observed co-clustering frequency in the latest 60 calendar months. Missing denominators are not filled; most pairs have 60 eligible observations. Ineligible at the endpoint: {unavailable or "none"}.',260)),
        ('page',None),('h','Temporal review. Continuous paths and initialization'),
        ('figure',('evolution_memberships.png','Continuous chronological membership paths. The vertical line begins the latest five-year interval. Colors retain R2K identities; grey marks unavailability. A frozen 2019 singleton is one state, not an invariant grouping.',245)),
        ('sub','Initialization and restart sensitivity (K=3, W=6)'),
        ('table',seeds),
        ('p',f'Seed comparisons cover {meta["summary_start"]}-{meta["summary_end"]}; all six variants recover identical partitions over that span. At {len(restarts)} annual/final endpoints, restarting and replaying 60 origin-time cross sections gives mean ARI {restarts.ari_to_continuous.mean():.3f}, minimum {restarts.ari_to_continuous.min():.3f}, against the continuous path. This is empirical evidence for this feature stream, not universal initialization independence. Features retain their causal 12-month pre-window warm-up.'),
        ('page',None),('h','Implementation review. Algorithm 2 checked independently'),
        ('p','The supplied 6 March 2024 paper is the authority for the algorithm. An independent NumPy replay checks 135 post-initialization updates across K=2-4 and W=1,6,12. It computes snapping, Lloyd means and bank classification without calling the production seed/label helpers or the scikit-learn solver for subsequent updates. Centers, seeds, counts, current labels and rolling labels agree. Finite replay coverage is not a proof for all inputs.'),
        ('table',_frame([
            ['Algorithm 2, steps 6-9','Initial K-Means once; initial IDs ordered once. Ten k-means++ starts are a documented first-fit variant.'],
            ['Steps 12-15','Previous current-Lloyd counts determine small-cluster-first priority. Distinct current feature locations are claimed; priority sorting never renames IDs.'],
            ['Steps 16-18','Lloyd refinement uses supplied seeds and one start. Counts precede rolling classification. Fixed-point fits at the iteration limit are accepted.'],
            ['Step 19 / section 5.2','Nearest centroid over the latest W states, including the current state. No averaging, voting or centroid smoothing. W=6 and 12 follow the paper\'s suggested range; W=1 is its linear control.'],
            ['Project choices / limits','Per-month cross-sectional z-scores, tie ordering and duplicate-location exclusions are explicit conventions. Old centers retain their original normalized coordinates.'],
        ],['Paper rule','Verified interpretation'],[.28,.72])),
        ('p','**Paper claim requiring qualification:** unchanged input does not universally imply unchanged output. A fixed 12-point, two-feature example with K=3, W=1, seed 2 and one first-fit start changes three labels on the second identical update (ARI 0.262); inertia improves from 4.884 to 4.538. The snapping step can move the solution to another local optimum. The code intentionally follows the stated rule rather than adding an undocumented no-op shortcut.'),
        ('p','Paper trace: Algorithm 2 p.4; centroid memory section 5.2 p.5; smallest-first ordering section 5.3 pp.6-7. No author-maintained reference implementation was identified by the search. Tests also cover chronology, missingness, label permutation, iteration limits and pair denominators. Revised histories and retrospective sample masks remain point-in-time limitations. This descriptive review includes post-2019 data; it is not neural forecast validation.'),
        ('page',None),
    ], ['evolution_coclustering.png','evolution_memberships.png'] if render_figures else []


def _report_blocks(folder: Path, *, render_figures: bool = True) -> tuple[list[tuple[str, object]], list[str]]:
    scores = pd.read_csv(folder/"comparison.csv")
    stability = pd.read_csv(folder/"stability.csv")
    counts = pd.read_csv(folder/"sequence_counts.csv")
    coverage = pd.read_csv(folder/"feature_coverage.csv")
    groups = json.loads((folder/"groups.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    figures = _figures(folder, scores, stability, counts, groups) if render_figures else []
    recommended = scores.loc[scores.method.eq("adjusted_average") & scores.k.eq(2)].iloc[0]
    main_methods = [m for m in METHOD_NAMES if m in scores.method.unique() and not m.startswith(("levels", "r2k"))]
    comparison = scores.loc[scores.k.eq(2) & scores.method.isin(main_methods)].set_index("method").reindex(main_methods)
    comparison = comparison[["sizes", "subwindow_ari", "bootstrap_mean_ari", "silhouette"]].reset_index()
    comparison["method"] = comparison.method.map(METHOD_NAMES)
    comparison.columns = ["Approach", "Currencies per group", "Window ARI", "Block ARI", "Silhouette"]
    comparison.attrs["widths"] = [.30, .22, .16, .16, .16]
    count_rows = []
    for group in groups["adjusted_average:k2"]["clusters"]:
        for horizon in (1, 3, 12):
            selected = counts.loc[counts.method.eq("adjusted_average") & counts.k.eq(2)
                                  & counts.cluster.eq(group["cluster"]) & counts.view.eq("macro")
                                  & counts.horizon.eq(horizon)].set_index("split")
            count_rows.append([f"{'AB'[group['cluster']]} ({len(group['currencies'])} FX)", horizon,
                               *[int(selected.loc[s, "usable_origins"]) for s in ("train", "validation", "test")]])
    k_rows = []
    for row in scores.loc[scores.method.eq("adjusted_average")].itertuples():
        k_rows.append([row.k, " / ".join(map(str, json.loads(row.sizes))), row.bootstrap_mean_ari, row.n_singletons])
    avail = coverage.loc[coverage.split.eq("train")].set_index("feature").available_fraction
    shapes = manifest["array_shapes"]
    blocks = [
        ("title", "Currency groups for joint FX forecasting"),
        ("subtitle", "Temporal clustering, R2K audit and reproducibility | 10 November 2026"),
        ("p", "Prepared for technical business stakeholders and the deep learning TA. Data: the project's local Bank for International Settlements (BIS) snapshot, January 1994-August 2026. The TA-requested temporal review and paper audit lead this report. The HTML is the interactive findings brief. This reference preserves the mathematics, data contract, robustness protocol, complete method/K results and alternative memberships. Sections 1-4 explain the project outcome; Appendices A-J provide the technical detail."),
        ("h", "1. Project outcome and recommendation"),
        ("callout", "**Delivered: a reproducible data layer and candidate currency partitions.** The 25/14 default is stable but uneven in coherence. K=2-6 were compared; two or three natural currency types and a forecast-optimal K have not been established. The next stage must test how grouping changes neural information sharing."),
        ("p", "The hypothesis is that a cluster-specific network can share predictive structure across related currencies while limiting negative transfer (harmful sharing) and complexity. Fixed groups impose an inductive bias: direct cross-currency interactions are restricted to a subset. This may improve estimation with limited data, but may discard useful cross-group signals. Neural forecasting has not yet tested this tradeoff."),
        ("table", _frame([
            ["Saved default", "Two groups from median-adjusted return correlation with average linkage. Fit January 2011-December 2019; keep membership fixed afterward."],
            ["Why this candidate", f"This is a reproducible baseline, not a selected optimum; membership is relatively stable across training subwindows (ARI {recommended.subwindow_ari:.3f}) and block resamples (mean ARI {recommended.bootstrap_mean_ari:.3f})."],
            ["Keep as challengers", "K=1 pooled, K=3 adjusted-average and K=4 adjusted-Ward controls. The complete K=2-6 grid stays available; K=3 frees history for a subset."],
            ["Main constraint", "Only 79 and 167 usable training origins at the 12-month horizon in the two recommended groups. Model capacity must reflect these small, overlapping samples."],
        ], ["Decision", "Project implication"], [.23, .77])),
        ("sub", "What has been delivered"),
        ("p", "Implemented: fixed currency channels, local/global features with conservative publication lags, availability/carry masks, training-fitted transforms and a framework-neutral loader. Each origin supplies a 12-month sequence and C-by-H targets. Neural architecture, optimization and forecasting benchmarks are the next stage; this report evaluates the grouping and data layer."),
        ("sub", "Recommended next experiment"),
        ("p", "Prioritize pooled K=1 versus adjusted-average K=3 as the first grouped forecasting comparison; retain K=2 as the stability control. K=3 is a candidate because it separates a heterogeneous complement and releases history, not because its separation is stronger. Match origins, targets, features and model capacity, then test extra history separately. Use zero-change and own-history controls. Limit additional candidates, including Ward K=4, given the small validation sample. Retain grouping only if per-currency MAE/RMSE gains justify its complexity; economic value requires a decision-specific evaluation."),
        ("page", None),
        ("h", "2. What the proposed groups mean"),
        ("p", "The recommended solution is a broad international group and a Europe-heavy group. These labels describe the observed membership; they are not causal or official economic classifications. Each group mixes advanced and emerging economies. Currency codes and group IDs below match the exported dataset."),
    ]
    for group in groups["adjusted_average:k2"]["clusters"]:
        letter = "AB"[group["cluster"]]
        blocks += [
            ("sub", f"Group {letter}: {len(group['currencies'])} currencies | dataset ID {group['cluster']}"),
            ("p", f"**Representative: {group['representative']}.** " + ", ".join(group["currencies"]) + "."),
        ]
    blocks += [
        ("p", "The representative is the medoid: the member with the lowest mean within-group distance. It aids interpretation, not model routing or portfolio weighting. Group A does not imply homogeneous exposures or forecastability. Fixed channel identities make the grouping usable with recurrent, convolutional or attention-based temporal architectures."),
        ("figure", ("group_correlations.png", "Figure 1. Original same-month FX return correlations, 2011-2019. The lines separate A (25 currencies) from B (14). Red indicates positive correlation; blue indicates negative correlation.", 360)),
        ("p", f"Overall within r={recommended.within_correlation:.3f}, between r={recommended.between_correlation:.3f}. Group A has within r=0.448 and B=0.750. Stable membership does not imply equal coherence; Appendices E-G examine the theory, K choices and factor references."),
        ("page", None),
        ("h", "3. Why this approach was selected"),
        ("p", "All main methods use the same 39 currencies and 108 monthly returns from 2011-2019. Raw correlation measures simultaneous FX/USD movements. Adjustment subtracts the monthly cross-currency median to reduce the influence of a shared movement. Neural inputs and targets retain the original FX changes."),
        ("table", comparison),
        ("p", "**Reading the table.** ARI (adjusted Rand index) equals 1 for identical membership, around 0 for chance-level agreement, and can be negative. Window ARI compares 2011-2015 with 2015-2019 (overlapping in 2015). Block ARI averages 50 twelve-month block resamples against the full fit. Neither is forecast accuracy or a probability."),
        ("p", "Adjusted average linkage supplies stable modeling units compared with the 38/1 raw-correlation split. Ward remains a challenger. Dynamic time warping (DTW) permits two-month phase alignment; its weaker stability is less compelling for synchronous channels. Size and stability motivate the default, not a demonstrated forecast optimum."),
        ("table", _frame(k_rows, ["Number of groups", "Adjusted-average sizes", "Mean block ARI", "Singletons"], [.22, .38, .22, .18])),
        ("p", "The full K=2-6 grid remains available. Singletons at K=4-6 are specific to adjusted average linkage. Adjusted Ward K=4 gives 12/11/6/10 without singletons, but group-wise diagnostics reveal a weak six-currency block (Appendix F). K=2 is a baseline, not a statistical optimum."),
        ("sub", "What the paper contributes"),
        ("p", "The Robust Rolling K-Means (R2K) paper motivates stable labels for evolving clusters. Our reference implementation groups recent mean-change/volatility states, rather than shared currency movements. Its K=2 snapshots yield a 1/38 split, using the shorter complete state history of July 2011-December 2019. Dynamic regime labels must not replace currency identities in neural channels."),
        ("p", "Level correlation yields 37/2 and window ARI -0.022, despite silhouette 0.551; R2K silhouette is 0.820. Silhouettes describe their own representation and cannot rank these methods as forecasting models. The default silhouette of 0.175 also signals limited separation, not sharply distinct economic regimes."),
        ("page", None),
        ("h", "4. Usable data and sample-size limits"),
        ("p", "Default eligibility requires complete mandatory histories across selected currencies for twelve months. Launches, peg exclusions and warm-up constrain coverage. Output dimension does not increase the number of independent sequences: adjacent windows overlap and currencies share shocks. This is a small-sample estimation problem despite the long raw panel."),
        ("table", _frame([
            ["Training", "Through December 2019", "Fit preprocessing and neural parameters. Group selection uses only 2011-2019."],
            ["Validation", "January 2020-December 2022", "Choose architecture and settings without refitting group membership."],
            ["Test", "January 2023-August 2026", "Evaluate the chosen settings. Earlier project inspection limits confirmatory claims."],
        ], ["Split", "Origin dates", "Purpose"], [.17, .35, .48])),
        ("sub", "Usable origins for the default macro-plus-credit inputs"),
        ("table", _frame(count_rows, ["Group", "Horizon (months)", "Train", "Validation", "Test"], [.23, .23, .18, .18, .18])),
        ("p", "Counts require complete mandatory inputs and at least one observed target. Policy and inflation are optional, with missingness flags. Requiring all currency targets leaves training/validation counts unchanged but reduces B's test counts by one. Both definitions are exported."),
        ("p", "Complete macro windows begin June 2012 for A and February 2005 for B; CNY's exclusions and FX warm-up bind A. At the twelve-month horizon, training origins stop in December 2018 so targets mature by December 2019."),
        ("sub", "A practical benefit of retaining three groups"),
        ("table", _frame([
            ["A split: 14 currencies, representative ZAR", "January 2004", 180],
            ["A split: 11 currencies, representative CNY", "June 2012", 79],
            ["B unchanged: 14 currencies, representative EUR", "February 2005", 167],
        ], ["Three-group alternative", "First usable input origin", "12-month training origins"], [.48, .27, .25])),
        ("p", "Splitting A releases history for 14 currencies, at lower membership stability (block ARI 0.673 versus 0.784). Match training and evaluation origins to isolate grouping effects; test extra history separately. B has 217 versus 167 twelve-month training origins with FX-only inputs, so macro comparisons also require matched samples."),
        ("page", None),
        ("h", "Appendix A. Mathematics and target interpretation"),
        ("p", "Notation: i and j are currencies, t is the origin month, h is the horizon in months, and q is local currency units per US dollar. Positive FX changes mean depreciation. Changes are in percent log units (100 times a natural-log ratio)."),
        ("sub", "A1. Monthly FX change"),
        ("equation", r"r_{i,t}=100\ln\!\left(\dfrac{q_{i,t}}{q_{i,t-1}}\right)"),
        ("p", "If q rises from 10 to 11, r = 9.53 percent log units (a 10% rise in local units per USD). Missing or excluded months break the history."),
        ("sub", "A2. Remove the shared movement, then standardize"),
        ("equation", r"m_t=\operatorname{median}_{j=1,\ldots,39}r_{j,t},\qquad e_{i,t}=r_{i,t}-m_t"),
        ("equation", r"z_{i,t}=\dfrac{e_{i,t}-\mu_i}{s_i},\qquad d_{ij}=\sqrt{2\left(1-\rho_{ij}\right)}"),
        ("p", "For each currency, mu and s are the residual mean and population standard deviation over 2011-2019; rho is the correlation of standardized histories. Average linkage merges groups with the smallest mean pairwise distance, until K groups remain. Ward instead minimizes the increase in within-group squared Euclidean dispersion of standardized histories. Median adjustment is a statistical proxy; it does not identify a causal USD factor or imply independence."),
        ("sub", "A3. Forecast the original cumulative FX movement"),
        ("equation", r"y_{i,t}^{(h)}=100\ln\!\left(\dfrac{q_{i,t+h}}{q_{i,t}}\right)=\sum_{u=1}^{h}r_{i,t+u},\qquad h\in\{1,3,12\}"),
        ("p", "Inputs cover t-11 through t. Targets require eligible, observed FX at every month from t to t+h and maturity within the split cutoff. Otherwise the mask is false and the export is zero. Targets are original FX movements, not adjusted residuals."),
        ("sub", "A4. Ignore unavailable targets in the loss"),
        ("equation", r"\mathcal{L}=\dfrac{\sum_{n,i,h}M_{n,i,h}\left(\widehat{y}_{n,i,h}-y_{n,i,h}\right)^2}{\sum_{n,i,h}M_{n,i,h}}"),
        ("p", "n indexes origins; M marks observed, mature targets. Require a positive denominator. Unscaled MSE can emphasize volatile currencies and longer horizons; specify loss weights and report currency/horizon-balanced metrics. Missing-target zeros are placeholders, not zero-change forecasts."),
        ("page", None),
        ("h", "Appendix B. Inputs, timing and missingness"),
        ("p", "The canonical panel contains seven local features per currency and three shared global features. The table gives reference timing at a month-end origin t and the share of training-period source-panel cells available before imputation. Local percentages pool all 39 currencies over January 1994-December 2019; global percentages count months. These percentages are not the share of complete joint sequences."),
        ("table", _frame([
            ["Local: FX changes (1/3/12 months)", "Observed through t; no filling", f"{avail['fx_change_1m']:.1%} / {avail['fx_change_3m']:.1%} / {avail['fx_change_12m']:.1%}"],
            ["Local: 12-month volatility", "12 consecutive monthly returns through t", f"{avail['volatility_12m']:.1%}"],
            ["Local: REER gap", "Reference month t-2; preceding 60-month baseline", f"{avail['reer_gap']:.1%}"],
            ["Local: policy-rate differential", "Domestic minus US; reference month t-1", f"{avail['rate_diff']:.1%}"],
            ["Local: inflation differential", "Domestic minus US annual log inflation; t-2", f"{avail['inflation_diff']:.1%}"],
            ["Global: broad USD NEER change", "One-month percent log change; t-1", f"{avail['usd_neer_change']:.1%}"],
            ["Global: outside-US USD credit stock", "Natural log of stock in USD millions; quarter of t minus 2", f"{avail['usd_credit_log_stock']:.1%}"],
            ["Global: USD credit annual growth", "100 times the four-quarter log change; same quarter lookup", f"{avail['usd_credit_yoy']:.1%}"],
        ], ["Input", "Definition / information timing", "Train availability"], [.31, .49, .20])),
        ("sub", "Rolling features and units"),
        ("equation", r"v_{i,t}=\sqrt{12}\;\operatorname{sd}_{\rm sample}\!\left(r_{i,t-11},\ldots,r_{i,t}\right)"),
        ("equation", r"g_{i,t}=100\left(\ln R_{i,t-2}-\frac{1}{60}\sum_{u=1}^{60}\ln R_{i,t-2-u}\right)"),
        ("p", "R is the real effective exchange-rate index (REER); NEER is the nominal effective exchange-rate index. The baseline excludes the current reference month. Volatility is annualized from monthly observations, not daily realized volatility. Policy differences are in percentage points; inflation differences use annual percent log inflation. Global USD variables may help prediction but do not discriminate currencies within a cross section."),
        ("sub", "Availability rules that matter for modeling"),
        ("p", "Policy and derived annual inflation permit one causal carry month, recorded in carry flags. Longer gaps remain missing. FX, CPI levels, REER, NEER and credit are not interpolated or carried. Credit uses the exact quarter two quarters before the origin's quarter and is held within that origin quarter. The flagged US October 2025 inflation fallback enters at the December 2025 origin."),
        ("p", "Policy sources are structurally absent for BAM, DZD, SGD and TWD; CPI sources are absent for BAM and TWD. Japan has additional policy gaps. Optional policy/inflation gaps use pooled training-feature medians and preserve false availability masks. Feature means and scales are fitted only through December 2019, shared across groups, and use all available training monthly cells. Mandatory gaps use masked zero placeholders; targets are unscaled. Negative policy rates and genuine extreme FX moves are retained."),
        ("page", None),
        ("h", "Appendix C. Neural interface and experimental design"),
        ("p", f"The shared bundle stores {shapes['X_local'][0]} windows before eligibility filtering. For cluster c and origin n, a proposed network maps local history X, availability A, carry flags B and global history G to joint forecasts. Currency order is fixed; C is group size and H the number of horizons."),
        ("equation", r"\widehat{\mathbf{Y}}_{n,c}=f_{\theta_c}(\mathbf{X}_{n,c},\mathbf{A}_{n,c},\mathbf{B}_{n,c},\mathbf{G}_n,\mathbf{A}^{G}_n)\in\mathbb{R}^{C\times H}"),
        ("table", _frame([
            ["Local inputs", "N x 12 x C x 7", "FX and local macro features; matching availability and carry masks"],
            ["Global inputs", "N x 12 x 3", "Shared USD/credit features; matching availability mask"],
            ["Targets", "N x C x H", "Original cumulative FX changes; matching target mask"],
            ["Metadata", "Fixed ordered identifiers", "Origins, input months, target endpoints, currency/feature orders, preprocessing and hashes"],
        ], ["Object", "Default macro-view shape", "Meaning"], [.19, .29, .52])),
        ("code", "from pathlib import Path\nfrom fx_forecasting.data.neural_dataset import load_cluster_dataset\nbatch = load_cluster_dataset(\n    Path(\"reports/clustering/handoff\"), k=2, cluster=0,\n    split=\"train\", horizon=3, feature_view=\"macro\",\n)\n# X_local (88,12,25,7); X_global (88,12,3); Y (88,25,1)\n# Inputs: include availability/carry flags. Loss: use Y_mask."),
        ("p", "Flattening local values, availability and carry flags plus global values/availability at each month gives 21C+6 inputs: 531 for A and 300 for B. Preserve the twelve-month time axis; concatenation is an architecture choice, not extra observations. Input flags must enter the network; masking target loss alone does not handle missing inputs."),
        ("sub", "Evaluation decisions for the next stage"),
        ("p", "**Ablations.** Own-history versus joint inputs tests cross-currency information sharing. K=1, K=2, K=3 and K=4 Ward test the grouping restriction. FX-only versus macro and credit/no-credit tests added information. Match origins, targets and features for each contrast; use comparable capacity or report parameter budgets. Evaluate extra training history separately."),
        ("p", "**Optimization and assessment.** Select architecture, regularization and early stopping on 2020-2022 validation, preserving temporal ordering. Report MAE/RMSE in original percent log units by currency/horizon and equal-currency aggregates across seeds. Fit any target scaling or loss weights on training only. Overlapping targets require dependence-aware uncertainty."),
        ("p", "**Loader controls.** method=\"all\", k=1 selects the all-currency control; k=3 selects three adjusted-average groups. feature_view=\"fx\" selects FX-only inputs; with_credit=False removes credit. Partial histories need explicit mask-aware handling. Economic value requires a decision rule, exposures and costs."),
        ("page", None),
        ("h", "Appendix D. Robustness, review and reproducibility"),
        ("p", "Returns are computed before resampling, so block joins do not manufacture FX returns. Methods use the same resampled months. DTW uses three seeds/three starts for the primary fit and one seed/three starts per resample. Stability is a sensitivity check, not a confidence interval."),
        ("table", _frame([
            ["Methodology / data review", "Separate co-movement, lag-tolerant shapes and moment states; retain macro inputs and publish true origin counts."],
            ["Goal / handoff review", "Deliver fixed channels, vector targets and masks. The earlier scalar Ridge screen did not test joint neural forecasting."],
            ["Correctness / scope review", "Fix iteration-limit convergence, plot scaling and stale output hashes. Keep one primary workflow, the paper reference and an optional historical audit."],
        ], ["Independent review", "Change made"], [.29, .71])),
        ("p", "Checks cover future-data invariance, fixed channels, training-only transforms, target maturity, FX eligibility, macro timing/carries, structural gaps, export round-trips and the three bug regressions. Notebook 05 runs from both supported directories. Training-fitted frozen groups may label training history; dynamic states need origin-time information."),
        ("p", "Limits: revised histories, retrospective regime exclusions, dependent observations and previously inspected evaluation data. This is not a real-time vintage study. Neural forecast gains, investment returns and universal stability are unproven."),
        ("code", 'python -m pip install -e ".[clustering,reports,dev]"\npython scripts/build_clustering_handoff.py\npython -m pytest -q'),
        ("p", "Notebook 05 is the walkthrough. NPZ arrays, JSON metadata, Parquet panels and result tables sit beside this report; manifest.json records versions and hashes. Markdown and PDF share this full reference. The HTML selects the findings, memberships and diagnostics from the same saved results, with robustness and methodology collapsed."),
        ("sub", "Sources"),
        ("p", "[1] Holmes, Hirsa, Klinkert and Malhotra, Robust Rolling K-Means, supplied version dated 6 March 2024: [paper](https://ssrn.com/abstract=4729393). [2] Project sample rules: config/research_sample.json and docs/data/sample_and_treatment.md. [3] Implementation definitions: [SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), [tslearn DTW](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html), and [scikit-learn clustering evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation). Numerical findings in this report are from the saved project comparison and eligibility tables, not results reported in the R2K paper."),
    ]
    temporal, temporal_figures = _temporal_blocks(folder, render_figures=render_figures)
    blocks[3:3] = temporal
    figures.extend(temporal_figures)
    blocks.extend(_theory_blocks(folder))
    blocks.extend(_reference_blocks(folder, scores, groups))
    return blocks, figures


def _reference_blocks(folder: Path, scores: pd.DataFrame, groups: dict) -> list:
    """Keep diagnostic definitions and the complete numerical record in PDF/MD."""
    quality = pd.read_csv(folder/"group_quality.csv").merge(
        pd.read_csv(folder/"group_recovery.csv"), on=["method", "k", "cluster"], how="left")
    q = quality.loc[(quality.method.eq("adjusted_average") & quality.k.isin([2, 3])) |
                    (quality.method.eq("adjusted_ward") & quality.k.eq(4) & quality.cluster.isin([1, 2]))].copy()
    q["Group"] = [f"{'Avg' if r.method == 'adjusted_average' else 'Ward'} K{r.k} / {r.representative}"
                  for r in q.itertuples()]
    q["Gap"] = q.within_raw_correlation - q.outside_raw_correlation
    metrics = q[["Group", "within_raw_correlation", "outside_raw_correlation", "Gap", "mean_silhouette", "mean_jaccard"]].copy()
    metrics.columns = ["Adjusted-return group", "Within r", "Outside r", "Gap", "Silhouette", "Jaccard"]
    metrics.attrs["widths"] = [.29, .14, .14, .13, .15, .15]
    blocks = [
        ("page", None), ("h", "Appendix H. Reading the diagnostic scores"),
        ("sub", "H1. Correlation gaps measure descriptive separation"),
        ("equation", r"\Delta_g=\overline{\rho}_{\rm within\;g}-\overline{\rho}_{\rm g\;to\;others}"),
        ("p", "Within correlation averages distinct unordered pairs; outside correlation averages member-to-nonmember pairs. Self-correlations are excluded; a singleton has no within-pair statistic. The HTML shows original and median-adjusted gaps separately. Positive gaps describe relative co-movement, not predictive information."),
        ("sub", "H2. Silhouette describes a currency's boundary"),
        ("equation", r"s_i=\dfrac{b_i-a_i}{\max(a_i,b_i)},\qquad -1\leq s_i\leq 1"),
        ("p", "a is mean distance to the currency's own group; b is its smallest mean distance to another group. Negative values flag poorly assigned members; singletons receive zero. Silhouettes average currencies, whereas correlation gaps average pairs. Read each silhouette in its own representation/distance space."),
        ("sub", "H3. Recovery measures repeatability"),
        ("equation", r"J_g^{(b)}=\max_h\dfrac{|G_g\cap G_h^{(b)}|}{|G_g\cup G_h^{(b)}|}"),
        ("p", "Match each reference group to its best-overlap group in each block resample, then average Jaccard over 50 replications. Matching ignores label numbers and need not be one-to-one. ARI compares the whole partition with chance-adjusted pair agreement: 1 is identical, around 0 is chance-level; negatives are possible. Neither is a probability or forecast score."),
        ("table", metrics),
        ("p", "Selected groups above: correlations use original nominal returns; silhouettes use adjusted-return distances. K=3's ZAR group gains history despite weak separation. Ward's SGD group shows why avoiding singletons is insufficient."),
    ]
    for title, methods in [
        ("I1. Synchronous return methods", ["returns_average", "returns_ward", "adjusted_average", "adjusted_ward"]),
        ("I2. Alignment, levels and rolling-state references", ["dtw", "levels_average", "r2k_w1", "r2k_w6"]),
    ]:
        rows = []
        for method in methods:
            for r in scores.loc[scores.method.eq(method)].sort_values("k").itertuples():
                rows.append([METHOD_NAMES[method], r.k, " / ".join(map(str, json.loads(r.sizes))),
                             r.silhouette, r.subwindow_ari, r.bootstrap_mean_ari, r.n_singletons])
        blocks += [
            ("page", None), ("h", "Appendix I. Complete method/K comparison"), ("sub", title),
            ("p", "Same 39 currencies; fit cutoff December 2019. Window ARI compares overlapping 2011-2015 and 2015-2019 fits. Block ARI averages 50 twelve-month resamples. A dash means not evaluated. No row is an out-of-sample forecast result."),
            ("table", _frame(rows, ["Method", "K", "Group sizes", "Silhouette", "Window ARI", "Block ARI", "Singletons"],
                              [.24, .06, .22, .12, .12, .13, .11])),
            ("p", "Original returns include common FX/USD movements; median-adjusted returns remove each month's shared median. Average linkage and Ward optimize different merge criteria. Silhouettes must be read in the selected representation and distance space; higher K can fragment a stable partition without improving its forecasting usefulness." if title.startswith("I1") else
             "DTW allows phase alignment; levels retain trends; R2K compares annual-change/volatility states. Section 5.3 of the supplied paper preserves one-point outlier clusters through seed selection. Robustness concerns initialization and labels, rather than trimming the squared-distance loss. ARS's singleton state does not establish bad data. R2K states begin July 2011; bootstrap estimates for these frozen references are unavailable."),
        ]
    blocks += [("page", None), ("h", "Appendix J. Alternative memberships and files"),
               ("p", "The K=2 default membership is in Section 2. Alternatives below retain fixed currency identities and use the same 2011-2019 fit. Group letters and IDs apply within each method/K; a medoid is an interpretation aid, not a forecasting weight.")]
    for method, k in [("adjusted_average", 3), ("adjusted_ward", 4)]:
        blocks.append(("sub", f"{METHOD_NAMES[method]}, K={k}"))
        for g in groups[f"{method}:k{k}"]["clusters"]:
            blocks.append(("p", f"**Group {'ABCDEF'[g['cluster']]} / ID {g['cluster']} / medoid {g['representative']} ({len(g['currencies'])} currencies):** "+", ".join(g["currencies"])+"."))
    blocks += [
        ("sub", "Reproduce and audit"),
        ("p", "Run the command in Appendix D from the repository root; Notebook 05 verifies hashes before reuse. All paths below are relative to reports/clustering/handoff/. Group loaders select channels from the common bundle, rather than creating separate ragged datasets."),
        ("table", _frame([
            ["dataset.npz + dataset_metadata.json", "Numeric tensors, masks, ordered identifiers and preprocessing state"],
            ["groups.json + memberships.csv", "Group channel indices, currency memberships and representatives"],
            ["monthly_local.parquet + monthly_global.parquet", "Auditable local and shared-global monthly feature panels"],
            ["comparison.csv + stability.csv", "Full method/K scores and individual resample agreements"],
            ["group_quality.csv + group_recovery.csv", "Group coherence and best-match Jaccard"],
            ["sequence_counts.csv + manifest.json", "Eligibility by split/view/horizon, plus source/config/code/output hashes"],
        ], ["File(s)", "Purpose"], [.44, .56])),
    ]
    return blocks


def _theory_blocks(folder: Path) -> list:
    """Separate population structure, partition stability and predictive sharing."""
    quality = pd.read_csv(folder/"group_quality.csv").merge(
        pd.read_csv(folder/"group_recovery.csv"), on=["method", "k", "cluster"], how="left")
    reference = pd.read_csv(folder/"factor_reference.csv")
    spectrum = pd.read_csv(folder/"factor_spectrum.csv")
    metadata = json.loads((folder/"theory_metadata.json").read_text())
    raw = spectrum.loc[spectrum.representation.eq("raw_standardized")].set_index("component")
    a = quality.loc[quality.method.eq("adjusted_average") & quality.k.eq(2)].set_index("cluster")
    chosen = quality.loc[(quality.method.eq("adjusted_average") & quality.k.eq(2)) |
                         (quality.method.eq("adjusted_average") & quality.k.eq(3) & quality.cluster.eq(0)) |
                         (quality.method.eq("adjusted_ward") & quality.k.eq(4) & quality.cluster.isin([1, 2]))].copy()
    chosen["Group"] = [f"{'Average' if row.method == 'adjusted_average' else 'Ward'} K={row.k} / {row.representative}"
                       for row in chosen.itertuples()]
    chosen = chosen[["Group", "n_currencies", "within_raw_correlation", "outside_raw_correlation", "mean_silhouette", "mean_jaccard"]]
    chosen.columns = ["Adjusted representation / medoid", "Size", "Within r", "Outside r", "Silhouette", "Jaccard"]
    chosen.attrs["widths"] = [.31, .08, .14, .14, .16, .17]
    ref = reference.loc[reference.method.eq("adjusted_average") & reference.k.eq(2)].copy()
    ref["Reference 5%-95%"] = [f"{row.reference_p05:.3f} to {row.reference_p95:.3f}" for row in ref.itertuples()]
    ref = ref[["factor_rank", "observed_silhouette", "reference_median", "Reference 5%-95%"]]
    ref.columns = ["Retained PCA factors", "Observed silhouette", "Reference median", "Reference 5%-95%"]
    return [
        ("page", None),
        ("h", "Appendix E. What clustering can establish"),
        ("sub", "E1. Geometry describes movements"),
        ("p", "The objects being clustered are 39 currencies, represented by 108 monthly returns each. For standardized currency histories z, correlation distance has a Euclidean interpretation:"),
        ("equation", r"\dfrac{1}{T}\|\mathbf{z}_i-\mathbf{z}_j\|_2^2=2(1-\rho_{ij})=d_{ij}^2"),
        ("p", "Average linkage merges sets with the smallest mean pairwise distance; Ward minimizes added within-group squared dispersion. These objectives organize response histories, not conditional forecasting functions. Negatively correlated currencies can still carry useful opposite-signed information; a positive-correlation grouping can place such predictors in different groups."),
        ("p", "Subtracting a shared cross-sectional statistic changes dependence. With independent equal-variance currencies, subtracting the monthly arithmetic mean gives the residual correlation below. The project uses a median, whose covariance differs; the example shows why adjustment does not guarantee independent shocks."),
        ("equation", r"\operatorname{Corr}(r_i-\bar r,r_j-\bar r)=-\dfrac{1}{N-1},\qquad i\ne j"),
        ("sub", "E2. Covariance and predictability are different objects"),
        ("equation", r"\mathbf{r}_{t+1}=A\mathbf{r}_t+\boldsymbol{\varepsilon}_{t+1}"),
        ("p", "Even if A=0, correlated innovations create strong contemporaneous clusters without conditional-mean predictability. Conversely, let r1,t be an independent shock and r2,t+1 equal a times that shock plus independent noise. Their same-month covariance can be zero while r1,t predicts r2,t+1. Clustering same-month returns therefore neither establishes nor rules out useful cross-currency neural inputs."),
        ("sub", "E3. A hard partition restricts the forecasting model"),
        ("equation", r"A_{ij}=0\quad\mathrm{if}\quad g(i)\ne g(j)"),
        ("p", "In the linear analogy, separate within-group models impose the restriction above. A nonlinear model using only its group's local inputs imposes a similar information restriction, although common global inputs remain available. This can reduce harmful sharing but discard cross-group signal. More groups also replicate encoders if models are trained separately; smaller groups do not automatically mean fewer total parameters."),
        ("sub", "E4. Select K against the actual objective"),
        ("equation", r"(\widehat K,\widehat\lambda)=\arg\min_{K,\lambda}\widehat R_{\rm validation}(f_{K,\lambda})"),
        ("p", "Here lambda includes architecture, regularization and sharing choices. Use matched origins and comparable capacity. K=1 is the all-currency control; its silhouette is undefined and its membership stability is trivially perfect. The K=2-6 grid is an exploratory range, not a theoretical upper bound. K=2 is an operational baseline, not an inferred true or forecast-optimal cluster count."),
        ("page", None),
        ("h", "Appendix F. What the deeper diagnostics show"),
        ("table", chosen),
        ("p", f"Within/outside r are original-return correlations; silhouettes use the adjusted representation. The 25-currency group's raw gap is only {a.loc[0, 'within_raw_correlation']-a.loc[0, 'outside_raw_correlation']:.3f}; the EUR group's within correlation is {a.loc[1, 'within_raw_correlation']:.3f}. A stable complement to a coherent group can itself be heterogeneous. Ward K=4 avoids singletons, yet its six-currency SGD group has five negative silhouettes and weak recovery. Avoiding singletons is not sufficient evidence for a good partition."),
        ("p", "Jaccard measures overlap with each reference group's best matching group after a twelve-month time-block resample. It is label-invariant and group-specific; it is not a probability of correct economic membership. The pairwise coassignment export records how often each pair stays together across these same resamples."),
        ("sub", "F1. Continuous factors can reproduce cluster-like structure"),
        ("p", f"PC1 explains {raw.loc[1, 'variance_share']:.1%} of standardized return variation; the first three PCs explain {raw.loc[3, 'cumulative_share']:.1%}. These are descriptive statistical factors, not identified USD or tradable risk factors. Median subtraction removes the same monthly quantity from every currency; it does not estimate heterogeneous factor loadings or guarantee independent residuals."),
        ("equation", r"Z=F_qB_q^\top+E_q,\qquad Z^*=F_qB_q^\top+\mathrm{Shift}(E_q)"),
        ("table", ref),
        ("p", f"Z is the standardized training-return matrix. Each reference retains its rank-q PCA reconstruction, independently shifts residual columns, and transforms back to return units before reapplying grouping. The {metadata['reference_reps']} replications per rank are diagnostic only, never neural training data. Fitted factor paths/loadings and residual circular structure are retained; estimation uncertainty and nonstationarity are not modeled."),
        ("p", "The observed K=2 separation exceeds the one-factor reference range but lies inside the three-factor range. Richer continuous exposures can therefore reproduce separation of this magnitude. The three-factor reference can itself retain regional/group-like loading structure. This is consistent with a factor-based explanation, not proof that discrete clusters are absent. These are conditional reference ranges, not confidence intervals, a classical gap statistic or significance tests."),
        ("page", None),
        ("h", "Appendix G. Revised interpretation and research choices"),
        ("table", _frame([
            ["K=1 / all 39", "Essential pooled control. Preserves all cross-currency inputs; needs regularization and explicit channel identities."],
            ["Adjusted average K=2", "Stable reproducible baseline, with one coherent Europe-heavy block and a heterogeneous 25-currency complement."],
            ["Adjusted average K=3", "Frees longer histories for 14 currencies; their mean silhouette is only 0.079 and three members have negative values."],
            ["Adjusted Ward K=4", "Sizes 12/11/6/10 retain multivariate groups. A six-currency group is weak; examine its recovery and boundary members."],
            ["K=5-6", "Remain available. Singletons affect average linkage here, not every method; adjusted Ward K=6 has no singleton. Model fragmentation and stability still require evaluation."],
        ], ["Candidate", "Interpretation for the neural experiment"], [.29, .71])),
        ("p", "The present evidence supports common structure and a particularly coherent Europe-heavy block more clearly than a universal partition into two or three natural currency types. Larger K is not prohibited, and no fixed K has demonstrated superior forecasts. Retain the complete K=2-6 comparison rather than presenting the default as a selected optimum."),
        ("sub", "G1. Test information access separately from output grouping"),
        ("p", "A shared trunk with currency or group-specific heads can retain all-currency inputs while organizing outputs by group. Separate within-group networks remove cross-group local information. Compare these restrictions on matched origins and target cells; otherwise any difference may reflect information access, parameter duplication or historical coverage rather than better groups. The default twelve-month validation sample has only 24 matured origins. Predeclare a limited candidate/capacity budget to control validation overfitting. These are teammate experiments, not implemented neural results."),
        ("sub", "G2. Soft sharing is another hypothesis"),
        ("equation", r"\mathcal{L}_{\rm total}=\mathcal{L}_{\rm forecast}+\dfrac{\lambda}{2}\sum_{i,j}w_{ij}\|\theta_i-\theta_j\|_2^2"),
        ("p", "For comparable linear output-head parameters theta on a shared representation, nonnegative training-fitted affinities w can encourage similar tasks to share without disconnecting groups. This is a regularization hypothesis, motivated by clustered multi-task learning, not evidence that return correlations identify optimal neural sharing. Tune lambda on validation; evaluate a zero-penalty control. Arbitrary deep-network weights need not be comparable because of parameter symmetries."),
        ("sub", "Theory references"),
        ("p", "Mantegna (1999), [correlation-based financial taxonomy](https://arxiv.org/abs/cond-mat/9802256); Hennig (2007), [cluster-wise stability](https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf), adapted here to time-block rather than currency resampling; Tibshirani, Walther and Hastie (2001), [reference-distribution motivation for cluster counts](https://doi.org/10.1111/1467-9868.00293); Jacob, Bach and Vert (2008), [clustered multi-task learning](https://arxiv.org/abs/0809.2085). The latter treats linear task relationships, not a proof of neural forecasting gains in this FX sample."),
    ]


def _inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" color="#176b80"><u>\1</u></a>', text)


def _math_flowable(expression: str):
    """Draw MathText glyphs as embedded-font vectors, keeping equations sharp."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Flowable

    parsed = MathTextParser("path").parse("$" + expression + "$", dpi=72,
                                         prop=FontProperties(size=12.5, math_fontfamily="dejavusans"))
    fonts = {}
    for glyph in parsed.glyphs:
        path = str(glyph[0].fname)
        if path not in fonts:
            name = "Math_" + Path(path).stem
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, path))
            fonts[path] = name

    class Equation(Flowable):
        def __init__(self):
            super().__init__()
            self.width = float(parsed.width)
            self.height = float(parsed.height) + 16
            self.spaceAfter = 7
            self.keepWithNext = True

        def wrap(self, available_width, available_height):
            if self.width > available_width:
                raise ValueError(f"Equation exceeds report width: {expression}")
            return self.width, self.height

        def draw(self):
            canvas = self.canv
            canvas.setFillColorRGB(.10, .22, .29)
            baseline = float(parsed.depth) + 8
            for glyph in parsed.glyphs:
                font, size, code = glyph[:3]
                x, y = glyph[-2:]  # Supports both five- and six-field MathText glyphs.
                canvas.setFont(fonts[str(font.fname)], float(size))
                canvas.drawString(float(x), float(y)+baseline, chr(code))
            for x, y, width, height in parsed.rects:
                canvas.rect(float(x), float(y)+baseline, float(width), float(height), fill=1, stroke=0)
    return Equation()


def _render_pdf(blocks: list, folder: Path):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
        Preformatted, KeepTogether, PageBreak,
    )
    from PIL import Image as PILImage

    doc = SimpleDocTemplate(str(folder/"clustering_report.pdf"), pagesize=(612, 792),
                            leftMargin=48, rightMargin=48, topMargin=43, bottomMargin=42,
                            title="Currency groups for joint FX forecasting", author="FX research project")
    styles = getSampleStyleSheet()
    for name, settings in {
        "Body": dict(fontName="Helvetica", fontSize=10, leading=14.1, spaceAfter=9,
                     allowWidows=0, allowOrphans=0, textColor=colors.HexColor("#263d49")),
        "Heading": dict(fontName="Helvetica-Bold", fontSize=15, leading=19,
                        spaceAfter=12, keepWithNext=True, textColor=colors.HexColor("#176b80")),
        "Subheading": dict(fontName="Helvetica-Bold", fontSize=10.7, leading=14,
                           spaceBefore=6, spaceAfter=6, keepWithNext=True, textColor=colors.HexColor("#203d4b")),
        "ReportTitle": dict(fontName="Helvetica-Bold", fontSize=26, leading=30,
                            spaceAfter=12, textColor=colors.HexColor("#153f51")),
        "Subtitle": dict(fontName="Helvetica", fontSize=9.5, leading=13,
                         spaceAfter=15, textColor=colors.HexColor("#5c707a")),
        "Cell": dict(fontName="Helvetica", fontSize=8.6, leading=11.5,
                     textColor=colors.HexColor("#263d49")),
        "HeaderCell": dict(fontName="Helvetica-Bold", fontSize=8.6, leading=11.5,
                           textColor=colors.HexColor("#153f51")),
        "Caption": dict(fontName="Helvetica-Oblique", fontSize=8.5, leading=11.5,
                        textColor=colors.HexColor("#546773"), spaceAfter=10),
        "CodeSmall": dict(fontName="Courier", fontSize=8.0, leading=11.3,
                          backColor=colors.HexColor("#f0f4f6"), borderPadding=7, spaceBefore=3, spaceAfter=12),
        "Callout": dict(fontName="Helvetica", fontSize=11, leading=15.5, spaceAfter=14,
                        backColor=colors.HexColor("#eaf3f5"), borderPadding=10,
                        textColor=colors.HexColor("#153f51")),
    }.items():
        styles.add(ParagraphStyle(name="FX"+name, **settings))
    story = []
    for kind, value in blocks:
        if kind in ("title", "subtitle", "h", "sub", "p", "callout"):
            name = {"title": "ReportTitle", "subtitle": "Subtitle", "h": "Heading",
                    "sub": "Subheading", "p": "Body", "callout": "Callout"}[kind]
            story.append(Paragraph(_inline(str(value)), styles["FX"+name]))
        elif kind == "page":
            story.append(PageBreak())
        elif kind == "equation":
            story.append(_math_flowable(str(value)))
        elif kind == "code":
            story.append(KeepTogether([Preformatted(str(value), styles["FXCodeSmall"], maxLineLength=90)]))
        elif kind == "table":
            frame = value
            rows = [[Paragraph(_inline(str(column)), styles["FXHeaderCell"]) for column in frame.columns]]
            rows += [[Paragraph(_inline(_cell(x)), styles["FXCell"]) for x in row]
                     for row in frame.itertuples(index=False, name=None)]
            widths = frame.attrs.get("widths", [1/len(frame.columns)]*len(frame.columns))
            table = Table(rows, colWidths=[doc.width*x for x in widths], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfecef")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8f9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#9fb8c2")),
            ]))
            story.extend([table, Spacer(1, 10)])
        elif kind == "figure":
            name, caption, max_height = value
            with PILImage.open(folder/name) as picture:
                width, height = picture.size
            display_height = min(max_height, doc.width*height/width)
            story.append(KeepTogether([
                Image(str(folder/name), width=display_height*width/height, height=display_height),
                Paragraph(_inline(caption), styles["FXCaption"]),
            ]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d3dfe3")); canvas.line(48, 33, 564, 33)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#637781"))
        canvas.drawString(48, 21, "Joint FX forecasting | methodology and reproducibility reference")
        canvas.drawRightString(564, 21, str(document.page)); canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def _write_teammate_brief(folder: Path) -> list[str]:
    """A short, generated selection guide; canonical memberships stay in groups.json."""
    groups = json.loads((folder/'groups.json').read_text())
    counts = pd.read_csv(folder/'sequence_counts.csv')
    candidates = [
        ('Pooled control', 'all', 1, 'Essential benchmark with all cross-currency inputs.'),
        ('First grouped experiment', 'adjusted_average', 3, '14/14/11 co-movement groups; longer histories for two groups, but limited separation.'),
        ('Paper-based comparator', 'r2k_w6', 3, 'December 2019 return/volatility states. One singleton and a large complement; not a forecasting winner.'),
        ('Optional stability control', 'adjusted_average', 2, 'Merge adjusted K=3 groups 0 and 2. Fewer models; a heterogeneous 25-currency group.'),
    ]
    lines = ['# Currency groupings teammates can try',
             'Start with pooled K=1 versus adjusted-return K=3, then add the R2K K=3, W=6 comparator. K=2 is an optional stability control. These are candidate experiments; no neural forecasting gains have been demonstrated.',
             '**Freeze date: December 2019.** Co-movement groups use January 2011-December 2019; R2K uses its eligible July 2011-December 2019 state history. The continuous R2K run has the same December 2019 partition (ARI 1), although numeric IDs can differ. Use the IDs in this guide and groups.json with the loader. Do not use a 2026 snapshot to define groups for 2020-2026 evaluation.',
             '## Candidate memberships and available training sequences',
             'Counts below use 12 input months, FX + macro + credit inputs, complete mandatory inputs and at least one observed target. Optional macro missingness remains flagged. Columns count monthly origins, not independent samples; longer horizons overlap. Each target must mature by December 2019.']
    for title, method, k, reason in candidates:
        definition = groups[f'{method}:k{k}']
        lines += [f'### {title}: `{method}`, K={k}', reason]
        rows = []
        for group in definition['clusters']:
            c = counts.loc[counts.method.eq(method) & counts.k.eq(k) & counts.cluster.eq(group['cluster']) &
                           counts.view.eq('macro') & counts.split.eq('train')].set_index('horizon')
            rows.append([group['cluster'], len(group['currencies']), *[int(c.loc[h, 'usable_origins']) for h in (1,3,12)]])
        lines.append(_table(_frame(rows, ['Group ID', 'Currencies', 'Train N: 1m', 'Train N: 3m', 'Train N: 12m'])))
        for group in definition['clusters']:
            lines.append(f"**Group {group['cluster']} ({len(group['currencies'])} currencies):** "+', '.join(group['currencies'])+'.')
    lines += ['## Load a group',
              'Run from the repository root after installing the project. The existing builder supplies these files; no separate datasets are created per group.',
              '```bash\npython -m pip install -e ".[clustering,reports,dev]"\n# Only if exports are absent:\npython scripts/build_clustering_handoff.py\n```',
              '```python\nfrom pathlib import Path\nfrom fx_forecasting.data.neural_dataset import load_cluster_dataset\n\nroot = Path("reports/clustering/handoff")\nbatch = load_cluster_dataset(\n    root, method="adjusted_average", k=3, cluster=0,\n    split="train", horizon=3, feature_view="macro",\n)\nprint(batch.currencies)\nprint(batch.X_local.shape)   # (189, 12, 14, 7)\nprint(batch.X_global.shape)  # (189, 12, 3)\nprint(batch.Y.shape)         # (189, 14, 1)\n# Include X_local_mask, X_local_carried and X_global_mask as model inputs.\n# Apply Y_mask in every loss and metric.\n```',
              'Change method/k to `all`/1, `r2k_w6`/3 or `adjusted_average`/2; select every listed cluster ID and fit/evaluate all 39 currencies. ARS is univariate in the R2K snapshot: do not silently drop it. A representative/medoid is not a target weight.',
              'Use split="validation" (2020-2022) to select settings and split="test" (2023-August 2026) for descriptive final assessment. Previously inspected post-2019 data mean this is not an untouched confirmatory holdout. horizon=None selects all three horizons; feature_view="fx" uses four local FX features and no global features.',
              '## Make the comparison interpretable',
              'For the first comparison, intersect origins separately within each split across the pooled batch and every selected group; verify the same currency/horizon target cells are scored. Compare capacity/parameter budgets and use target masks. With the default macro-plus-credit inputs, the common training counts are 90/88/79 for 1/3/12-month horizons. Evaluate the extra available group history in a separate experiment, so history length is not confused with the value of grouping.',
              'Report per-currency and equal-currency aggregate errors in original FX units across seeds. Keep feature views and horizon choices matched. The 12-month validation horizon has only 24 matured origins, so limit tuning. The 2019 R2K snapshot is a fixed grouping; causal evolving R2K labels are a separate potential context input on the same fixed currency axis.',
              'Canonical files: [memberships](groups.json), [all sequence counts](sequence_counts.csv), [dataset metadata](dataset_metadata.json), [source/config/code/output hashes](manifest.json), and [temporal explorer](clustering_report.html#evolution).']
    (folder/'teammate_quickstart.md').write_text('\n\n'.join(lines)+'\n')
    return ['teammate_quickstart.md']


def write_clustering_report(output_dir: Path) -> list[str]:
    """Render the full Markdown/PDF reference and short HTML from saved results."""
    folder = Path(output_dir)
    from fx_forecasting.evaluation.clustering_theory import write_structure_diagnostics
    settings = json.loads((folder/"settings.json").read_text())
    diagnostics = write_structure_diagnostics(folder, bootstrap_reps=settings["bootstrap_reps"], seed=settings["seed"])
    from fx_forecasting.evaluation.clustering_evolution import write_evolution_diagnostics
    temporal_outputs = write_evolution_diagnostics(folder, k_values=tuple(settings["k_values"]), seed=settings["seed"])
    blocks, figures = _report_blocks(folder)
    markdown = []
    for kind, value in blocks:
        if kind == "title": markdown.append("# " + str(value))
        elif kind == "subtitle": markdown.append("*" + str(value) + "*")
        elif kind == "h": markdown.append("## " + str(value))
        elif kind == "sub": markdown.append("### " + str(value))
        elif kind == "p": markdown.append(str(value))
        elif kind == "callout": markdown.append("> " + str(value))
        elif kind == "table": markdown.append(_table(value))
        elif kind == "equation": markdown.append("$$\n" + str(value) + "\n$$")
        elif kind == "page": markdown.append("---")
        elif kind == "code":
            language = "bash" if str(value).startswith("python -m pip") else "python"
            markdown.append("```"+language+"\n"+str(value)+"\n```")
        elif kind == "figure": markdown.append(f"![{value[1]}]({value[0]})")
    (folder/"clustering_report.md").write_text("\n\n".join(markdown)+"\n")
    _render_pdf(blocks, folder)
    from fx_forecasting.evaluation.clustering_html import write_clustering_html
    html_outputs = write_clustering_html(folder)
    teammate_outputs = _write_teammate_brief(folder)
    return [*diagnostics, *temporal_outputs, *figures, "clustering_report.md", "clustering_report.pdf", *html_outputs, *teammate_outputs]
