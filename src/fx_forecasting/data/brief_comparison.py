"""Closing discussion section: executive requirements versus local evidence."""

from __future__ import annotations

import html

import pandas as pd


def render_brief_comparison(overview, overlap, choices):
    """Keep source-universe counts distinct from selected-panel coverage."""
    roster=choices["currencies"]
    areas={row["area"] for row in roster}
    matched=overlap.loc[overlap.has_reer]
    chosen=overlap.loc[overlap.area.isin(areas)]
    if len(chosen)!=len(roster):
        raise ValueError("Selected currency representatives do not match the reviewed FX snapshot")
    primary=len(roster)
    policy_n=int(chosen.has_policy.sum())
    cpi_n=int(chosen.has_cpi.sum())
    advanced=sum(row["group"]=="advanced" for row in roster)
    emde=primary-advanced
    absent_policy=', '.join(chosen.loc[~chosen.has_policy,"currency"])
    absent_cpi=', '.join(chosen.loc[~chosen.has_cpi,"currency"])
    fx_areas=len(overlap)
    fx_currencies=overlap.currency.nunique()
    matched_codes=matched.currency.nunique()
    nonusd_codes=matched.loc[matched.currency.ne("USD"),"currency"].nunique()
    flow=[(fx_areas,"FX reference areas","Full end-of-month export"),
          (fx_currencies,"currency codes","Different counting unit"),
          (len(matched),"REER-matched areas","Cross-source overlap"),
          (matched_codes,"currency codes","Shared codes collapsed"),
          (nonusd_codes,"non-USD currencies","Exploratory EDA sample"),
          (primary,"selected currencies","After AED / HKD / SAR exclusions")]
    flow_html='<div class="scope-flow">'+''.join(
        f'<div class="flow-step"><strong>{n}</strong><span>{html.escape(label)}</span><small>{html.escape(reason)}</small></div>'
        for n,label,reason in flow)+'</div>'
    comparison=[
        ["USD exchange rates","39 target currencies",f"{fx_areas} reference areas / {fx_currencies} currency codes; {nonusd_codes} in EDA; {primary} selected for the primary design",
         "Bulk coverage is broader than a study sample. Matching 39 does not prove that our list is the executives’ intended list.","Validate the roster, not just its size."],
        ["Effective exchange rates","63 economies",f"{int(overview.loc['reer','series'])} real and {int(overview.loc['neer','series'])} nominal broad series",
         "The selected export count is one economy larger. The brief’s source vintage and selection are unspecified.","Record the snapshot and broad-basket definition; preserve valid source series."],
        ["Policy rates","34 of the 39 currencies",f"{int(overview.loc['policy','series'])} source histories; {policy_n} matches within our chosen {primary}",
         "Source histories, currency matches and usable monthly observations are different measures.","Reconcile 35 versus 34; do not discard a valid series to force the advertised count."],
        ["Consumer prices","62 economies",f"{int(overview.loc['cpi','series'])} source economies; {cpi_n} matches within our chosen {primary}",
         "One more source economy is available, while some selected currencies still lack CPI.","Document overlap and missingness separately from the total export count."],
        ["USD credit","Outside-US aggregate + 12 emerging economies; quarterly from 2000",f"{int(overview.loc['usd_credit','series'])} series, through 2026Q1; expected 2026Q2 absent",
         "The series count agrees. The latest available quarter is earlier than the monthly FX endpoint.","Q1 supplies July/August 2026 origins under the chosen two-quarter lookup; country credit coverage is not the FX roster."],
        ["Study window","January 1994–August 2026", "392 monthly periods; individual histories, launches and peg exclusions differ",
         "A shared calendar does not imply 39 complete eligible histories at every month or horizon.","Use an unbalanced panel and explicit masks; show usable counts after applying them."],
    ]
    def table(frame):
        return '<div class="table-scroll">'+frame.to_html(index=False,border=0,classes="data-table",escape=True)+'</div>'
    comparison_table=table(pd.DataFrame(comparison,columns=["Item","Executive brief","What we found","Why the distinction matters","Initial action"]))
    questions=[
        ("Which 39 currencies did the executives intend?",
         f"The brief does not list them or establish whether 39 is counted before or after peg exclusions. Our post-exclusion list has {policy_n} policy matches, versus the brief’s 34.",
         "Use the current roster as version 1; confirm the post-exclusion interpretation and request the original list if it exists before locking the modeling sample.","Selected locally · correspondence unconfirmed"),
        ("What counts as a peg, and how should transition months enter?",
         "The brief names dollar pegs, CNY and MYR periods but supplies no boundaries. De jure and de facto CNY dates differ.",
         "Use documented intervals and whole-month transition exclusions; retain EUR-linked dollar movements and test the alternative CNY cutoff.","Rules selected · implementation pending"),
        ("How do we retain countries with absent or incomplete macro inputs?",
         f"Policy sources are absent for {absent_policy}; CPI for {absent_cpi}. Japan has long policy gaps and US CPI is missing October 2025.",
         "Preserve nulls; permit flagged one-month causal carries; use training-window replacements and missing indicators only for optional model inputs. Compare a complete-case sensitivity.","Treatment selected · sensitivity required"),
        ("Are we testing dollar-credit stock or dollar-credit growth?",
         "The executive question concerns the stock. Report 2’s correlations use lagged annual log growth, which answers a different exploratory question.",
         "Keep the lagged outside-US stock in the primary planned test. Discuss its transformation and scaling; use growth as a separate sensitivity, not a replacement for the stock test.","Requirement clear · feature form pending"),
        ("What timing assumptions make the comparison interpretable?",
         "Reference dates are not release dates. Latest revised exports cannot demonstrate real-time vintage availability.",
         "Apply the documented reference-period lags, disclose the pseudo-out-of-sample scope and keep forecast origins identical across compared models.","Convention selected · not real-time vintages"),
    ]
    question_html='<div class="discussion-questions">'+''.join(
        '<article><div class="discussion-status">'+html.escape(status)+'</div><h3>'+html.escape(question)+
        '</h3><p><b>Why discuss it:</b> '+html.escape(why)+'</p><p><b>Current approach:</b> '+html.escape(how)+'</p></article>'
        for question,why,how,status in questions)+'</div>'
    currency_map={row["area"]:row["currency"] for row in roster}
    intervals=pd.DataFrame([{"Currency":currency_map[row["area"]],"Exclude from":row["start"],
        "Through (inclusive)":row["end"],"Reason":row["reason"]} for row in choices["excluded_month_intervals"]])
    roster_table=pd.DataFrame([{"Area":row["area"],"Currency":row["currency"],"Economy":row["economy"],
        "Group":"Advanced" if row["group"]=="advanced" else "EMDE", "First eligible month":row["first_eligible_month"]}
        for row in roster])
    return f'''
<section id="brief-comparison" class="brief-discussion">
<div class="eyebrow">08 / EXECUTIVE BRIEF VS OBSERVED DATA</div>
<h2>What differs, what needs discussion, and where we stand.</h2>
<p class="discussion-lede"><b>Yes, the download contains more than 39 currencies.</b> The bulk export, the exploratory comparison and the chosen research panel have different scopes. The initial task is to make those scopes explicit and validate the selection—not to force every source count to match the brief.</p>
<div class="discussion-meta">Brief supplied 13 October 2026 · source snapshot 8 October 2026 · choices recorded 13 October 2026</div>
<h3>How the broad export becomes a candidate modeling sample</h3>
{flow_html}
<p class="caption">The 192-to-147 step changes the counting unit; it is not a deletion of 45 currencies. The {len(matched)}-to-{matched_codes} step collapses shared currency codes, including national EUR histories. We interpret {primary} as the count after permanent USD-peg exclusions; this interpretation and the exact roster need reconciliation with the executives. Historical peg intervals remove monthly observations rather than entire currencies.</p>
<h3>The count differences and their implications</h3>
{comparison_table}
<p class="caption">The +1 EER/CPI discrepancies may reflect different release vintages or selections; the original snapshot was not supplied, so their precise cause is unverified. A matching source does not imply complete or timely monthly inputs. All displayed counts come from the saved quality tables and chosen configuration.</p>
<h3>The questions to settle at the first design stage</h3>
{question_html}
<details><summary>Expand the selected roster and peg boundaries</summary>
<p>The current reporting split is {advanced} advanced and {emde} IMF emerging market and developing economies. This is a static April 2026 reporting classification; EMDE is broader than an investment-index definition. It does not certify historical group membership.</p>
{table(roster_table)}{table(intervals)}
<p>Currency starts, target intervals crossing excluded periods, short carries and long gaps are specified in the decision record. These are design choices; they have not been applied to a production panel.</p></details>
<div class="project-position"><h3>Where the project is now</h3>
<div class="position-grid"><div><span class="position-label">Completed</span><p>Download, ingestion, source-quality checks and descriptive EDA.</p></div>
<div><span class="position-label">Selected and documented</span><p>The {primary}-currency design, groups, peg/launch masks and missing-data/timing conventions.</p></div>
<div><span class="position-label">Not implemented yet</span><p>The production panel and features; required benchmark and neural-network comparisons. EDA correlations and historical-mean diagnostics do not answer whether the specified networks beat random walk.</p></div></div>
<p><b>Next decision gate:</b> reconcile the intended roster and policy count, confirm the stock-credit specification, then apply the chosen rules consistently when production processing is authorised. This section addresses the initial choices; later model selection and significance testing remain outside the current step.</p></div>
<div class="links"><a class="download" href="quality/project_brief.md">Executive instructions as supplied ↗</a><a class="download" href="quality/sample_and_treatment.md">Decision record and sources ↗</a><a class="download" href="quality/research_sample.json" download>Selected roster and rules ↗ JSON</a><a class="download" href="data_eda.html#credit-section">Credit-growth exploration in Report 2 ↗</a></div>
</section>'''
