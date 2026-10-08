"""Basic descriptive BIS data review. No imputation, inference, or models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from fx_forecasting.config import PROJECT_ROOT, PROCESSED_DIR, RAW_DIR
from fx_forecasting.data.loaders import BISExport, load_eda_datasets

START, END = "1994-01", "2026-08"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
UNITS = {
    "fx": "Local currency units per USD; end of month",
    "reer": "Broad real effective exchange-rate index, 2020 = 100",
    "neer": "Broad nominal effective exchange-rate index, 2020 = 100",
    "policy": "Policy rate, percent per annum; end of month",
    "cpi": "Consumer price index, 2010 = 100",
    "usd_credit": "Total USD credit to non-banks, millions of USD; quarterly stock",
}


def sample_values(key: str, export: BISExport) -> pd.DataFrame:
    if key == "usd_credit":
        # August is inside Q3. Expect only completed quarters, through Q2.
        last_quarter = pd.Period(END, "M").asfreq("Q") - 1
        periods = pd.period_range("2000Q1", last_quarter, freq="Q")
    else:
        periods = pd.period_range(START, END, freq="M")
    return export.values.reindex(periods)


def missing_profile(key: str, export: BISExport, values: pd.DataFrame) -> pd.DataFrame:
    records = []
    for series in values:
        observed = values[series].notna().to_numpy()
        locations = np.flatnonzero(observed)
        missing = ~observed
        if len(locations):
            first, last = int(locations[0]), int(locations[-1])
            leading, trailing = first, len(values) - last - 1
            internal = int(missing[first:last + 1].sum())
            internal_missing = missing[first:last + 1]
            longest = 0
            run = 0
            for absent in internal_missing:
                run = run + 1 if absent else 0
                longest = max(longest, run)
        else:
            first = last = None
            leading = trailing = internal = longest = 0
        meta = export.metadata.loc[series]
        area_column = "BORROWERS_CTY" if key == "usd_credit" else "REF_AREA"
        name_column = "Borrowers' country" if key == "usd_credit" else "Reference area"
        full_valid = export.values[series].dropna()
        records.append({
            "dataset": key, "series": series,
            "area": meta[area_column], "economy": meta[name_column],
            "currency": meta.get("CURRENCY", ""),
            "expected_periods": len(values), "observed": int(observed.sum()),
            "missing": int(missing.sum()), "missing_pct": 100 * missing.mean(),
            "leading_missing": leading, "internal_missing": internal,
            "trailing_missing": trailing,
            "all_missing": len(values) if not len(locations) else 0,
            "longest_internal_gap": longest,
            "first_in_sample": str(values.index[first]) if first is not None else "",
            "last_in_sample": str(values.index[last]) if last is not None else "",
            "source_first": str(full_valid.index.min()) if not full_valid.empty else "",
            "source_last": str(full_valid.index.max()) if not full_valid.empty else "",
            "min": values[series].min(), "median": values[series].median(),
            "max": values[series].max(), "units": UNITS[key],
        })
    return pd.DataFrame(records)


def area_values(export: BISExport, values: pd.DataFrame) -> pd.DataFrame:
    column = "BORROWERS_CTY" if export.name == "WS_GLI" else "REF_AREA"
    result = values.rename(columns=export.metadata[column].to_dict())
    if not result.columns.is_unique:
        raise ValueError("Multiple selected series for the same reference area")
    return result


def gap_details(key: str, export: BISExport, values: pd.DataFrame) -> pd.DataFrame:
    records = []
    name_column = "Borrowers' country" if key == "usd_credit" else "Reference area"
    for series in values:
        missing = values[series].isna().to_numpy()
        observed = np.flatnonzero(~missing)
        starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
        ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
        for start, end in zip(starts, ends):
            kind = "all_missing" if not len(observed) else (
                "leading" if end < observed[0] else "trailing" if start > observed[-1] else "internal"
            )
            records.append({"dataset": key, "series": series,
                            "economy": export.metadata.loc[series, name_column],
                            "gap_type": kind, "start": str(values.index[start]),
                            "end": str(values.index[end]), "periods": int(end - start + 1)})
    return pd.DataFrame(records, columns=["dataset", "series", "economy", "gap_type", "start", "end", "periods"])


def markdown_table(frame: pd.DataFrame) -> str:
    """Small Markdown tables without an optional tabulate dependency."""
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def make_figures(datasets: dict, samples: dict, profiles: pd.DataFrame) -> list[Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    paths = []

    def save(fig, name):
        path = FIGURE_DIR / f"eda_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4))
    for key in ("fx", "reer", "neer", "policy", "cpi"):
        values = samples[key]
        ax.plot(values.index.to_timestamp(), values.notna().sum(axis=1), label=key)
    ax.set(title="Available monthly series in the study window", ylabel="Series with a value", xlabel="Month")
    ax.legend(ncol=5, loc="lower left")
    save(fig, "coverage")

    fig, ax = plt.subplots(figsize=(10, 4))
    summary = profiles.groupby("dataset")[["leading_missing", "internal_missing", "trailing_missing", "all_missing"]].sum()
    denominators = profiles.groupby("dataset")["expected_periods"].sum()
    (summary.div(denominators, axis=0) * 100).plot.barh(stacked=True, ax=ax, color=["#8db7ce", "#d55e00", "#f2c56c", "#757575"])
    ax.set(title="Missing observations by type", xlabel="Percent of expected series-period cells", ylabel="")
    ax.legend(["Before first value", "Within observed history", "After last value", "Entire sample absent"], fontsize=9)
    save(fig, "missing")

    # Display only FX reference areas represented in the broad REER dataset.
    # This is an exploratory comparison group, not the unspecified 39 currencies.
    reer_areas = set(datasets["reer"].metadata["REF_AREA"])
    fx_meta = datasets["fx"].metadata
    ids = fx_meta.loc[fx_meta["REF_AREA"].isin(reer_areas)].sort_values("Reference area").index
    matrix = samples["fx"][ids].isna().T
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(["white", "#b64c12"]), vmin=0, vmax=1)
    ticks = np.arange(0, len(matrix.columns), 48)
    ax.set_xticks(ticks, [str(matrix.columns[i]) for i in ticks], rotation=45)
    ax.set_yticks(np.arange(len(ids)), fx_meta.loc[ids, "Reference area"], fontsize=8)
    ax.set(title="FX availability for reference areas with broad REER data\nWhite = observed; orange = missing", xlabel="Month")
    ax.grid(False)
    save(fig, "fx_missing_map")

    fx = area_values(datasets["fx"], samples["fx"])
    examples = [c for c in ("BR", "MX", "ZA", "TR", "CN", "JP", "GB") if c in fx]
    recent = fx.loc["2010-01":, examples]
    base = recent.loc[pd.Period("2010-01", "M")]
    fig, ax = plt.subplots(figsize=(11, 5))
    indexed = recent.div(base.where(base > 0), axis=1) * 100
    indexed.index = indexed.index.to_timestamp()
    indexed.plot(ax=ax)
    ax.set(yscale="log", title="Illustrative exchange-rate paths, January 2010 = 100",
           ylabel="Local currency per USD, indexed (log scale)", xlabel="Month")
    save(fig, "fx_paths")

    changes = fx[examples].where(fx[examples] > 0).pct_change(fill_method=None) * 100
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, area in zip(axes.flat, examples):
        changes[area].dropna().plot.hist(bins=35, ax=ax, color="#377da3")
        ax.set(title=area, xlabel="Monthly change (%)", ylabel="Months")
    for ax in list(axes.flat)[len(examples):]:
        ax.set_visible(False)
    fig.suptitle("Monthly FX changes for illustrative currencies, 1994–August 2026\nPositive = depreciation against USD; no missing values filled", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fx_changes")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for key, title, ax in zip(("reer", "policy", "cpi"),
                              ("Broad real effective rates (2020 = 100)", "Policy rates from 2000 (% per annum)", "CPI (2010 = 100, log scale)"), axes.flat):
        data = area_values(datasets[key], samples[key])
        areas = [a for a in ("US", "BR", "MX", "ZA", "CN") if a in data]
        data = data[areas].copy()
        if key == "policy":
            # Preserve early observations in tables; the 1994 Brazil rates would
            # flatten all the other curves on this illustrative linear chart.
            data = data.loc["2000-01":]
        data.index = data.index.to_timestamp()
        data.plot(ax=ax, linewidth=1)
        ax.set(title=title, xlabel="")
        if key == "cpi":
            ax.set_yscale("log")
    dollar = area_values(datasets["neer"], samples["neer"])["US"]
    axes[1, 1].plot(dollar.index.to_timestamp(), dollar)
    axes[1, 1].set(title="BIS broad nominal USD index (2020 = 100)", xlabel="")
    fig.tight_layout()
    save(fig, "macro")

    credit = area_values(datasets["usd_credit"], samples["usd_credit"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(credit.index.to_timestamp(how="end"), credit["3P"] / 1_000_000)
    axes[0].set(title="USD credit to non-banks outside the US", ylabel="Trillions of USD")
    for area in ("BR", "CN", "MX", "IN", "TR"):
        axes[1].plot(credit.index.to_timestamp(how="end"), credit[area] / 1_000, label=area)
    axes[1].set(title="Illustrative country USD credit stocks", ylabel="Billions of USD")
    axes[1].legend(ncol=3)
    fig.tight_layout()
    save(fig, "credit")
    return paths


def run_eda() -> dict:
    for directory in (PROCESSED_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    datasets = load_eda_datasets()
    samples = {key: sample_values(key, data) for key, data in datasets.items()}
    profiles = pd.concat([missing_profile(key, data, samples[key]) for key, data in datasets.items()], ignore_index=True)
    gaps = pd.concat([gap_details(key, data, samples[key]) for key, data in datasets.items()], ignore_index=True)
    audit = pd.DataFrame([data.audit for data in {d.name: d for d in datasets.values()}.values()])
    overview = profiles.groupby("dataset", sort=False).agg(
        series=("series", "size"), expected=("expected_periods", "sum"),
        observed=("observed", "sum"), missing=("missing", "sum"),
        leading=("leading_missing", "sum"), internal=("internal_missing", "sum"),
        trailing=("trailing_missing", "sum"),
        first=("first_in_sample", "min"), last=("last_in_sample", "max"),
    ).reset_index()
    overview["missing_pct"] = overview["missing"] / overview["expected"] * 100
    overview["complete_series"] = overview["dataset"].map(profiles.loc[profiles["missing"].eq(0)].groupby("dataset").size()).fillna(0).astype(int)
    fx_metadata = datasets["fx"].metadata
    overlap = fx_metadata[["REF_AREA", "Reference area", "CURRENCY"]].rename(columns={"REF_AREA": "area", "Reference area": "economy", "CURRENCY": "currency"}).copy()
    for key in ("reer", "neer", "policy", "cpi", "usd_credit"):
        column = "BORROWERS_CTY" if key == "usd_credit" else "REF_AREA"
        overlap[f"has_{key}"] = overlap["area"].isin(datasets[key].metadata[column])
    # Same currency code can describe separate countries and historical backcasts.
    # Record them rather than treating them as duplicated observations.
    shared = overlap.groupby("currency").filter(lambda x: len(x) > 1)
    fx = area_values(datasets["fx"], samples["fx"])
    changes = fx.where(fx > 0).pct_change(fill_method=None) * 100
    moves = changes.stack().rename("monthly_change_pct").reset_index()
    moves.columns = ["period", "area", "monthly_change_pct"]
    moves = moves.merge(overlap[["area", "economy", "currency"]], on="area", validate="many_to_one")
    moves["abs_change_pct"] = moves["monthly_change_pct"].abs()
    largest_moves = moves.nlargest(20, "abs_change_pct")
    nonpositive = pd.DataFrame([
        {"dataset": key, "nonpositive_values": int((value <= 0).to_numpy().sum())}
        for key, value in samples.items() if key != "policy"
    ])
    for key, value in samples.items():
        value.to_csv(PROCESSED_DIR / f"eda_{key}.csv", index_label="period")
        datasets[key].metadata.to_csv(PROCESSED_DIR / f"eda_{key}_metadata.csv")
    tables = {"overview": overview, "missing_by_series": profiles, "missing_gap_dates": gaps, "raw_audit": audit,
              "currency_overlap": overlap.reset_index(), "shared_currency_codes": shared.reset_index(),
              "largest_fx_changes": largest_moves, "nonpositive_values": nonpositive}
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / f"{name}.csv", index=False)
    figures = make_figures(datasets, samples, profiles)
    manifest = json.loads((RAW_DIR / "download_manifest.json").read_text()) if (RAW_DIR / "download_manifest.json").exists() else {}
    summary = {"monthly_start": START, "monthly_end": END,
               "sources": manifest, "overview": overview.to_dict("records"),
               "raw_audit": audit.to_dict("records"),
               "selection": {key: data.metadata.index.tolist() for key, data in datasets.items()}}
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = write_report(tables, datasets, figures, manifest)
    return {"datasets": datasets, "samples": samples, "tables": tables, "figures": figures, "report": report}


def write_report(tables: dict, datasets: dict, figures: list[Path], manifest: dict) -> Path:
    profiles, overview = tables["missing_by_series"], tables["overview"]
    dates = sorted({m["downloaded_at_utc"][:10] for m in manifest.values()})
    report = ["# Initial BIS data review", "",
              f"Source snapshot downloaded (UTC): {', '.join(dates) or 'unrecorded'}. Monthly study window: January 1994–August 2026 (392 months).",
              "Quarterly credit window: 2000Q1–2026Q2 (106 completed quarters by August 2026). Source files remain unchanged. No filling, interpolation, outlier removal, peg exclusions, models, or statistical tests were applied.", "",
              "## Main findings", "",
              "The five requested datasets are downloaded. The bulk exports contain a larger universe than the project sample. The exact 39-currency roster is not supplied, so this review covers all end-of-month FX series and reports reference-area overlaps without defining the final modeling sample.", "",
              markdown_table(overview[["dataset", "series", "observed", "missing", "missing_pct", "internal", "complete_series", "last"]]), "",
              "Both broad effective-rate datasets have complete monthly coverage. Policy rates have the largest missing share; most missing cells occur before the first value or after the last value, including national euro-area histories that end at euro adoption. Japan has 116 internally missing policy-rate months, which need individual review.", "",
              "CPI has one internal missing month, in the US series; most other missing CPI cells reflect late starts or unavailable recent data. All 13 USD-credit series are complete from 2000Q1 through 2026Q1. Reindexing to the expected completed-quarter calendar exposes the absent 2026Q2 observation in each series (13 trailing missing cells).", "",
              "Missing percentages use the full expected calendar for every selected series. Leading gaps, gaps between the first and last observed value, and trailing gaps are reported separately in `missing_by_series.csv`. A trailing gap may be a publication delay or a discontinued national series; it is not automatically an error.", "",
              "## Duplicate and value checks", "",
              markdown_table(tables["raw_audit"][["dataset", "raw_series_rows", "raw_exact_duplicate_rows", "raw_duplicate_series_rows", "invalid_numeric_tokens"]]), "",
              "Each source row is one series. Duplicate series identifiers and duplicate complete source rows were checked across every raw frequency. Unique series identifiers and unique time columns also establish unique series-period keys in each selected matrix. Blank cells and the BIS `NaN` marker are recognized as missing. Other nonnumeric tokens are counted separately.", "",
              f"The FX export has {datasets['fx'].metadata['REF_AREA'].nunique()} reference areas and {datasets['fx'].metadata['CURRENCY'].nunique()} currency codes. Shared currency codes (including EUR) describe different reference areas and historical backcasts; they are not duplicate series-period records.", "",
              markdown_table(tables["nonpositive_values"]), "",
              "Policy rates are excluded from the nonpositive-value check because zero and negative rates can be valid observations.", "",
              "## Coverage and source definitions", "",
              "- FX: monthly `COLLECTION=E`, quoted as local currency units per USD. An increase means depreciation against USD.",
              "- REER and NEER: monthly broad baskets (`EER_BASKET=B`); separate real (`R`) and nominal (`N`) indices, 2020 = 100. The USD index is `M:N:B:US`, the BIS broad nominal index rather than ICE DXY.",
              "- Policy rates: native BIS monthly series, measured at month end. National euro-area histories can end at euro adoption. No ECB substitution is applied in this EDA.",
              "- CPI: monthly index levels (`UNIT_MEASURE=628`), 2010 = 100. The alternative year-over-year inflation series are excluded to avoid mixing units.",
              "- USD credit: quarterly `USD`, non-banks `N`, all lenders `A`, position `I`, total credit `B`, unit `USD`. Values are in millions of USD. The outside-US aggregate is `Q:USD:3P:N:A:I:B:USD`. Country series cover Argentina, Brazil, Chile, China, India, Indonesia, Malaysia, Mexico, Russia, Saudi Arabia, Türkiye and South Africa.", "",
              f"Current exports contain {len(datasets['reer'].metadata)} broad EER economies and {len(datasets['cpi'].metadata)} CPI economies, compared with the brief's 63 and 62. There are {len(datasets['policy'].metadata)} monthly policy-rate histories; coverage of 34 of 39 currencies cannot be confirmed without that roster.", "",
              "### Series with the most missing observations", "",
              markdown_table(profiles.nlargest(15, "missing_pct")[["dataset", "economy", "missing_pct", "leading_missing", "internal_missing", "trailing_missing", "first_in_sample", "last_in_sample"]]), "",
              "### Internal gaps to inspect", "",
              markdown_table(profiles.loc[profiles["internal_missing"].gt(0)].nlargest(15, "internal_missing")[["dataset", "economy", "internal_missing", "longest_internal_gap"]]), "",
              "### Selected internal gap dates", "",
              markdown_table(tables["missing_gap_dates"].loc[
                  tables["missing_gap_dates"]["gap_type"].eq("internal") &
                  tables["missing_gap_dates"]["dataset"].isin(["policy", "cpi"])
              ][["dataset", "economy", "start", "end", "periods"]]), "",
              "## Initial visual EDA", "",
              "The charts show available-series counts, missingness, illustrative FX paths and monthly changes, macroeconomic series, and dollar-credit stocks. FX paths use January 2010 = 100 to make currencies with different denominations comparable. Examples are illustrative and do not define the final sample.", "",
              "The illustrative policy-rate chart starts in 2000 because Brazil's very high early-1994 rates would flatten the other lines on a linear axis. All early observations remain in the exported matrices and range tables. The monthly FX-change histograms show most observations near zero with some large depreciation months. Türkiye's indexed FX path rises substantially relative to the other illustrative currencies. The outside-US USD-credit stock rises over the displayed history, while individual country paths differ.", "",
              "### Largest observed monthly FX changes", "",
              markdown_table(tables["largest_fx_changes"].head(10)[["period", "economy", "currency", "monthly_change_pct"]]), "",
              "Large changes are review flags, not proven data errors. Currency redenominations, regime shifts and historical source breaks need checking before modeling; values are retained. Monthly changes use adjacent calendar months and never forward-fill missing values.", ""]
    for path in figures:
        report += [f"![{path.stem.replace('eda_', '').replace('_', ' ')}](../figures/{path.name})", ""]
    report += ["## Decisions still needed before modeling", "",
               "Confirm the exact 39 currencies, currency/economy mapping, emerging/advanced groups and peg exclusion dates. Review short, discontinued and internally incomplete series individually. Choose any shared euro-area policy-rate mapping explicitly. Do not backfill unavailable history.", "",
               "Credit remains quarterly and unlagged here. The project’s two-quarter lag and monthly availability rule belong in feature construction. These are latest revised historical exports, not a real-time vintage; reference dates do not prove when a predictor was known.", "",
               "## Sources and reproduction", "",
               "Official [BIS bulk download listing](https://data.bis.org/bulkdownload), [FX definitions](https://data.bis.org/topics/XRU), [effective exchange rates](https://data.bis.org/topics/EER), [policy rates](https://data.bis.org/topics/CBPOL), [consumer prices](https://data.bis.org/topics/CPI), and [global liquidity](https://data.bis.org/topics/GLI).", "",
               "```bash", "python scripts/download_data.py", "python scripts/run_eda.py", "```", "",
               "`notebooks/01_eda.ipynb` runs the same analysis and displays the tables and charts. `data/raw/download_manifest.json` records exact URLs, download timestamps, hashes and HTTP metadata. `data/processed/eda_*.csv` contains the selected descriptive matrices and metadata, and `reports/eda/*.csv` contains the quality tables.", ""]
    for name, source in manifest.items():
        report.append(f"- [{name}]({source['source_url']}), CSV SHA-256 `{source['sha256']}`.")
    path = REPORT_DIR / "README.md"
    path.write_text("\n".join(report) + "\n")
    return path
