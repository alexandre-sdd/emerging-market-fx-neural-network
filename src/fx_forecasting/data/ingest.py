"""Ingest BIS source exports into typed observation and series tables.

This stage only selects source definitions, normalizes their structure, and
validates records. It does not construct predictors, targets, or a joined panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fx_forecasting.config import INGESTED_DIR, RAW_DIR
from fx_forecasting.data.loaders import BISExport, load_project_datasets

# Values retain the source scale. For example, credit is in millions of USD,
# represented by unit_code=USD and unit_multiplier=6, without rescaling values.
MEASURES = {
    "fx": ("LCU_PER_USD", "Local currency units per USD", None, "E"),
    "reer": ("INDEX", "Broad real effective rate, 2020 = 100", 2020, "A"),
    "neer": ("INDEX", "Broad nominal effective rate, 2020 = 100", 2020, "A"),
    "policy": ("PERCENT_PER_ANNUM", "Percent per annum", None, "E"),
    "cpi": ("INDEX", "Consumer price index, 2010 = 100", 2010, "E"),
    "usd_credit": ("USD", "Millions of USD", None, None),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_periods(component: str, start: str, end: str) -> pd.PeriodIndex:
    for value in (start, end):
        if not re.fullmatch(r"\d{4}-\d{2}", value):
            raise ValueError("Dates must use YYYY-MM")
    first, last = pd.Period(start, "M"), pd.Period(end, "M")
    if first > last:
        raise ValueError("Start month must not be after end month")
    if component != "usd_credit":
        return pd.period_range(first, last, freq="M")
    first_quarter = max(pd.Period("2000Q1", "Q"), first.asfreq("Q"))
    last_quarter = last.asfreq("Q")
    if last.month % 3:
        last_quarter -= 1
    if first_quarter > last_quarter:
        raise ValueError("The sample must include a completed credit quarter from 2000 onward")
    return pd.period_range(first_quarter, last_quarter, freq="Q")


def normalize_export(component: str, export: BISExport, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain every selected series/calendar cell, including missing values."""
    for field in ("raw_exact_duplicate_rows", "raw_duplicate_series_rows", "raw_blank_series_ids", "invalid_numeric_tokens"):
        if export.audit[field]:
            raise ValueError(f"{export.name}: {field}={export.audit[field]}; inspect the source before ingestion")
    if not export.metadata.index.is_unique or export.metadata.index.hasnans:
        raise ValueError(f"{export.name}: invalid or duplicate series identifiers")
    if export.metadata.empty:
        raise ValueError(f"{export.name}: no selected series")
    if not export.values.columns.is_unique or set(export.values.columns) != set(export.metadata.index):
        raise ValueError(f"{export.name}: observation columns do not match series metadata")
    if not export.values.index.is_unique:
        raise ValueError(f"{export.name}: duplicate time periods")
    calendar = expected_periods(component, start, end)
    values = export.values.reindex(calendar)
    observations = values.rename_axis("period").reset_index().melt(
        id_vars="period", var_name="series_id", value_name="value",
    )
    observations["period_start"] = observations["period"].dt.start_time
    observations["period_end"] = observations["period"].dt.end_time.dt.normalize()
    observations["period"] = observations["period"].astype(str).str.replace(r"(\d{4})Q", r"\1-Q", regex=True)
    observations["dataset"] = export.name
    observations["frequency"] = "Q" if component == "usd_credit" else "M"
    observations["value"] = pd.to_numeric(observations["value"], errors="raise").astype("Float64")
    if not np.isfinite(observations["value"].dropna().to_numpy(dtype=float)).all():
        raise ValueError(f"{export.name}: infinite numeric observations")
    observations = observations[["dataset", "series_id", "frequency", "period", "period_start", "period_end", "value"]]

    unit_code, unit_label, base_year, collection_default = MEASURES[component]
    series_records = []
    for series_id, source in export.metadata.iterrows():
        country = "BORROWERS_CTY" if component == "usd_credit" else "REF_AREA"
        country_name = "Borrowers' country" if component == "usd_credit" else "Reference area"
        series_records.append({
            "dataset": export.name, "series_id": series_id, "component": component,
            "frequency": source["FREQ"], "reference_area": source[country],
            "reference_area_name": source[country_name],
            "currency_code": source.get("CURRENCY") or source.get("CURR_DENOM") or None,
            "unit_code": unit_code, "unit_label": unit_label,
            "unit_multiplier": int(source.get("UNIT_MULT") or 0),
            "index_base_year": base_year,
            "collection": source.get("COLLECTION") or collection_default,
            "source_title": source.get("TITLE") or source.get("TITLE_TS") or None,
            "source_metadata_json": json.dumps(source.to_dict(), sort_keys=True, ensure_ascii=False),
        })
    series = pd.DataFrame(series_records)
    series["unit_multiplier"] = series["unit_multiplier"].astype("Int64")
    series["index_base_year"] = series["index_base_year"].astype("Int64")
    return observations, series


def ingest_data(*, raw_dir: Path = RAW_DIR, output_dir: Path = INGESTED_DIR,
                start: str = "1994-01", end: str = "2026-08") -> dict:
    # Validate sample dates before parsing source files or writing output.
    expected_periods("fx", start, end)
    expected_periods("usd_credit", start, end)
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if raw_dir.resolve() == output_dir.resolve():
        raise ValueError("Ingested output must use a separate directory from raw exports")
    download_manifest_path = raw_dir / "download_manifest.json"
    download_manifest = json.loads(download_manifest_path.read_text()) if download_manifest_path.exists() else {}
    sources = {}
    for name in ("WS_XRU", "WS_EER", "WS_CBPOL", "WS_LONG_CPI", "WS_GLI"):
        path = raw_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run scripts/download_data.py first.")
        digest = sha256_file(path)
        provenance = download_manifest.get(name, {})
        if provenance.get("sha256") and digest != provenance["sha256"]:
            raise ValueError(f"{name}: source differs from its download manifest; verify the source or refresh the download")
        sources[name] = {"file": str(path.resolve()), "sha256": digest,
                         "source_url": provenance.get("source_url"),
                         "downloaded_at_utc": provenance.get("downloaded_at_utc")}
    datasets = load_project_datasets(raw_dir=raw_dir)
    observation_frames, series_frames, counts = [], [], {}
    for component, export in datasets.items():
        observations, series = normalize_export(component, export, start, end)
        observation_frames.append(observations)
        series_frames.append(series)
        counts[component] = {"dataset": export.name, "series": len(series),
                             "rows": len(observations),
                             "nonmissing_values": int(observations["value"].notna().sum()),
                             "missing_values": int(observations["value"].isna().sum())}
    observations = pd.concat(observation_frames, ignore_index=True).sort_values(
        ["dataset", "series_id", "period_start"], ignore_index=True,
    )
    series = pd.concat(series_frames, ignore_index=True).sort_values(["dataset", "series_id"], ignore_index=True)
    if observations.duplicated(["dataset", "series_id", "period"]).any():
        raise ValueError("Duplicate observation keys; nothing was written")
    if series.duplicated(["dataset", "series_id"]).any():
        raise ValueError("Duplicate series keys; nothing was written")
    joined = observations.merge(series[["dataset", "series_id"]], on=["dataset", "series_id"], how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Observations without series metadata; nothing was written")
    # Check sources again to prevent recording a hash from before a changed read.
    for name, source in sources.items():
        if sha256_file(Path(source["file"])) != source["sha256"]:
            raise ValueError(f"{name}: source changed during ingestion; nothing was written")
    manifest = {
        "schema_version": 1, "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "monthly_sample": {"start": start, "end": end},
        "credit_sample": {"start": str(expected_periods("usd_credit", start, end).min()),
                          "end": str(expected_periods("usd_credit", start, end).max())},
        "sources": sources, "components": counts,
        "observation_key": ["dataset", "series_id", "period"],
        "value_convention": "Native BIS values and units; nulls retain missing observations. No rescaling or imputation.",
        "currency_universe": "All selected source series; the final 39-currency roster is not specified.",
        "raw_audits": {name: data.audit for name, data in {d.name: d for d in datasets.values()}.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Finish parsing and validation before replacing any previously ingested files.
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        staging = Path(temporary)
        observations.to_parquet(staging / "observations.parquet", index=False)
        series.to_parquet(staging / "series.parquet", index=False)
        manifest["outputs"] = {
            name: {"sha256": sha256_file(staging / name), "rows": count}
            for name, count in (("observations.parquet", len(observations)), ("series.parquet", len(series)))
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name in ("observations.parquet", "series.parquet", "manifest.json"):
            (staging / name).replace(output_dir / name)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest BIS wide CSV exports into tidy Parquet tables. No features, EDA, or statistical tests.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=INGESTED_DIR)
    parser.add_argument("--start", default="1994-01", help="First sample month, YYYY-MM")
    parser.add_argument("--end", default="2026-08", help="Last sample month, YYYY-MM")
    args = parser.parse_args(argv)
    try:
        manifest = ingest_data(raw_dir=args.raw_dir, output_dir=args.output_dir, start=args.start, end=args.end)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    print(f"Ingested {sum(c['rows'] for c in manifest['components'].values()):,} rows into {args.output_dir}")
    print("Saved observations.parquet, series.parquet, and manifest.json")


if __name__ == "__main__":
    main()
