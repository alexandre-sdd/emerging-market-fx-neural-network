"""Read unchanged BIS wide CSV exports without mixing frequencies or units."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fx_forecasting.config import RAW_DIR


@dataclass
class BISExport:
    name: str
    metadata: pd.DataFrame
    values: pd.DataFrame  # periods in rows; BIS series identifiers in columns
    audit: dict


def load_export(name: str, frequency: str = "M", *, raw_dir: Path | None = None) -> BISExport:
    """Read native monthly/quarterly observations; audit all original rows.

    The bulk file contains unrelated frequencies on the same wide time axis.
    Blanks on that axis are structural, so only columns matching the requested
    frequency enter the observation matrix. No imputation or deduplication.
    """
    path = (RAW_DIR if raw_dir is None else Path(raw_dir)) / f"{name}.csv"
    seen_rows: set[bytes] = set()
    ids: Counter = Counter()
    frequencies: Counter = Counter()
    duplicate_rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate CSV columns in {path}")
        series_position = header.index("Series")
        frequency_position = header.index("FREQ")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Malformed row in {path}")
            # Hash parsed cells so quotation differences do not hide duplicates.
            digest = hashlib.sha256(repr(row).encode()).digest()
            duplicate_rows += digest in seen_rows
            seen_rows.add(digest)
            ids[row[series_position]] += 1
            frequencies[row[frequency_position]] += 1
    pattern = r"\d{4}-\d{2}" if frequency == "M" else r"\d{4}-Q[1-4]"
    if frequency not in {"M", "Q"}:
        raise ValueError("This BIS loader supports monthly and quarterly data only")
    periods = [c for c in header if re.fullmatch(pattern, c)]
    metadata_columns = [c for c in header if not c[:1].isdigit()]
    frame = pd.read_csv(
        path, usecols=metadata_columns + periods, dtype=str,
        keep_default_na=False, encoding="utf-8-sig",
    )
    frame = frame.loc[frame["FREQ"].eq(frequency)].copy()
    metadata = frame[metadata_columns].set_index("Series")
    source_missing = frame[periods].isin(["", "NaN", "nan"])
    raw = frame[periods].mask(source_missing)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    invalid_examples = []
    if invalid.to_numpy().any():
        invalid_examples = [
            {"series": frame.iloc[i]["Series"], "period": periods[j], "token": raw.iloc[i, j]}
            for i, j in zip(*invalid.to_numpy().nonzero())
        ][:10]
    numeric.index = metadata.index
    values = numeric.T
    values.index = pd.PeriodIndex(values.index, freq=frequency)
    values = values.sort_index()
    return BISExport(name, metadata, values, {
        "dataset": name,
        "raw_series_rows": sum(frequencies.values()),
        "raw_exact_duplicate_rows": int(duplicate_rows),
        "raw_duplicate_series_rows": sum(n - 1 for n in ids.values()),
        "raw_blank_series_ids": ids.get("", 0),
        "raw_frequencies": dict(frequencies),
        "loaded_frequency": frequency,
        "explicit_nan_tokens": int(frame[periods].isin(["NaN", "nan"]).to_numpy().sum()),
        "invalid_numeric_tokens": int(invalid.to_numpy().sum()),
        "invalid_numeric_examples": invalid_examples,
    })


def select_series(export: BISExport, **filters: str) -> BISExport:
    mask = pd.Series(True, index=export.metadata.index)
    for column, value in filters.items():
        mask &= export.metadata[column].eq(value)
    metadata = export.metadata.loc[mask].copy()
    if metadata.empty:
        raise ValueError(f"No {export.name} series match {filters}")
    if not metadata.index.is_unique:
        raise ValueError(f"Duplicate series identifiers in selected {export.name} data")
    return BISExport(export.name, metadata, export.values[metadata.index], export.audit)


def load_project_datasets(*, raw_dir: Path | None = None) -> dict[str, BISExport]:
    """Choose the project's definitions, keeping the FX universe unrestricted.

    GLI includes total USD credit outside the US plus the 12 country series.
    Geographic aggregates and bank-loan/debt-security components are excluded.
    """
    fx = select_series(load_export("WS_XRU", raw_dir=raw_dir), COLLECTION="E")
    eer = load_export("WS_EER", raw_dir=raw_dir)
    policy = load_export("WS_CBPOL", raw_dir=raw_dir)
    cpi = select_series(load_export("WS_LONG_CPI", raw_dir=raw_dir), UNIT_MEASURE="628")
    credit = select_series(
        load_export("WS_GLI", "Q", raw_dir=raw_dir), CURR_DENOM="USD", BORROWERS_SECTOR="N",
        LENDERS_SECTOR="A", L_POS_TYPE="I", L_INSTR="B", UNIT_MEASURE="USD",
    )
    keep = credit.metadata["BORROWERS_CTY"].str.fullmatch(r"[A-Z]{2}") | credit.metadata["BORROWERS_CTY"].eq("3P")
    credit = BISExport(credit.name, credit.metadata.loc[keep], credit.values.loc[:, keep], credit.audit)
    return {
        "fx": fx,
        "reer": select_series(eer, EER_TYPE="R", EER_BASKET="B"),
        "neer": select_series(eer, EER_TYPE="N", EER_BASKET="B"),
        "policy": policy,
        "cpi": cpi,
        "usd_credit": credit,
    }


def load_eda_datasets() -> dict[str, BISExport]:
    """Compatibility entry point for the existing exploration notebook."""
    return load_project_datasets()
