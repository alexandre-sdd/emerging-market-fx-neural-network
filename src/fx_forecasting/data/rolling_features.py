"""Causal FX-only cross sections for the R2K-Means research experiment.

This deliberately does not implement the project's production macro panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT

FEATURES = ("annual_log_change", "annualized_volatility")


def load_selected_fx(
    ingested_dir: Path = INGESTED_DIR,
    sample_path: Path = PROJECT_ROOT / "config/research_sample.json",
    *, observations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read exact configured series; mask launches, pegs, and invalid FX."""
    cfg = json.loads(Path(sample_path).read_text())
    path = Path(ingested_dir) / "observations.parquet"
    if observations is None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run scripts/process_data.py first.")
        rows = pd.read_parquet(path, columns=["dataset", "frequency", "series_id", "period", "value"])
    else:
        rows = observations
    sources = {c["source_series"]["fx"]: c["currency"] for c in cfg["currencies"]}
    if len(sources) != len(cfg["currencies"]) or len(set(sources.values())) != len(sources):
        raise ValueError("Sample must map unique FX series to unique currencies")
    rows = rows.loc[rows.dataset.eq("WS_XRU") & rows.frequency.eq("M") & rows.series_id.isin(sources)].copy()
    missing = set(sources) - set(rows.series_id)
    if missing:
        raise ValueError(f"Configured FX series absent: {sorted(missing)}")
    rows["currency"] = rows.series_id.map(sources)
    fx = rows.pivot(index="period", columns="currency", values="value")
    fx.index = pd.PeriodIndex(fx.index, freq="M")
    calendar = pd.period_range(**cfg["monthly_window"], freq="M")
    fx = fx.reindex(index=calendar, columns=sorted(sources.values())).astype(float)
    fx = fx.where(np.isfinite(fx) & fx.gt(0))
    for c in cfg["currencies"]:
        fx.loc[fx.index < pd.Period(c["first_eligible_month"], "M"), c["currency"]] = np.nan
    area_to_currency = {c["area"]: c["currency"] for c in cfg["currencies"]}
    for interval in cfg["excluded_month_intervals"]:
        fx.loc[interval["start"]:interval["end"], area_to_currency[interval["area"]]] = np.nan
    metadata = pd.DataFrame(cfg["currencies"]).drop(columns="source_series").set_index("currency")
    return fx, metadata, cfg


def make_rolling_features(fx: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Trailing annualized mean log change and volatility, percent units.

    Require lookback consecutive monthly returns (lookback + 1 FX levels).
    Positive change is local-currency depreciation because BIS quotes LCU/USD.
    Output includes ineligible rows with NaNs, so coverage remains auditable.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    if not isinstance(fx.index, pd.PeriodIndex) or fx.index.freqstr != "M":
        raise ValueError("FX requires a monthly PeriodIndex")
    if fx.empty or not fx.index.is_unique or not fx.columns.is_unique:
        raise ValueError("FX calendar and currencies must be nonempty and unique")
    expected = pd.period_range(fx.index.min(), fx.index.max(), freq="M")
    if not fx.index.equals(expected):
        raise ValueError("FX must include every calendar month in ascending order")
    logfx = np.log(fx.where(np.isfinite(fx) & fx.gt(0)))
    changes = 100 * logfx.diff()  # no fill and no bridging an excluded month
    moments = {
        FEATURES[0]: 12 * changes.rolling(lookback, min_periods=lookback).mean(),
        FEATURES[1]: np.sqrt(12) * changes.rolling(lookback, min_periods=lookback).std(ddof=1),
    }
    frames = []
    for currency in fx:
        part = pd.DataFrame({name: value[currency] for name, value in moments.items()})
        part["currency"] = currency
        part.index.name = "period"
        frames.append(part.reset_index())
    return pd.concat(frames, ignore_index=True).set_index(["period", "currency"]).sort_index()


def normalize_cross_section(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current-period z-scores, ddof=0; no time-series/future observations.

    A constant feature is mapped to zero with scale 1 and flagged. Missing rows
    must be filtered explicitly by the caller before invoking this function.
    """
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Normalize a nonempty finite cross section")
    mean, std = frame.mean(), frame.std(ddof=0)
    constant = std.le(np.finfo(float).eps)
    scale = std.mask(constant, 1.0)
    stats = pd.DataFrame({"mean": mean, "scale": scale, "constant": constant})
    return (frame - mean) / scale, stats
