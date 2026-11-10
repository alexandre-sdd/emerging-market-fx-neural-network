"""Canonical causal FX/macro features for the multivariate neural handoff.

Economic values and NaNs are preserved. Learned preprocessing belongs to the
training-cutoff dataset builder. Lags describe revised-history assumptions, not
historical real-time vintages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT
from fx_forecasting.data.rolling_features import load_selected_fx

LOCAL_FEATURES = ("fx_change_1m", "fx_change_3m", "fx_change_12m", "volatility_12m",
                  "reer_gap", "rate_diff", "inflation_diff")
GLOBAL_FEATURES = ("usd_neer_change", "usd_credit_log_stock", "usd_credit_yoy")
OPTIONAL_LOCAL = ("rate_diff", "inflation_diff")


@dataclass
class FeaturePanel:
    fx: pd.DataFrame
    metadata: pd.DataFrame
    config: dict
    local: pd.DataFrame
    global_features: pd.DataFrame
    provenance: pd.DataFrame
    global_provenance: pd.DataFrame


def _source(obs: pd.DataFrame, dataset: str, source: str, calendar: pd.PeriodIndex) -> pd.Series:
    frequency = "Q" if calendar.freqstr.startswith("Q") else "M"
    selected = obs.loc[obs.dataset.eq(dataset) & obs.frequency.eq(frequency)
                       & obs.series_id.eq(source), ["period", "value"]].copy()
    if selected.empty:
        raise ValueError(f"Configured series absent: {dataset}/{source}")
    selected.index = pd.PeriodIndex(selected.period.astype(str).str.replace("-Q", "Q", regex=False), freq=calendar.freq)
    if not selected.index.is_unique:
        raise ValueError(f"Duplicate observation periods: {dataset}/{source}")
    values = selected.value.astype(float)
    return values.where(np.isfinite(values)).reindex(calendar)


def _carry_one(raw):
    # Using raw.shift, rather than the carried result, limits this to one month.
    carried = raw.isna() & raw.shift(1).notna()
    return raw.where(~carried, raw.shift(1)), carried


def _long(wides, calendar, currencies):
    index = pd.MultiIndex.from_product([calendar, currencies], names=["period", "currency"])
    return pd.DataFrame({name: value.reindex(index=calendar, columns=currencies).to_numpy().reshape(-1)
                         for name, value in wides.items()}, index=index)


def build_feature_panel(*, ingested_dir: Path = INGESTED_DIR,
                        sample_path: Path = PROJECT_ROOT / "config/research_sample.json",
                        observations: pd.DataFrame | None = None) -> FeaturePanel:
    """Build local/global features and provenance from exact configured sources.

    Optional macro gaps remain NaN. Carry flags refer to available values after
    the lag; nominal and effective reference periods are distinguished.
    """
    if observations is None:
        path = Path(ingested_dir) / "observations.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run scripts/process_data.py first.")
        observations = pd.read_parquet(path, columns=["dataset", "frequency", "series_id", "period", "value"])
    required = {"dataset", "frequency", "series_id", "period", "value"}
    if not required.issubset(observations.columns):
        raise ValueError(f"Observation columns missing: {sorted(required-set(observations.columns))}")
    fx, metadata, cfg = load_selected_fx(ingested_dir, sample_path, observations=observations)
    calendar, currencies = fx.index, fx.columns
    lags = cfg["macro_availability"]["reference_lags_months"]
    for key in ("policy", "cpi_inflation", "reer", "usd_neer"):
        if not isinstance(lags[key], int) or lags[key] < 0:
            raise ValueError(f"Invalid reference lag for {key}")
    for key in ("policy", "inflation"):
        if cfg["missing_data"][key]["causal_carry_limit_months"] != 1:
            raise ValueError("The panel implements exactly one causal macro carry month")

    def broadcast(series):
        return pd.DataFrame(np.repeat(series.to_numpy()[:, None], len(currencies), axis=1),
                            index=calendar, columns=currencies)

    entries = {c["currency"]: c for c in cfg["currencies"]}
    prelaunch = pd.DataFrame(False, index=calendar, columns=currencies)
    regime = prelaunch.copy()
    for currency, entry in entries.items():
        prelaunch[currency] = calendar < pd.Period(entry["first_eligible_month"], "M")
    area_to_currency = {c["area"]: c["currency"] for c in cfg["currencies"]}
    for interval in cfg["excluded_month_intervals"]:
        regime.loc[interval["start"]:interval["end"], area_to_currency[interval["area"]]] = True
    macro = {}
    for key, dataset in (("reer", "WS_EER"), ("policy", "WS_CBPOL"), ("cpi", "WS_LONG_CPI")):
        table = pd.DataFrame(np.nan, index=calendar, columns=currencies)
        for currency, entry in entries.items():
            source = entry["source_series"][key]
            if source is not None:
                values = _source(observations, dataset, source, calendar)
                table[currency] = values if key == "policy" else values.where(values.gt(0))
        macro[key] = table.mask(prelaunch)

    logs = np.log(fx)
    features = {}
    for h in (1, 3, 12):
        valid = fx.notna().rolling(h+1, min_periods=h+1).sum().eq(h+1)
        features[f"fx_change_{h}m"] = (100*logs.diff(h)).where(valid)
    features["volatility_12m"] = np.sqrt(12)*features["fx_change_1m"].rolling(12, min_periods=12).std(ddof=1)
    reer = np.log(macro["reer"]).shift(lags["reer"])
    features["reer_gap"] = 100*(reer-reer.shift(1).rolling(60, min_periods=60).mean())

    policy_us_raw = _source(observations, "WS_CBPOL", "M:US", calendar)
    policy, policy_carried = _carry_one(macro["policy"])
    policy_us, policy_us_carried = _carry_one(policy_us_raw)
    policy, policy_us = policy.shift(lags["policy"]), policy_us.shift(lags["policy"])
    policy_carried = policy_carried.shift(lags["policy"], fill_value=False)
    policy_us_carried = policy_us_carried.shift(lags["policy"], fill_value=False)
    features["rate_diff"] = policy.sub(policy_us, axis=0)

    cpi_us = _source(observations, "WS_LONG_CPI", "M:US:628", calendar)
    cpi_us = cpi_us.where(cpi_us.gt(0))
    inflation, inflation_carried = _carry_one(100*np.log(macro["cpi"]).diff(12))
    inflation_us, inflation_us_carried = _carry_one(100*np.log(cpi_us).diff(12))
    inflation = inflation.shift(lags["cpi_inflation"])
    inflation_us = inflation_us.shift(lags["cpi_inflation"])
    inflation_carried = inflation_carried.shift(lags["cpi_inflation"], fill_value=False)
    inflation_us_carried = inflation_us_carried.shift(lags["cpi_inflation"], fill_value=False)
    features["inflation_diff"] = inflation.sub(inflation_us, axis=0)
    local = _long({name: features[name] for name in LOCAL_FEATURES}, calendar, currencies)

    def reference(lag):
        return broadcast(pd.Series((calendar-lag).astype(str), index=calendar))

    def effective_reference(lag, carried, available):
        months = np.repeat((calendar-lag).asi8[:, None], len(currencies), axis=1)-carried.to_numpy(dtype=int)
        result = pd.DataFrame(months.astype("datetime64[M]").astype(str), index=calendar, columns=currencies)
        return result.where(available)

    absent_policy = pd.Series({c: entries[c]["source_series"]["policy"] is None for c in currencies})
    absent_cpi = pd.Series({c: entries[c]["source_series"]["cpi"] is None for c in currencies})
    provenance_wide = {
        "policy_carried_domestic": policy_carried, "policy_carried_us": broadcast(policy_us_carried),
        "inflation_carried_domestic": inflation_carried, "inflation_carried_us": broadcast(inflation_us_carried),
        "policy_missing": features["rate_diff"].isna(), "inflation_missing": features["inflation_diff"].isna(),
        "policy_source_absent": pd.DataFrame(np.tile(absent_policy.to_numpy(), (len(calendar), 1)), index=calendar, columns=currencies),
        "cpi_source_absent": pd.DataFrame(np.tile(absent_cpi.to_numpy(), (len(calendar), 1)), index=calendar, columns=currencies),
        "fx_regime_excluded": regime, "prelaunch": prelaunch, "fx_missing": fx.isna(),
        "fx_reference_period": reference(0), "reer_reference_period": reference(lags["reer"]),
        "policy_reference_period": reference(lags["policy"]),
        "inflation_reference_period": reference(lags["cpi_inflation"]),
        "policy_value_reference_period_domestic": effective_reference(lags["policy"], policy_carried, policy.notna()),
        "policy_value_reference_period_us": effective_reference(lags["policy"], broadcast(policy_us_carried), broadcast(policy_us.notna())),
        "inflation_value_reference_period_domestic": effective_reference(lags["cpi_inflation"], inflation_carried, inflation.notna()),
        "inflation_value_reference_period_us": effective_reference(lags["cpi_inflation"], broadcast(inflation_us_carried), broadcast(inflation_us.notna())),
    }
    for key in ("fx", "reer", "policy", "cpi"):
        provenance_wide[f"{key}_source"] = pd.DataFrame(
            [[entries[c]["source_series"][key] for c in currencies]]*len(calendar), index=calendar, columns=currencies,
        )
    provenance = _long(provenance_wide, calendar, currencies)

    usd_neer = _source(observations, "WS_EER", "M:N:B:US", calendar)
    usd_neer = np.log(usd_neer.where(usd_neer.gt(0))).shift(lags["usd_neer"])
    credit_id = cfg["macro_availability"]["usd_credit_series"]
    credit_rows = observations.loc[observations.dataset.eq("WS_GLI") & observations.series_id.eq(credit_id)]
    if credit_rows.empty:
        raise ValueError(f"Configured series absent: WS_GLI/{credit_id}")
    quarters = pd.PeriodIndex(credit_rows.period.astype(str).str.replace("-Q", "Q", regex=False), freq="Q")
    credit_calendar = pd.period_range(quarters.min(), quarters.max(), freq="Q")
    credit = _source(observations, "WS_GLI", credit_id, credit_calendar)
    credit_log = np.log(credit.where(credit.gt(0)))
    credit_lag = cfg["macro_availability"]["usd_credit_quarter_lag"]
    if not isinstance(credit_lag, int) or credit_lag < 0:
        raise ValueError("Invalid USD credit quarter lag")
    quarter_lookup = calendar.asfreq("Q")-credit_lag
    global_features = pd.DataFrame({
        "usd_neer_change": 100*usd_neer.diff(),
        "usd_credit_log_stock": credit_log.reindex(quarter_lookup).to_numpy(),
        "usd_credit_yoy": (100*credit_log.diff(4)).reindex(quarter_lookup).to_numpy(),
    }, index=calendar)[list(GLOBAL_FEATURES)]
    global_features.index.name = "period"
    global_provenance = pd.DataFrame({
        "usd_neer_source": "M:N:B:US", "usd_neer_reference_period": (calendar-lags["usd_neer"]).astype(str),
        "usd_neer_previous_reference_period": (calendar-lags["usd_neer"]-1).astype(str),
        "usd_credit_source": credit_id, "usd_credit_reference_quarter": quarter_lookup.astype(str),
        "usd_credit_yoy_base_quarter": (quarter_lookup-4).astype(str),
        **{f"{name}_missing": global_features[name].isna() for name in GLOBAL_FEATURES},
    }, index=calendar)
    global_provenance.index.name = "period"
    metadata = metadata.reindex(currencies)
    metadata["policy_source_absent"], metadata["cpi_source_absent"] = absent_policy, absent_cpi
    return FeaturePanel(fx, metadata, cfg, local, global_features, provenance, global_provenance)
