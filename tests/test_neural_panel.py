"""Causal local/global panel rules before neural-dataset preprocessing."""

import json

import numpy as np
import pandas as pd
import pytest

from fx_forecasting.data.panel import GLOBAL_FEATURES, LOCAL_FEATURES, build_feature_panel


@pytest.fixture
def sources(tmp_path):
    dates = pd.period_range("2018-01", "2026-08", freq="M")
    entries = [
        {"area": "AA", "currency": "AAA", "economy": "A", "group": "test", "first_eligible_month": "2018-01",
         "source_series": {"fx": "M:AA:AAA:E", "reer": "M:R:B:AA", "policy": "M:AA", "cpi": "M:AA:628"}},
        {"area": "BB", "currency": "BBB", "economy": "B", "group": "test", "first_eligible_month": "2018-01",
         "source_series": {"fx": "M:BB:BBB:E", "reer": "M:R:B:BB", "policy": None, "cpi": None}},
        {"area": "XM", "currency": "EUR", "economy": "Euro", "group": "test", "first_eligible_month": "2020-01",
         "source_series": {"fx": "M:XM:EUR:E", "reer": "M:R:B:XM", "policy": "M:XM", "cpi": "M:XM:628"}},
    ]
    cfg = {"currencies": entries, "monthly_window": {"start": str(dates.min()), "end": str(dates.max())},
           "excluded_month_intervals": [{"area": "AA", "start": "2021-03", "end": "2021-04"}],
           "macro_availability": {"reference_lags_months": {"policy": 1, "cpi_inflation": 2, "reer": 2, "usd_neer": 1},
                                  "usd_credit_quarter_lag": 2, "usd_credit_series": "Q:USD:3P:N:A:I:B:USD"},
           "missing_data": {"policy": {"causal_carry_limit_months": 1}, "inflation": {"causal_carry_limit_months": 1}}}
    records = []

    def monthly(dataset, source, values):
        records.extend({"dataset": dataset, "series_id": source, "frequency": "M", "period": str(t), "value": value}
                       for t, value in zip(dates, values))

    t = np.arange(len(dates))
    for i, entry in enumerate(entries):
        mapping = entry["source_series"]
        monthly("WS_XRU", mapping["fx"], np.exp(.01*(i+1)*t))
        monthly("WS_EER", mapping["reer"], 100*np.exp(.002*t))
        if mapping["policy"]:
            monthly("WS_CBPOL", mapping["policy"], -.5+.001*t)
        if mapping["cpi"]:
            monthly("WS_LONG_CPI", mapping["cpi"], 100*np.exp(.003*t))
    monthly("WS_CBPOL", "M:US", .2+.002*t)
    monthly("WS_LONG_CPI", "M:US:628", 100*np.exp(.002*t))
    monthly("WS_EER", "M:N:B:US", 100*np.exp(.001*t))
    quarters = pd.period_range("2017Q1", "2026Q2", freq="Q")
    records.extend({"dataset": "WS_GLI", "series_id": cfg["macro_availability"]["usd_credit_series"],
                    "frequency": "Q", "period": str(q), "value": 1000*np.exp(.04*i)}
                   for i, q in enumerate(quarters))
    obs = pd.DataFrame(records)
    cfg_path = tmp_path / "sample.json"
    cfg_path.write_text(json.dumps(cfg))
    return obs, cfg_path, dates, quarters


def panel(sources, obs=None):
    observations, path, _, _ = sources
    return build_feature_panel(sample_path=path, observations=observations if obs is None else obs)


def set_value(obs, source, period, value):
    obs.loc[obs.series_id.eq(source) & obs.period.eq(period), "value"] = value


def test_feature_order_reference_lags_and_negative_policy_are_preserved(sources):
    result = panel(sources)
    assert tuple(result.local) == LOCAL_FEATURES
    assert tuple(result.global_features) == GLOBAL_FEATURES
    assert list(result.fx) == ["AAA", "BBB", "EUR"]
    row = result.local.loc[(pd.Period("2020-05", "M"), "AAA")]
    assert row.fx_change_1m == pytest.approx(1)
    assert row.fx_change_3m == pytest.approx(3)
    assert row.fx_change_12m == pytest.approx(12)
    # At May 2020, April is t=27 relative to January 2018.
    assert row.rate_diff == pytest.approx((-.5+.001*27)-(.2+.002*27))
    assert row.inflation_diff == pytest.approx(100*(.003-.002)*12)
    assert result.provenance.loc[(pd.Period("2020-05", "M"), "AAA"), "policy_reference_period"] == "2020-04"
    assert result.global_features.loc["2020-05", "usd_neer_change"] == pytest.approx(.1)
    assert not result.provenance.policy_carried_domestic.any()


def test_fx_exclusions_and_missing_levels_reset_every_dependent_path(sources):
    obs = sources[0].copy()
    set_value(obs, "M:BB:BBB:E", "2023-07", np.nan)
    set_value(obs, "M:BB:BBB:E", "2023-09", -1)
    result = panel(sources, obs)
    a = result.local.xs("AAA", level="currency")
    assert a.loc["2021-03":"2021-05", "fx_change_1m"].isna().all()
    assert a.loc["2021-06", "fx_change_1m"] == pytest.approx(1)
    assert np.isnan(a.loc["2021-06", "fx_change_3m"])
    assert a.loc["2021-08", "fx_change_3m"] == pytest.approx(3)
    b = result.local.xs("BBB", level="currency")
    assert b.loc["2023-07":"2023-10", "fx_change_1m"].isna().all()
    assert np.isnan(b.loc["2023-12", "volatility_12m"])
    assert b.loc["2024-10", "volatility_12m"] == pytest.approx(0, abs=1e-12)
    assert result.provenance.loc[(pd.Period("2021-03", "M"), "AAA"), "fx_regime_excluded"]


def test_policy_carry_only_first_gap_month_and_records_actual_reference(sources):
    obs = sources[0].copy()
    for period in ("2022-03", "2022-04", "2022-05"):
        set_value(obs, "M:AA", period, np.nan)
    set_value(obs, "M:US", "2022-06", np.nan)
    result = panel(sources, obs)
    p = result.provenance.xs("AAA", level="currency")
    a = result.local.xs("AAA", level="currency")
    assert p.loc["2022-04", "policy_carried_domestic"]
    assert p.loc["2022-04", "policy_reference_period"] == "2022-03"
    assert p.loc["2022-04", "policy_value_reference_period_domestic"] == "2022-02"
    assert not p.loc["2022-05", "policy_carried_domestic"]
    assert a.loc["2022-05":"2022-06", "rate_diff"].isna().all()
    assert pd.isna(p.loc["2022-05", "policy_value_reference_period_domestic"])
    assert p.loc["2022-07", "policy_carried_us"]


def test_inflation_carries_derived_rate_and_us_shutdown_is_causal(sources):
    obs = sources[0].copy()
    set_value(obs, "M:US:628", "2025-10", np.nan)
    set_value(obs, "M:AA:628", "2022-03", np.nan)
    set_value(obs, "M:AA:628", "2022-04", np.nan)
    result = panel(sources, obs)
    p = result.provenance.xs("AAA", level="currency")
    a = result.local.xs("AAA", level="currency")
    assert p.loc["2025-12", "inflation_carried_us"]
    assert p.loc["2025-12", "inflation_reference_period"] == "2025-10"
    assert p.loc["2025-12", "inflation_value_reference_period_us"] == "2025-09"
    assert a.loc["2025-12", "inflation_diff"] == pytest.approx(1.2)
    assert p.loc["2022-05", "inflation_carried_domestic"]
    assert not p.loc["2022-06", "inflation_carried_domestic"]
    assert np.isnan(a.loc["2022-06", "inflation_diff"])
    assert obs.loc[obs.series_id.eq("M:US:628") & obs.period.eq("2025-10"), "value"].isna().all()


def test_structural_macro_absence_and_launch_masks_do_not_remove_currency(sources):
    result = panel(sources)
    b = result.local.xs("BBB", level="currency")
    assert b[["rate_diff", "inflation_diff"]].isna().all().all()
    p = result.provenance.xs("BBB", level="currency")
    assert p.policy_source_absent.all() and p.cpi_source_absent.all()
    assert not p.policy_carried_domestic.any()
    euro = result.local.xs("EUR", level="currency")
    assert euro.loc[:"2019-12"].isna().all().all()
    assert euro.loc["2020-02", "rate_diff"] < 0
    assert np.isnan(euro.loc["2021-02", "inflation_diff"])
    assert euro.loc["2021-03", "inflation_diff"] == pytest.approx(1.2)
    assert np.isnan(euro.loc["2025-02", "reer_gap"])
    assert np.isfinite(euro.loc["2025-03", "reer_gap"])


def test_reer_gap_uses_lagged_current_value_and_previous_sixty_months(sources):
    result = panel(sources)
    # At March 2023 the last observed REER reference is January 2023 (t=60).
    # The previous 60 log-level offsets average 29.5, so 60-29.5 = 30.5.
    assert result.local.loc[(pd.Period("2023-03", "M"), "AAA"), "reer_gap"] == pytest.approx(100*.002*30.5)


def test_credit_mapping_respects_quarter_boundaries_without_missing_quarter_carry(sources):
    obs = sources[0].copy()
    source = "Q:USD:3P:N:A:I:B:USD"
    set_value(obs, source, "2026Q2", np.nan)
    set_value(obs, source, "2023Q2", np.nan)
    result = panel(sources, obs)
    global_values, provenance = result.global_features, result.global_provenance
    assert provenance.loc["2024-03", "usd_credit_reference_quarter"] == "2023Q3"
    assert provenance.loc["2024-04", "usd_credit_reference_quarter"] == "2023Q4"
    assert global_values.loc["2024-04", "usd_credit_yoy"] == pytest.approx(16)
    assert global_values.loc["2023-10":"2023-12", "usd_credit_log_stock"].isna().all()
    assert global_values.loc["2026-07", "usd_credit_log_stock"] == pytest.approx(np.log(1000)+.04*36)
    assert provenance.loc["2026-08", "usd_credit_reference_quarter"] == "2026Q1"


def test_future_observation_changes_leave_feature_and_provenance_prefix_unchanged(sources):
    obs = sources[0].copy()
    changed = obs.copy()
    cutoff = pd.Period("2023-01", "M")
    end_month = [pd.Period(row.period, row.frequency).asfreq("M", "end") for row in changed.itertuples()]
    changed.loc[np.array(end_month) > cutoff, "value"] *= 100
    a, b = panel(sources, obs), panel(sources, changed)
    prefix = a.local.index.get_level_values("period") <= cutoff
    pd.testing.assert_frame_equal(a.local.loc[prefix], b.local.loc[prefix])
    pd.testing.assert_frame_equal(a.provenance.loc[prefix], b.provenance.loc[prefix])
    pd.testing.assert_frame_equal(a.global_features.loc[:cutoff], b.global_features.loc[:cutoff])


def test_configured_optional_source_must_exist_and_duplicates_fail(sources):
    obs = sources[0]
    with pytest.raises(ValueError, match="Configured series absent: WS_CBPOL/M:AA"):
        panel(sources, obs.loc[~obs.series_id.eq("M:AA")])
    duplicate = pd.concat([obs, obs.loc[obs.series_id.eq("M:AA")].head(1)])
    with pytest.raises(ValueError, match="Duplicate observation"):
        panel(sources, duplicate)
