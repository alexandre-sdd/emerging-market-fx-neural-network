"""Chronology, gap handling, missing policy and matched forecast comparisons."""

import json

import numpy as np
import pandas as pd
import pytest

from fx_forecasting.data.currency_context import (
    rolling_pair_correlations, compact_correlations, correlation_regime_flags,
)
from fx_forecasting.data.panel import build_feature_panel, build_policy_divergence
from fx_forecasting.evaluation.context_screen import make_context_screen_table, run_context_screen


@pytest.fixture
def context_sources(tmp_path):
    dates = pd.period_range("2010-01", "2026-08", freq="M")
    rng = np.random.default_rng(118)
    t = np.arange(len(dates))
    entries, records = [], []

    def monthly(dataset, source, values):
        records.extend({"dataset": dataset, "series_id": source, "frequency": "M",
                        "period": str(date), "value": float(value)} for date, value in zip(dates, values))

    for i, currency in enumerate(("AAA", "CNY", "EUR", "MISS")):
        area = ("AA", "CN", "XM", "MI")[i]
        mapping = {"fx": f"M:{area}:{currency}:E", "reer": f"M:R:B:{area}",
                   "policy": None if currency == "MISS" else f"M:{area}",
                   "cpi": None if currency == "MISS" else f"M:{area}:628"}
        entries.append({"area": area, "currency": currency, "economy": currency, "group": "test",
                        "first_eligible_month": "2010-01", "source_series": mapping})
        monthly("WS_XRU", mapping["fx"], np.exp(rng.normal(0, .01, len(dates)).cumsum()))
        monthly("WS_EER", mapping["reer"], 100*np.exp(.001*t+.01*np.sin(t/8+i)))
        if mapping["policy"]:
            monthly("WS_CBPOL", mapping["policy"], -.5+.25*np.floor(t/(6+i)))
            monthly("WS_LONG_CPI", mapping["cpi"], 100*np.exp(.002*t))
    monthly("WS_CBPOL", "M:US", .25*np.floor(t/7))
    monthly("WS_LONG_CPI", "M:US:628", 100*np.exp(.0015*t))
    monthly("WS_EER", "M:N:B:US", 100*np.exp(.001*t+.01*np.sin(t/12)))
    for i, quarter in enumerate(pd.period_range("2008Q1", "2026Q2", freq="Q")):
        records.append({"dataset": "WS_GLI", "series_id": "Q:USD:3P:N:A:I:B:USD",
                        "frequency": "Q", "period": str(quarter), "value": 1000*np.exp(.01*i)})
    cfg = {"currencies": entries, "monthly_window": {"start": str(dates.min()), "end": str(dates.max())},
           "excluded_month_intervals": [],
           "macro_availability": {"reference_lags_months": {"policy": 1, "cpi_inflation": 2, "reer": 2, "usd_neer": 1},
                                  "usd_credit_quarter_lag": 2, "usd_credit_series": "Q:USD:3P:N:A:I:B:USD"},
           "missing_data": {"policy": {"causal_carry_limit_months": 1}, "inflation": {"causal_carry_limit_months": 1}}}
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(cfg))
    observations = pd.DataFrame(records)
    panel = build_feature_panel(observations=observations, sample_path=path)
    return panel, observations, path


def test_correlations_match_direct_calculation_and_use_exact_pair_counts(context_sources):
    panel, _, _ = context_sources
    pairs = rolling_pair_correlations(panel.fx, windows=(9,))
    date = pd.Period("2018-06", "M")
    raw = 100*np.log(panel.fx).diff()
    for view in ("raw", "median_adjusted"):
        values = raw if view == "raw" else raw.sub(raw.median(axis=1), axis=0)
        history = values.loc[date-8:date]
        row = pairs.loc[pairs.period.eq(date) & pairs.currency_a.eq("AAA") & pairs.currency_b.eq("EUR")
                        & pairs.return_view.eq(view)].iloc[0]
        assert row.observed_pairs == 9
        assert row.correlation == pytest.approx(history.AAA.corr(history.EUR), abs=1e-10)
    assert len(pairs) == len(panel.fx)*6*2
    assert pairs.loc[pairs.period.eq(panel.fx.index[0]), "observed_pairs"].eq(0).all()


def test_fx_gap_breaks_full_window_and_constant_currency_stays_undefined(context_sources):
    fx = context_sources[0].fx.copy()
    fx.loc["2017-01", "AAA"] = np.nan
    fx["MISS"] = 1.
    pairs = rolling_pair_correlations(fx, windows=(3,), return_views=("raw",))
    history = pairs.loc[pairs.currency_a.eq("AAA") & pairs.currency_b.eq("EUR")].set_index("period")
    assert history.loc["2017-01":"2017-04", "correlation"].isna().all()
    assert history.loc["2017-04", "observed_pairs"] == 2
    assert np.isfinite(history.loc["2017-05", "correlation"])
    assert pairs.loc[pairs.currency_b.eq("MISS"), "correlation"].isna().all()
    assert pairs.loc[pairs.currency_b.eq("MISS") & pairs.period.eq(pd.Period("2018-01", "M")), "observed_pairs"].eq(3).all()


def test_compact_features_exclude_self_and_require_enough_peers(context_sources):
    fx = context_sources[0].fx
    pairs = rolling_pair_correlations(fx, windows=(9,), return_views=("raw",))
    groups = {"AAA": 0, "CNY": 1, "EUR": 0, "MISS": 0}
    features, counts = compact_correlations(pairs, list(fx), groups)
    assert features.xs("EUR", level="currency").corr_eur.isna().all()
    assert features.xs("CNY", level="currency")[["corr_cny", "corr_peers"]].isna().all().all()
    assert counts.xs("CNY", level="currency").peer_count.eq(0).all()
    date = pd.Period("2018-05", "M")
    returns = 100*np.log(fx.loc[date-9:date]).diff().iloc[1:]
    expected = np.mean([returns.AAA.corr(returns.EUR), returns.AAA.corr(returns.MISS)])
    assert features.loc[(date, "AAA"), "corr_peers"] == pytest.approx(expected)
    assert counts.loc[(date, "AAA"), "peer_count"] == 2


def test_future_changes_leave_relationship_prefix_and_training_thresholds_unchanged(context_sources):
    fx = context_sources[0].fx
    changed = fx.copy()
    changed.loc["2020-01":] *= np.exp(np.arange(len(changed.loc["2020-01":]))[:, None]*.1)
    groups = dict.fromkeys(fx, 0)
    first = rolling_pair_correlations(fx, windows=(9,))
    second = rolling_pair_correlations(changed, windows=(9,))
    cutoff = pd.Period("2019-12", "M")
    pd.testing.assert_frame_equal(first.loc[first.period <= cutoff], second.loc[second.period <= cutoff])
    a, _ = compact_correlations(first, list(fx), groups)
    b, _ = compact_correlations(second, list(fx), groups)
    fa, ta = correlation_regime_flags(a)
    fb, tb = correlation_regime_flags(b)
    pd.testing.assert_frame_equal(ta, tb)
    pd.testing.assert_frame_equal(fa.loc[:cutoff], fb.loc[:cutoff])


def test_constant_or_insufficient_threshold_histories_are_not_false_regimes(context_sources):
    panel = context_sources[0]
    index = panel.local.index
    features = pd.DataFrame({"constant": 1., "missing": np.nan}, index=index)
    flags, thresholds = correlation_regime_flags(features)
    assert flags.isna().all().all()
    assert not thresholds.usable.any()


def test_policy_changes_honor_lag_and_preserve_structural_missingness(context_sources):
    panel, obs, _ = context_sources
    local, shared = build_policy_divergence(panel, obs)
    levels = obs.loc[obs.dataset.eq("WS_CBPOL")].pivot(index="period", columns="series_id", values="value")
    assert local.loc[(pd.Period("2018-02", "M"), "AAA"), "domestic_policy_change_1m"] == pytest.approx(
        levels.loc["2018-01", "M:AA"]-levels.loc["2017-12", "M:AA"])
    difference = levels["M:AA"]-levels["M:US"]
    assert local.loc[(pd.Period("2018-02", "M"), "AAA"), "rate_diff_change_3m"] == pytest.approx(
        difference.loc["2018-01"]-difference.loc["2017-10"])
    assert shared.loc["2018-02", "us_policy_change_1m"] == pytest.approx(
        levels.loc["2018-01", "M:US"]-levels.loc["2017-12", "M:US"])
    assert local.xs("MISS", level="currency").isna().all().all()


def test_carried_levels_do_not_create_policy_changes_and_future_values_do_not_leak(context_sources):
    _, obs, path = context_sources
    obs = obs.copy()
    obs.loc[obs.series_id.eq("M:AA") & obs.period.eq("2018-03"), "value"] = np.nan
    panel = build_feature_panel(observations=obs, sample_path=path)
    local, shared = build_policy_divergence(panel, obs)
    assert panel.provenance.loc[(pd.Period("2018-04", "M"), "AAA"), "policy_carried_domestic"]
    history = local.xs("AAA", level="currency")
    assert history.loc["2018-04":"2018-05", "domestic_policy_change_1m"].isna().all()
    assert history.loc["2018-04":"2018-07", "rate_diff_change_3m"].isna().all()
    assert np.isfinite(history.loc["2018-08", "rate_diff_change_3m"])
    changed = obs.copy()
    changed.loc[changed.period.ge("2020-01") & changed.dataset.eq("WS_CBPOL"), "value"] += 100
    altered_local, altered_shared = build_policy_divergence(panel, changed)
    pd.testing.assert_frame_equal(local.loc[:"2019-12"], altered_local.loc[:"2019-12"])
    pd.testing.assert_frame_equal(shared.loc[:"2019-12"], altered_shared.loc[:"2019-12"])


def screen_table(context_sources):
    panel, obs, _ = context_sources
    policy, shared = build_policy_divergence(panel, obs)
    pairs = rolling_pair_correlations(panel.fx, windows=(9,), return_views=("raw",))
    correlations, _ = compact_correlations(pairs, list(panel.fx), dict.fromkeys(panel.fx, 0))
    return make_context_screen_table(panel, correlations.join(policy), shared)


def test_screen_uses_same_target_cells_mature_targets_and_validation_only_selection(context_sources):
    result = run_context_screen(screen_table(context_sources), l2_grid=(.01, .1))
    for split, frame in result.predictions.groupby("split"):
        reference = None
        for _, forecasts in frame.groupby("model"):
            observed = forecasts.set_index(["origin", "currency"]).actual.sort_index()
            if reference is None:
                reference = observed
            else:
                pd.testing.assert_series_equal(reference, observed)
        if split == "validation":
            assert frame.origin.max() <= "2022-09"
        else:
            assert frame.origin.max() <= "2026-05"
    assert result.fits.latest_training_endpoint.le("2019-12").all()
    assert result.fits.training_last_origin.le("2019-09").all()
    for row in result.fits.itertuples():
        grid = result.validation_grid.loc[result.validation_grid.model.eq(row.model)]
        best = grid.sort_values(["validation_equal_currency_rmse", "l2"]).iloc[0]
        assert row.selected_l2 == best.l2
    assert np.isfinite(result.predictions.prediction).all()


def test_test_outcomes_do_not_affect_model_selection_or_preprocessing(context_sources):
    table = screen_table(context_sources)
    changed = table.copy()
    changed.loc[changed.index.get_level_values("period") > pd.Period("2022-12", "M"), "target"] += 1000
    a = run_context_screen(table, l2_grid=(.01, .1))
    b = run_context_screen(changed, l2_grid=(.01, .1))
    pd.testing.assert_frame_equal(a.validation_grid, b.validation_grid)
    pd.testing.assert_frame_equal(a.fits, b.fits)
    pd.testing.assert_frame_equal(a.preprocessing, b.preprocessing)
    np.testing.assert_array_equal(a.predictions.prediction, b.predictions.prediction)


@pytest.mark.parametrize("windows", [(1,), (True,), (9, 9)])
def test_invalid_windows_fail_before_computation(context_sources, windows):
    with pytest.raises(ValueError, match="Windows"):
        rolling_pair_correlations(context_sources[0].fx, windows=windows)
