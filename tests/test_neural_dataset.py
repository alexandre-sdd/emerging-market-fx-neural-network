"""Fixed-channel neural exports, training-only transforms, and split maturity."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from fx_forecasting.data.panel import LOCAL_FEATURES, build_feature_panel
from fx_forecasting.data.neural_dataset import (
    build_neural_dataset, eligibility_tables, frozen_groups, load_cluster_dataset,
    save_neural_dataset, select_cluster_dataset,
)


@pytest.fixture
def neural_sources(tmp_path):
    """A small real-schema source panel with launch/gap/structural missingness."""
    dates = pd.period_range("2010-01", "2026-08", freq="M")
    entries = [
        {"area": "AA", "currency": "AAA", "economy": "A", "group": "test", "first_eligible_month": "2010-01",
         "source_series": {"fx": "M:AA:AAA:E", "reer": "M:R:B:AA", "policy": "M:AA", "cpi": "M:AA:628"}},
        {"area": "BB", "currency": "BBB", "economy": "B", "group": "test", "first_eligible_month": "2010-01",
         "source_series": {"fx": "M:BB:BBB:E", "reer": "M:R:B:BB", "policy": None, "cpi": None}},
        {"area": "CC", "currency": "CCC", "economy": "C", "group": "test", "first_eligible_month": "2012-01",
         "source_series": {"fx": "M:CC:CCC:E", "reer": "M:R:B:CC", "policy": "M:CC", "cpi": "M:CC:628"}},
    ]
    cfg = {"currencies": entries, "monthly_window": {"start": str(dates.min()), "end": str(dates.max())},
           "excluded_month_intervals": [],
           "macro_availability": {"reference_lags_months": {"policy": 1, "cpi_inflation": 2, "reer": 2, "usd_neer": 1},
                                  "usd_credit_quarter_lag": 2, "usd_credit_series": "Q:USD:3P:N:A:I:B:USD"},
           "missing_data": {"policy": {"causal_carry_limit_months": 1}, "inflation": {"causal_carry_limit_months": 1}}}
    rows = []
    t = np.arange(len(dates))

    def add_monthly(dataset, source, values):
        rows.extend({"dataset": dataset, "frequency": "M", "series_id": source, "period": str(month), "value": value}
                    for month, value in zip(dates, values))

    rng = np.random.default_rng(41)
    for i, entry in enumerate(entries):
        series = entry["source_series"]
        fx = np.exp(np.cumsum(rng.normal(.002*(i+1), .01, len(dates))))
        if i == 0:
            fx[dates.get_loc("2020-06")] = np.nan
        add_monthly("WS_XRU", series["fx"], fx)
        add_monthly("WS_EER", series["reer"], 100*np.exp(.001*t+.01*np.sin(t/12)))
        if series["policy"]:
            rate = -.5+.0001*t*t+.1*i
            if i == 0:
                rate[dates.get_loc("2018-03")] = np.nan
            add_monthly("WS_CBPOL", series["policy"], rate)
        if series["cpi"]:
            add_monthly("WS_LONG_CPI", series["cpi"], 100*np.exp((.002+.001*i)*t))
    add_monthly("WS_CBPOL", "M:US", .1+.001*t)
    cpi_us = 100*np.exp(.0015*t)
    cpi_us[dates.get_loc("2025-10")] = np.nan
    add_monthly("WS_LONG_CPI", "M:US:628", cpi_us)
    add_monthly("WS_EER", "M:N:B:US", 100*np.exp(.001*t+.01*np.sin(t/12)))
    for i, quarter in enumerate(pd.period_range("2008Q1", "2026Q2", freq="Q")):
        rows.append({"dataset": "WS_GLI", "frequency": "Q", "series_id": cfg["macro_availability"]["usd_credit_series"],
                     "period": str(quarter), "value": 1000*np.exp(.01*i)})
    observations = pd.DataFrame(rows)
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(cfg))
    panel = build_feature_panel(observations=observations, sample_path=path)
    membership = pd.DataFrame({"method": ["adjusted_average"]*3, "k": [2]*3,
                               "currency": ["CCC", "BBB", "AAA"], "cluster": [1, 0, 0],
                               "representative": ["CCC", "AAA", "AAA"]})
    groups = frozen_groups(membership, sorted(panel.fx.columns))
    return panel, groups, observations, path, membership


def test_tensor_axes_windows_groups_and_feature_views(neural_sources):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    a = data.arrays
    n = len(panel.fx)-11
    assert a["X_local"].shape == (n, 12, 3, 7)
    assert a["X_global"].shape == (n, 12, 3)
    assert a["Y"].shape == (n, 3, 3)
    assert data.metadata["currencies"] == ["AAA", "BBB", "CCC"]
    assert np.all(a["input_periods"][:, -1] == a["origins"])
    assert np.all(a["input_periods"] <= a["origins"][:, None])
    batch = select_cluster_dataset(data, cluster=0, horizon=3, feature_view="fx", complete_inputs=False)
    assert batch.currencies == ("AAA", "BBB")
    assert batch.horizons == (3,)
    assert batch.X_local.shape[1:] == (12, 2, 4)
    assert batch.X_global.shape[1:] == (12, 0)
    assert batch.Y.shape[1:] == (2, 1)
    ids = pd.Index(a["origins"]).get_indexer(batch.origins)
    np.testing.assert_array_equal(batch.X_local, a["X_local"][ids, :, :2, :4])
    np.testing.assert_array_equal(batch.Y[..., 0], a["Y"][ids, :2, 1])
    other = select_cluster_dataset(data, cluster=1, split="test", horizon=1,
                                   feature_view="macro", complete_inputs=False)
    assert other.currencies == ("CCC",)
    assert other.X_local.shape[2:] == (1, 7)
    assert other.X_global.shape[2] == 3


def test_future_values_do_not_change_training_inputs_targets_or_preprocessing(neural_sources):
    panel, groups, obs, path, _ = neural_sources
    changed = obs.copy()
    cutoff = pd.Period("2019-12", "M")
    end_months = np.array([pd.Period(row.period, row.frequency).asfreq("M", "end") for row in changed.itertuples()])
    changed.loc[end_months > cutoff, "value"] *= 1000
    future_panel = build_feature_panel(observations=changed, sample_path=path)
    original, revised = build_neural_dataset(panel, groups), build_neural_dataset(future_panel, groups)
    training = original.arrays["split"] == "train"
    for name in ("X_local", "X_local_mask", "X_local_carried", "X_global", "X_global_mask", "Y", "Y_mask",
                 "origins", "input_periods", "target_endpoints"):
        np.testing.assert_array_equal(original.arrays[name][training], revised.arrays[name][training], err_msg=name)
    assert original.metadata["preprocessing"] == revised.metadata["preprocessing"]
    assert original.groups == revised.groups


def test_target_masks_and_zero_values_enforce_split_cutoffs_and_fx_paths(neural_sources):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    a = data.arrays
    for split, cutoff in (("train", "2019-12"), ("validation", "2022-12"), ("test", "2026-08")):
        in_split = a["split"] == split
        after_cutoff = a["target_endpoints"][in_split] > cutoff
        invalid = np.broadcast_to(after_cutoff[:, None, :], a["Y_mask"][in_split].shape)
        assert not a["Y_mask"][in_split][invalid].any()
        assert np.all(a["Y"][in_split][invalid] == 0)
    i = np.flatnonzero(a["origins"] == "2019-11")[0]
    assert a["Y_mask"][i, :, 0].all()
    assert not a["Y_mask"][i, :, 1:].any()
    # May-to-June 2020 crosses AAA's missing FX observation; other currencies remain valid.
    i = np.flatnonzero(a["origins"] == "2020-05")[0]
    assert not a["Y_mask"][i, 0, 0]
    assert a["Y"][i, 0, 0] == 0
    assert a["Y_mask"][i, 1:, 0].all()
    assert np.all(a["Y"][~a["Y_mask"]] == 0)


def test_optional_imputation_mandatory_placeholders_absence_and_carries(neural_sources):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    a = data.arrays
    assert data.metadata["source_absent"]["BBB"] == {"policy": True, "cpi": True}
    j = LOCAL_FEATURES.index("rate_diff")
    stats = data.metadata["preprocessing"]["local"]["rate_diff"]
    training = panel.local.loc[panel.local.index.get_level_values("period") <= pd.Period("2019-12", "M"), "rate_diff"]
    assert stats["median"] == pytest.approx(training.median())
    expected = (stats["median"]-stats["mean"])/stats["scale"]
    assert not a["X_local_mask"][:, :, 1, j].any()
    np.testing.assert_allclose(a["X_local"][:, :, 1, j], expected, rtol=1e-6)
    reer = LOCAL_FEATURES.index("reer_gap")
    missing = ~a["X_local_mask"][..., reer]
    assert missing.any()
    assert np.all(a["X_local"][..., reer][missing] == 0)
    carried = np.flatnonzero(a["origins"] == "2018-04")[0]
    assert a["X_local_carried"][carried, -1, 0, j]
    assert a["X_local_mask"][carried, -1, 0, j]
    assert not a["X_local_carried"][:, :, 1].any()
    us_carry = np.flatnonzero(a["origins"] == "2025-12")[0]
    inflation = LOCAL_FEATURES.index("inflation_diff")
    assert a["X_local_carried"][us_carry, -1, 0, inflation]
    assert np.isfinite(a["X_local"]).all() and np.isfinite(a["X_global"]).all()


def test_npz_and_selection_roundtrip_without_pickle(neural_sources, tmp_path):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    names = save_neural_dataset(data, tmp_path)
    assert set(names) == {"dataset.npz", "dataset_metadata.json", "groups.json"}
    with np.load(tmp_path / "dataset.npz", allow_pickle=False) as archive:
        for key, expected in data.arrays.items():
            assert not archive[key].dtype.hasobject
            np.testing.assert_array_equal(archive[key], expected)
    options = dict(cluster=0, split="validation", horizon=12, feature_view="macro", with_credit=False)
    memory = select_cluster_dataset(data, **options)
    disk = load_cluster_dataset(tmp_path, **options)
    assert memory.currencies == disk.currencies and memory.local_features == disk.local_features
    for name in ("X_local", "X_local_mask", "X_local_carried", "X_global", "X_global_mask", "Y", "Y_mask",
                 "origins", "input_periods", "target_endpoints"):
        np.testing.assert_array_equal(getattr(memory, name), getattr(disk, name))


def test_loader_rejects_persisted_channel_mismatch_and_pickle_objects(neural_sources, tmp_path):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    save_neural_dataset(data, tmp_path)
    broken_groups = deepcopy(groups)
    broken_groups["adjusted_average:k2"]["clusters"][0]["indices"] = [1, 0]
    (tmp_path / "groups.json").write_text(json.dumps(broken_groups))
    with pytest.raises(ValueError, match="channel|indices|currency"):
        load_cluster_dataset(tmp_path, cluster=0)
    save_neural_dataset(data, tmp_path)
    corrupted = dict(data.arrays)
    corrupted["origins"] = corrupted["origins"].astype(object)
    np.savez_compressed(tmp_path / "dataset.npz", **corrupted)
    with pytest.raises(ValueError, match="Object arrays|pickle"):
        load_cluster_dataset(tmp_path, cluster=0)


def test_loader_rejects_unsupported_schema_and_wrong_currency_axis(neural_sources, tmp_path):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    save_neural_dataset(data, tmp_path)
    bad_metadata = deepcopy(data.metadata)
    bad_metadata["schema_version"] = 999
    (tmp_path / "dataset_metadata.json").write_text(json.dumps(bad_metadata))
    with pytest.raises(ValueError, match="schema|version"):
        load_cluster_dataset(tmp_path, cluster=0)
    save_neural_dataset(data, tmp_path)
    corrupted = dict(data.arrays)
    corrupted["Y"] = corrupted["Y"][:, :2]
    np.savez_compressed(tmp_path / "dataset.npz", **corrupted)
    with pytest.raises(ValueError, match="shape|axis|currency"):
        load_cluster_dataset(tmp_path, cluster=0)


def test_eligibility_counts_match_public_loader_and_strict_target_selection(neural_sources):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    details, summary = eligibility_tables(data)
    for split in ("train", "validation", "test"):
        for horizon in (1, 3, 12):
            row = summary.loc[summary.method.eq("adjusted_average") & summary.k.eq(2) & summary.cluster.eq(0)
                              & summary.view.eq("macro") & summary.horizon.eq(horizon) & summary.split.eq(split)].iloc[0]
            batch = select_cluster_dataset(data, cluster=0, split=split, horizon=horizon)
            strict = select_cluster_dataset(data, cluster=0, split=split, horizon=horizon, require_all_targets=True)
            assert len(batch.origins) == row.usable_origins
            assert len(strict.origins) == row.complete_target_origins
            assert len(strict.origins) <= len(batch.origins)
    assert details.origin.nunique() == len(data.arrays["origins"])
    assert not details.loc[details.split.eq("train") & details.origin.eq("2019-12"), "usable_origin"].any()


def test_malformed_group_axis_calendar_and_feature_schema_fail(neural_sources):
    panel, groups, _, _, memberships = neural_sources
    with pytest.raises(ValueError, match="exact currency axis"):
        frozen_groups(pd.concat([memberships, memberships.iloc[:1]]), sorted(panel.fx.columns))
    broken = deepcopy(groups)
    broken["adjusted_average:k2"]["clusters"][0]["indices"] = [1, 0]
    with pytest.raises(ValueError, match="channel indices"):
        build_neural_dataset(panel, broken)
    missing_currency = deepcopy(groups)
    missing_currency["adjusted_average:k2"]["clusters"].pop()
    with pytest.raises(ValueError, match="exact currency axis"):
        build_neural_dataset(panel, missing_currency)
    with pytest.raises(ValueError, match="ascending monthly calendar"):
        build_neural_dataset(replace(panel, fx=panel.fx.drop(panel.fx.index[20])), groups)
    with pytest.raises(KeyError, match="fx_change_1m"):
        build_neural_dataset(replace(panel, local=panel.local.drop(columns="fx_change_1m")), groups)
    absent = panel.local.copy()
    absent["rate_diff"] = np.nan
    with pytest.raises(ValueError, match="entirely absent in training"):
        build_neural_dataset(replace(panel, local=absent), groups)


@pytest.mark.parametrize("options", [{"lookback": 0}, {"lookback": True}, {"horizons": (1, 1)},
                                     {"horizons": (0,)}, {"validation_end": "2019-12"}])
def test_invalid_dataset_parameters(neural_sources, options):
    panel, groups, _, _, _ = neural_sources
    with pytest.raises(ValueError):
        build_neural_dataset(panel, groups, **options)


@pytest.mark.parametrize("options", [{"split": "unknown"}, {"cluster": 99}, {"horizon": 2}, {"feature_view": "unknown"}])
def test_invalid_public_selection(neural_sources, options):
    panel, groups, _, _, _ = neural_sources
    data = build_neural_dataset(panel, groups)
    with pytest.raises(ValueError):
        select_cluster_dataset(data, **options)
