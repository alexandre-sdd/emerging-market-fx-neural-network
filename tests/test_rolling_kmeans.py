"""Algorithm, chronology, and gap handling checks for the R2K-Means experiment."""

import numpy as np
import pandas as pd
import pytest

from fx_forecasting.data.rolling_features import (
    load_selected_fx, make_rolling_features, normalize_cross_section,
)
from fx_forecasting.evaluation.rolling_clustering import (
    align_centers, partition_metrics, run_cross_sections,
)
from fx_forecasting.models.rolling_kmeans import RobustRollingKMeans, select_initial_centers


def cross(right=10.0):
    return pd.DataFrame({"x": [0.0, 0.2, right, right + 0.2], "y": [0.0] * 4},
                        index=["a", "b", "c", "d"])


def test_small_clusters_choose_first_and_claim_entire_duplicate_location():
    seeds, order = select_initial_centers(
        np.array([[0.0], [0.0], [4.0], [10.0]]), np.array([[0.1], [0.2]]), np.array([4, 1]),
    )
    np.testing.assert_array_equal(order, [1, 0])
    np.testing.assert_array_equal(seeds[:, 0], [4, 0])


def test_equal_size_and_distance_ties_are_deterministic():
    seeds, order = select_initial_centers(
        np.array([[-1.0], [1.0], [3.0]]), np.array([[0.0], [0.0]]), np.array([2, 2]),
    )
    np.testing.assert_array_equal(order, [0, 1])
    np.testing.assert_array_equal(seeds[:, 0], [-1, 1])


def test_static_separated_clouds_keep_centers_and_ids():
    model = RobustRollingKMeans(2, 3)
    first = model.update(cross())
    for _ in range(5):
        result = model.update(cross().iloc[::-1])
        pd.testing.assert_series_equal(result.labels.sort_index(), first.labels.sort_index())
        np.testing.assert_allclose(result.centers, first.centers)
    assert model.centroid_history_.shape == (3, 2, 2)


def test_rolling_rule_uses_nearest_historical_centroid_not_mean():
    rolling, current = RobustRollingKMeans(2, 2), RobustRollingKMeans(2, 1)
    for frame in (cross(10), cross(20)):
        old = rolling.update(frame)
        now = current.update(frame)
    point = pd.DataFrame({"x": [7.0], "y": [0.0]}, index=["query"])
    right_id = int(old.labels.loc["c"])
    assert rolling.predict(point).iloc[0] == right_id  # historical center near 10
    assert current.predict(point).iloc[0] == now.labels.loc["a"]  # current center near 0
    mean_centers = rolling.centroid_history_.mean(axis=0)
    assert np.argmin(((mean_centers - [7, 0]) ** 2).sum(axis=1)) != right_id


def test_window_one_labels_equal_current_lloyd_labels():
    model = RobustRollingKMeans(2, 1)
    for right in (10, 20, 8):
        result = model.update(cross(right))
        np.testing.assert_array_equal(result.labels, result.lloyd_labels)


def test_classification_window_does_not_change_fitted_centroid_path():
    current, rolling = RobustRollingKMeans(2, 1), RobustRollingKMeans(2, 6)
    for right in (10, 20, 8, 17, 5):
        a, b = current.update(cross(right)), rolling.update(cross(right))
        np.testing.assert_allclose(a.centers, b.centers)
        np.testing.assert_array_equal(a.counts, b.counts)
        assert a.inertia == pytest.approx(b.inertia)


def test_invalid_update_preserves_model_and_previous_result():
    model = RobustRollingKMeans(2, 2)
    first = model.update(cross())
    before = model.centroid_history_
    for bad in (cross().assign(x=np.nan), cross().assign(x=0.0)):
        with pytest.raises(ValueError):
            model.update(bad)
        np.testing.assert_array_equal(before, model.centroid_history_)
        assert model.step_count_ == 1
    model.update(cross(20))
    np.testing.assert_allclose(np.sort(first.centers[:, 0]), [0.1, 10.1])


def test_asset_order_and_new_assets_do_not_rename_clusters():
    left, right = RobustRollingKMeans(2, 3), RobustRollingKMeans(2, 3)
    for values in (cross(), cross(12), cross(20)):
        a, b = left.update(values), right.update(values.iloc[::-1])
        pd.testing.assert_series_equal(a.labels.sort_index(), b.labels.sort_index())
        np.testing.assert_allclose(a.centers, b.centers)
    expanded = pd.concat([cross(20), pd.DataFrame({"x": [20.3], "y": [0]}, index=["entrant"])])
    result = left.update(expanded)
    assert result.labels.loc["entrant"] == result.labels.loc["c"]


def test_schema_and_iteration_limit_fail_without_commit():
    model = RobustRollingKMeans(2, 2)
    model.update(cross())
    with pytest.raises(ValueError, match="Feature"):
        model.update(cross()[["y", "x"]])
    with pytest.raises(ValueError, match="unique"):
        model.update(cross().rename(index={"b": "a"}))
    model = RobustRollingKMeans(3, 2, max_iter=1, n_init=1, random_state=0)
    unfinished = pd.DataFrame(np.random.default_rng(0).normal(size=(12, 2)),
                              index=[f"a{i:02}" for i in range(12)], columns=["x", "y"])
    with pytest.raises(RuntimeError, match="iteration limit"):
        model.update(unfinished)
    assert model.step_count_ == 0


def test_converged_last_allowed_iteration_is_accepted():
    points = pd.DataFrame({"x": [0.0, 10.0]}, index=["left", "right"])
    model = RobustRollingKMeans(2, 1, max_iter=1, n_init=1)
    result = model.update(points)
    assert result.n_iter == 1
    assert result.inertia == pytest.approx(0)
    assert result.labels.nunique() == 2
    assert model.step_count_ == 1


def test_unfinished_later_update_preserves_fitted_history():
    def frame(seed):
        return pd.DataFrame(np.random.default_rng(seed).normal(size=(12, 2)),
                            index=[f"a{i:02}" for i in range(12)], columns=["x", "y"])

    model = RobustRollingKMeans(3, 2, n_init=1, random_state=0)
    model.update(frame(0))
    previous = model.centroid_history_
    model.max_iter = 1
    with pytest.raises(RuntimeError, match="iteration limit"):
        model.update(frame(8))
    assert model.step_count_ == 1
    np.testing.assert_array_equal(previous, model.centroid_history_)


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"n_clusters": 0}, {"window": 1.5}, {"n_init": True}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RobustRollingKMeans(**kwargs)


def test_gap_resets_return_history_and_requires_full_warmup():
    index = pd.period_range("2000-01", periods=9, freq="M")
    fx = pd.DataFrame({"AAA": np.exp(np.arange(9) * 0.01)}, index=index)
    fx.iloc[3] = np.nan
    result = make_rolling_features(fx, lookback=2).xs("AAA", level="currency")
    assert result.iloc[2].notna().all()
    assert result.iloc[3:6].isna().all().all()  # gap, next return, insufficient warmup
    assert result.iloc[6].notna().all()
    assert result.iloc[6].annual_log_change == pytest.approx(12)


def test_feature_and_experiment_prefix_are_independent_of_future_values():
    dates = pd.period_range("2000-01", periods=22, freq="M")
    rng = np.random.default_rng(7)
    fx = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .02, (22, 8)), axis=0)),
                      index=dates, columns=list("ABCDEFGH"))
    original = make_rolling_features(fx, 3)
    changed_fx = fx.copy()
    changed_fx.iloc[15:] *= 50
    changed = make_rolling_features(changed_fx, 3)
    prefix = original.index.get_level_values("period") < dates[15]
    pd.testing.assert_frame_equal(original.loc[prefix], changed.loc[prefix])
    eligible = original.loc[original.index.get_level_values("period") >= dates[3]]
    a = run_cross_sections(eligible, n_clusters=2, windows=(1, 3))
    b = run_cross_sections(eligible.loc[eligible.index.get_level_values("period") < dates[15]],
                           n_clusters=2, windows=(1, 3))
    pd.testing.assert_frame_equal(a.labels.loc[a.labels.period.lt(str(dates[15]))].reset_index(drop=True), b.labels)
    pd.testing.assert_frame_equal(a.centroids.loc[a.centroids.period.lt(str(dates[15]))].reset_index(drop=True), b.centroids)


def test_normalization_is_contemporaneous_and_constant_features_are_flagged():
    frame = pd.DataFrame({"x": [1.0, 3.0], "constant": [4.0, 4.0]})
    scaled, stats = normalize_cross_section(frame)
    np.testing.assert_array_equal(scaled.x, [-1, 1])
    assert scaled.constant.eq(0).all()
    assert stats.loc["constant", "constant"]


def test_permutation_invariant_metric_distinguishes_pure_relabeling():
    a = pd.Series([0, 0, 1, 1], index=list("abcd"))
    b = pd.Series([1, 1, 0, 0], index=list("abcd"))
    metrics = partition_metrics(b, a)
    assert metrics["switch_rate"] == 1
    assert metrics["ari"] == 1
    centers, mapping = align_centers(np.array([[10.0], [0.0]]), np.array([[0.0], [10.0]]))
    np.testing.assert_array_equal(centers[:, 0], [0, 10])
    np.testing.assert_array_equal(mapping, [1, 0])


def test_switch_denominator_excludes_entrants_and_exits():
    before = pd.Series([0, 1, 1], index=list("abc"))
    after = pd.Series([1, 1, 0], index=list("bcd"))
    metrics = partition_metrics(after, before)
    assert metrics["n_common"] == 2
    assert metrics["switch_rate"] == 0


def test_missing_calendar_months_are_rejected():
    fx = pd.DataFrame({"A": [1, 2, 3]}, index=pd.PeriodIndex(["2000-01", "2000-03", "2000-04"], freq="M"))
    with pytest.raises(ValueError, match="calendar month"):
        make_rolling_features(fx, 2)


def test_configured_launches_and_exclusion_boundaries(tmp_path):
    import json
    currencies = [
        {"area": "X", "currency": "AAA", "first_eligible_month": "2000-03", "source_series": {"fx": "M:X:AAA:E"}},
        {"area": "Y", "currency": "BBB", "first_eligible_month": "2000-01", "source_series": {"fx": "M:Y:BBB:E"}},
    ]
    cfg = {"currencies": currencies, "monthly_window": {"start": "2000-01", "end": "2000-06"},
           "excluded_month_intervals": [{"area": "Y", "start": "2000-03", "end": "2000-04"}]}
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(cfg))
    rows = [{"dataset": "WS_XRU", "frequency": "M", "series_id": c["source_series"]["fx"],
             "period": str(month), "value": 1.0}
            for c in currencies for month in pd.period_range("2000-01", "2000-06", freq="M")]
    pd.DataFrame(rows).to_parquet(tmp_path / "observations.parquet")
    fx, _, _ = load_selected_fx(tmp_path, sample)
    assert fx.loc[:"2000-02", "AAA"].isna().all()
    assert fx.loc["2000-03", "AAA"] == 1
    assert fx.loc["2000-03":"2000-04", "BBB"].isna().all()
    assert fx.loc["2000-02", "BBB"] == fx.loc["2000-05", "BBB"] == 1


def test_bis_runner_exports_replayable_tables_and_verified_hashes(tmp_path):
    import hashlib
    import json
    from fx_forecasting.evaluation.rolling_clustering import run_bis_experiment

    rng = np.random.default_rng(12)
    dates = pd.period_range("2000-01", "2002-12", freq="M")
    values = np.exp(np.cumsum(rng.normal(0, .02, (len(dates), 6)), axis=0))
    currencies = [{"area": f"X{i}", "currency": f"CC{i}", "first_eligible_month": "2000-01",
                   "group": "test", "source_series": {"fx": f"M:X{i}:CC{i}:E"}} for i in range(6)]
    cfg = {"currencies": currencies, "monthly_window": {"start": "2000-01", "end": "2002-12"},
           "excluded_month_intervals": []}
    source = tmp_path / "input"
    source.mkdir()
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(cfg))
    rows = [{"dataset": "WS_XRU", "frequency": "M", "series_id": c["source_series"]["fx"],
             "period": str(month), "value": values[t, i]}
            for i, c in enumerate(currencies) for t, month in enumerate(dates)]
    pd.DataFrame(rows).to_parquet(source / "observations.parquet")
    before = (source / "observations.parquet").read_bytes()
    output = tmp_path / "output"
    result, summary, manifest = run_bis_experiment(
        ingested_dir=source, sample_path=sample, output_dir=output,
        start="2001-01", evaluation_start="2002-01", n_clusters=2, windows=(1, 3),
    )
    pd.testing.assert_frame_equal(pd.read_parquet(output / "labels.parquet"), result.labels)
    assert len(summary) == 4
    assert summary.n_months.eq(12).all()
    assert result.coverage.n_eligible.eq(6).all()
    assert (source / "observations.parquet").read_bytes() == before
    for name, details in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == details["sha256"]
    disk_manifest = json.loads((output / "run_manifest.json").read_text())
    assert disk_manifest == manifest
