"""Check that the grouping audit answers its question without future labels."""

import numpy as np
import pandas as pd
import pytest

from fx_forecasting.data.rolling_features import make_rolling_features
from fx_forecasting.evaluation.grouping_screen import make_screen_panel, run_grouping_screen
from fx_forecasting.evaluation.r2k_audit import produced_output_hashes, stationary_counterexample
from fx_forecasting.evaluation.rolling_clustering import run_cross_sections


def fixture_panel():
    rng = np.random.default_rng(21)
    dates = pd.period_range("2000-01", "2006-12", freq="M")
    fx = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .03, (len(dates), 4)), axis=0)),
                      index=dates, columns=list("ABCD"))
    panel = make_screen_panel(fx, horizons=(1, 3))
    labels = pd.Series(0, index=panel.index)
    metadata = pd.DataFrame({"group": ["test"]*4}, index=list("ABCD"))
    return fx, panel, labels, metadata


def test_target_requires_every_intervening_fx_observation():
    fx, _, _, _ = fixture_panel()
    fx.loc["2003-02", "A"] = np.nan
    panel = make_screen_panel(fx, horizons=(3,))
    assert np.isnan(panel.loc[(pd.Period("2003-01", "M"), "A"), "target_3m"])
    assert np.isfinite(panel.loc[(pd.Period("2002-10", "M"), "A"), "target_3m"])


def test_one_group_is_exactly_the_pooled_model_and_endpoints_are_mature():
    _, panel, labels, metadata = fixture_panel()
    result = run_grouping_screen(panel, {"one_group": labels}, metadata,
                                 evaluation_start="2004-01", horizons=(1, 3),
                                 minimum_rows=20, minimum_months=12)
    predictions = result.forecasts.pivot(index=["origin", "currency", "horizon"], columns="model", values="prediction")
    np.testing.assert_allclose(predictions.one_group, predictions.pooled_ridge, atol=1e-12)
    assert (result.fits.latest_training_endpoint <= result.fits.origin).all()
    counts = result.forecasts.groupby(["model", "horizon"]).size().unstack()
    assert counts.eq(counts.iloc[0]).all().all()


def test_unrealized_target_perturbations_do_not_change_predictions():
    _, panel, labels, metadata = fixture_panel()
    changed = panel.copy()
    period = panel.index.get_level_values("period")
    cutoff = pd.Period("2004-03", "M")
    changed.loc[period+3 > cutoff, "target_3m"] += 1e5
    options = dict(evaluation_start="2004-01", horizons=(3,), minimum_rows=20, minimum_months=12)
    a = run_grouping_screen(panel, {"group": labels}, metadata, **options)
    b = run_grouping_screen(changed, {"group": labels}, metadata, **options)
    left = a.forecasts.loc[a.forecasts.origin.le(str(cutoff)), "prediction"]
    right = b.forecasts.loc[b.forecasts.origin.le(str(cutoff)), "prediction"]
    np.testing.assert_allclose(left, right)


def test_sparse_group_falls_back_to_pooled_model():
    _, panel, labels, metadata = fixture_panel()
    labels.loc[panel.index.get_level_values("period") >= pd.Period("2004-01", "M")] = 1
    result = run_grouping_screen(panel, {"group": labels}, metadata,
                                 evaluation_start="2004-01", horizons=(1,),
                                 minimum_rows=20, minimum_months=12)
    rows = result.forecasts.loc[result.forecasts.origin.eq("2004-01")]
    predictions = rows.pivot(index="currency", columns="model", values="prediction")
    np.testing.assert_allclose(predictions.group, predictions.pooled_ridge)
    assert rows.loc[rows.model.eq("group"), "fallback"].all()


def test_fixed_scaling_uses_only_initial_cross_section_and_is_prefix_invariant():
    fx, _, _, _ = fixture_panel()
    features = make_rolling_features(fx).dropna()
    first = features.index.get_level_values("period").min()
    result = run_cross_sections(features, n_clusters=2, windows=(1,), extra_baselines=True,
                                scaling="initial_cross_section")
    for feature, table in result.normalization.groupby("feature"):
        assert table["mean"].nunique() == 1
        assert table["scale"].nunique() == 1
        assert table["mean"].iloc[0] == pytest.approx(features.xs(first)[feature].mean())
    shortened = features.loc[features.index.get_level_values("period") < pd.Period("2004-01", "M")]
    prefix = run_cross_sections(shortened, n_clusters=2, windows=(1,), extra_baselines=True,
                                scaling="initial_cross_section")
    pd.testing.assert_frame_equal(result.labels.loc[result.labels.period.lt("2004-01")].reset_index(drop=True), prefix.labels)


def test_paper_stationary_claim_has_an_explicit_counterexample():
    result = stationary_counterexample()
    assert result["identity_switches"] > 0
    assert result["partition_ari"] < 1
    assert result["inertia_second"] < result["inertia_first"]


def test_snapshot_uses_recorded_fixed_scaling_coordinates():
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from fx_forecasting.evaluation.rolling_plots import plot_snapshot

    dates = pd.period_range("2000-01", periods=4, freq="M")
    initial = pd.DataFrame({"x": [0., 1., 4., 5.], "y": [0., 2., 4., 6.]}, index=list("ABCD"))
    features = pd.concat({month: initial + np.array([3., .7])*i
                          for i, month in enumerate(dates)}, names=["period", "currency"])
    result = run_cross_sections(features, n_clusters=2, windows=(1,), baselines=False,
                                scaling="initial_cross_section")
    fig = plot_snapshot(result, features, model="r2k_w1", window=1, period=str(dates[-1]))
    try:
        scatter = [collection for collection in fig.axes[0].collections if isinstance(collection, PathCollection)]
        plotted = np.concatenate([item.get_offsets() for item in scatter[::2]])
        expected = ((features.xs(dates[-1]) - initial.mean()) / initial.std(ddof=0)).to_numpy()
        order = lambda values: values[np.lexsort((values[:, 1], values[:, 0]))]
        np.testing.assert_allclose(order(plotted), order(expected))
    finally:
        plt.close(fig)


def test_audit_provenance_excludes_stale_outputs(tmp_path):
    import hashlib

    (tmp_path / "current.csv").write_text("fresh\n")
    (tmp_path / "stale.json").write_text('{"old": true}')
    (tmp_path / "older.parquet").write_bytes(b"old")
    output = produced_output_hashes(tmp_path, ["current.csv"])
    assert output == {"current.csv": hashlib.sha256(b"fresh\n").hexdigest()}
