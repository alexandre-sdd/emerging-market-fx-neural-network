"""Checks for frozen synchronized groups and their comparison protocol."""

import numpy as np
import pandas as pd
import pytest

from fx_forecasting.models.currency_clustering import _block_indices, _summarize, compare_clusterings


def example_fx():
    rng = np.random.default_rng(8)
    dates = pd.period_range("2009-01", "2022-12", freq="M")
    common = rng.normal(0, .01, (len(dates), 1))
    regional = np.repeat(rng.normal(0, .012, (len(dates), 2)), 4, axis=1)
    changes = common + regional + rng.normal(0, .004, (len(dates), 8))
    return pd.DataFrame(np.exp(np.cumsum(changes, axis=0)), index=dates,
                        columns=[f"C{i}" for i in range(8)])


def compare(fx, **kwargs):
    return compare_clusterings(fx, fit_start="2011-01", fit_end="2014-12",
                                k_values=(2, 3), bootstrap_reps=2, include_dtw=False, **kwargs)


def test_axes_distances_memberships_and_representatives_are_consistent():
    result = compare(example_fx())
    assert len(result.scores) == 7*2
    for (method, k), rows in result.memberships.groupby(["method", "k"]):
        assert sorted(rows.currency) == [f"C{i}" for i in range(8)]
        assert rows.cluster.nunique() == k
        for _, group in rows.groupby("cluster"):
            assert group.representative.nunique() == 1
            assert group.representative.iloc[0] in set(group.currency)
        ordered = rows.sort_values("currency")
        np.testing.assert_array_equal(ordered.channel_index, np.arange(8))
        assert rows.groupby("cluster").currency.min().is_monotonic_increasing
    for distance in result.distances.values():
        np.testing.assert_allclose(distance, distance.T)
        np.testing.assert_allclose(np.diag(distance), 0, atol=1e-12)
        assert np.isfinite(distance.to_numpy()).all()


def test_future_values_and_currency_column_order_cannot_change_training_groups():
    fx = example_fx()
    a = compare(fx)
    changed = fx.copy()
    changed.loc["2015-01":] = np.nan
    b = compare(changed[changed.columns[::-1]])
    pd.testing.assert_frame_equal(a.memberships, b.memberships)
    pd.testing.assert_frame_equal(a.scores, b.scores)
    pd.testing.assert_frame_equal(a.stability, b.stability)


def test_bootstraps_are_reproducible_and_have_explicit_counts():
    a, b = compare(example_fx()), compare(example_fx())
    pd.testing.assert_frame_equal(a.stability, b.stability)
    static = a.scores.loc[~a.scores.method.str.startswith("r2k")]
    assert static.bootstrap_reps.eq(2).all()
    assert a.scores.loc[a.scores.method.str.startswith("r2k"), "bootstrap_reps"].eq(0).all()
    assert a.stability.loc[a.stability.kind.eq("bootstrap"), "block_length"].eq(12).all()


def test_missing_common_month_is_rejected_instead_of_filled_or_dropped():
    fx = example_fx()
    fx.loc["2012-03", "C0"] = np.nan
    with pytest.raises(ValueError, match="complete common"):
        compare(fx)


def test_constant_history_is_rejected_explicitly():
    fx = example_fx()
    fx["C0"] = 1.0
    with pytest.raises(ValueError, match="Constant"):
        compare(fx)


def test_reference_moments_disclose_warmup():
    fx = example_fx()
    fx.loc["2009-01":"2010-07", "C0"] = np.nan
    result = compare(fx)
    reference = result.scores.loc[result.scores.method.str.startswith("r2k")]
    assert reference.fit_start.eq("2011-08").all()
    assert reference.n_months.eq(41).all()


def test_zero_bootstraps_still_produces_explicit_empty_diagnostics():
    result = compare_clusterings(example_fx(), fit_start="2011-01", fit_end="2014-12",
                                 k_values=(2,), bootstrap_reps=0, include_dtw=False)
    assert result.scores.bootstrap_reps.eq(0).all()
    assert result.scores.bootstrap_mean_ari.isna().all()


def test_singleton_is_reported_and_represented_without_hiding_it():
    values = np.arange(8, dtype=float)
    distance = np.abs(values[:, None] - values[None, :])
    score, members = _summarize("test", 2, np.array([0]*7+[1]), distance,
                                [f"C{i}" for i in range(8)], np.eye(8),
                                "2011-01", "2014-12", 48, "test")
    assert score["min_cluster_size"] == 1
    assert score["n_singletons"] == 1
    singleton = next(row for row in members if row["currency"] == "C7")
    assert singleton["representative"] == "C7"
    assert singleton["silhouette"] == 0


def test_block_bootstrap_preserves_observed_month_order_within_each_block():
    indices = _block_indices(np.random.default_rng(42), 31)
    assert len(indices) == 31
    assert indices.min() >= 0 and indices.max() < 31
    for first in range(0, len(indices), 12):
        np.testing.assert_array_equal(np.diff(indices[first:first+12]) % 31,
                                      np.ones(len(indices[first:first+12])-1, dtype=int))


def test_dtw_optional_path_fits_the_same_frozen_axis():
    pytest.importorskip("tslearn")
    result = compare_clusterings(example_fx(), fit_start="2011-01", fit_end="2011-12",
                                 k_values=(2,), bootstrap_reps=0, include_dtw=True)
    rows = result.memberships.loc[result.memberships.method.eq("dtw")]
    assert len(rows) == 8
    assert rows.cluster.nunique() == 2
    assert result.scores.loc[result.scores.method.eq("dtw"), "n_months"].iloc[0] == 12
