"""Regression checks for training-only reference diagnostics, not forecasting."""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_samples

from fx_forecasting.evaluation.clustering_theory import (
    CORRELATION_METHODS, _best_jaccard, _shift_residuals, analyze_structure,
)
from fx_forecasting.models.currency_clustering import _geometry


def sample(ks=(2, 3)):
    rng = np.random.default_rng(318)
    common = rng.normal(size=(108, 1))
    values = common*rng.uniform(.2, 1.5, size=(1, 8))+rng.normal(size=(108, 8))
    history = pd.DataFrame(values, index=pd.period_range("2011-01", "2019-12", freq="M"),
                           columns=[f"C{i}" for i in range(8)])
    rows = []
    for method in CORRELATION_METHODS:
        _, distance, tree = _geometry(history, method)
        for k in ks:
            labels = cut_tree(tree, n_clusters=[k]).ravel()
            silhouettes = silhouette_samples(distance, labels, metric="precomputed")
            for i, currency in enumerate(history.columns):
                group = np.flatnonzero(labels == labels[i])
                medoid = group[np.argmin(distance[np.ix_(group, group)].mean(axis=1))]
                rows.append({"method": method, "k": k, "currency": currency, "cluster": labels[i],
                             "representative": history.columns[medoid], "silhouette": silhouettes[i]})
    return history, pd.DataFrame(rows)


def small_run(history, members, **kwargs):
    return analyze_structure(history, members, bootstrap_reps=3, reference_reps=3, **kwargs)


def test_future_values_and_column_permutations_do_not_change_training_diagnostics():
    history, members = sample()
    tables, matrices, metadata = small_run(history, members)
    future = pd.DataFrame(1e8, index=pd.period_range("2020-01", periods=18, freq="M"), columns=history.columns)
    changed = pd.concat([history, future]).iloc[:, ::-1]
    new_tables, new_matrices, new_metadata = small_run(changed, members.sample(frac=1, random_state=4))
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], new_tables[name])
    for name in matrices:
        np.testing.assert_array_equal(matrices[name], new_matrices[name])
    assert metadata == new_metadata


def test_cluster_recovery_is_label_invariant_and_uses_best_set_overlap():
    reference = np.array([True, True, False, False])
    assert _best_jaccard(reference, np.array([10, 10, 5, 5])) == 1
    assert _best_jaccard(reference, np.array([5, 5, 10, 10])) == 1
    assert _best_jaccard(reference, np.array([5, 10, 5, 10])) == pytest.approx(1/3)


def test_reference_shifts_preserve_residual_marginals_and_serial_spectrum():
    residuals = np.random.default_rng(12).normal(size=(108, 8))
    shifted = _shift_residuals(residuals, np.random.default_rng(15))
    np.testing.assert_array_equal(np.sort(residuals, axis=0), np.sort(shifted, axis=0))
    np.testing.assert_allclose(np.abs(np.fft.rfft(residuals, axis=0)),
                               np.abs(np.fft.rfft(shifted, axis=0)), atol=1e-12)
    assert not np.allclose(np.corrcoef(residuals.T), np.corrcoef(shifted.T))


def test_coassignment_and_factor_references_are_well_defined():
    history, members = sample()
    tables, matrices, meta = small_run(history, members)
    for matrix in matrices.values():
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), np.ones(8))
        assert ((matrix >= 0) & (matrix <= 1)).all()
    for _, frame in tables["factor_spectrum"].groupby("representation"):
        assert frame.variance_share.sum() == pytest.approx(1)
        assert frame.iloc[-1].cumulative_share == pytest.approx(1)
    ref = tables["factor_reference"]
    assert len(ref) == 16
    assert (ref.reference_p05 <= ref.reference_median).all()
    assert (ref.reference_median <= ref.reference_p95).all()
    assert (ref.groupby(["method", "k"]).observed_silhouette.nunique() == 1).all()
    assert meta["fit_end"] == "2019-12"


def test_singleton_coherence_is_missing_not_fabricated_zero():
    history, members = sample(ks=(7,))
    tables, _, _ = small_run(history, members)
    singles = tables["group_quality"].query("n_currencies == 1")
    assert not singles.empty
    assert singles.within_raw_correlation.isna().all()


def test_explicitly_skipped_bootstrap_has_no_fabricated_recovery():
    history, members = sample()
    tables, matrices, meta = analyze_structure(history, members, bootstrap_reps=0, reference_reps=1)
    assert matrices == {}
    assert tables["group_recovery"].mean_jaccard.isna().all()
    assert meta["bootstrap_reps"] == 0


def test_inconsistent_memberships_are_rejected():
    history, members = sample()
    with pytest.raises(ValueError, match="cover"):
        small_run(history, members.loc[members.currency.ne("C0")])
