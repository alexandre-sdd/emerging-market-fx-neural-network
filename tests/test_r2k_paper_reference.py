"""Independent NumPy replay of Algorithm 2, without production seed/label helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from fx_forecasting.models.rolling_kmeans import RobustRollingKMeans


def reference_update(points, old_centers, old_counts, bank, window):
    """Literal steps 12–19; numerical Lloyd iteration is independently written."""
    remaining = list(range(len(points)))
    starts = np.empty_like(old_centers)
    for label in sorted(range(len(old_centers)), key=lambda c: (old_counts[c], c)):
        chosen = min(remaining, key=lambda j: (sum((points[j]-old_centers[label])**2), j))
        starts[label] = points[chosen]
        remaining = [j for j in remaining if not np.array_equal(points[j], points[chosen])]
    centers = starts.copy()
    for _ in range(300):
        labels = np.array([min(range(len(centers)), key=lambda c: sum((x-centers[c])**2)) for x in points])
        assert len(set(labels)) == len(centers), 'Oracle fixture has an empty cluster'
        new = np.array([points[labels == c].mean(axis=0) for c in range(len(centers))])
        if np.allclose(new, centers, atol=1e-12, rtol=0):
            centers = new
            break
        centers = new
    else:
        raise AssertionError('Oracle did not converge')
    history = (bank+[centers])[-window:]
    # Exhaustively enumerate (age, identity), retaining the identity of the nearest center.
    rolling = np.array([min(((sum((x-center)**2), -age, label)
                             for age, state in enumerate(history)
                             for label, center in enumerate(state)))[2] for x in points])
    return starts, centers, np.bincount(labels, minlength=len(centers)), labels, rolling, history


@pytest.mark.parametrize('window', [1, 6, 12])
@pytest.mark.parametrize('k', [2, 3, 4])
def test_algorithm_two_matches_independent_numpy_replay(k, window):
    rng = np.random.default_rng(2026+k)
    # Persistent overlapping clouds move, stretch and exchange boundary observations.
    frames = [pd.DataFrame(rng.normal(size=(24, 2)) +
                          np.column_stack((np.linspace(-2, 2, 24), np.sin(np.arange(24)+t/3))),
                          index=[f'asset{i:02}' for i in range(24)], columns=['return', 'risk'])
              for t in range(16)]
    model = RobustRollingKMeans(k, window, n_init=10)
    initial = model.update(frames[0])
    centers, counts, bank = initial.centers, initial.counts, [initial.centers]
    for frame in frames[1:]:
        starts, centers, counts, labels, rolling, bank = reference_update(
            frame.to_numpy(), centers, counts, bank, window)
        actual = model.update(frame)
        np.testing.assert_allclose(actual.initial_centers, starts, atol=1e-12)
        np.testing.assert_allclose(actual.centers, centers, atol=1e-12)
        np.testing.assert_array_equal(actual.counts, counts)
        np.testing.assert_array_equal(actual.lloyd_labels, labels)
        np.testing.assert_array_equal(actual.labels, rolling)


def test_unchanged_data_claim_is_not_a_universal_algorithm_guarantee():
    frame = pd.DataFrame(np.random.default_rng(2).normal(size=(12, 2)),
                         index=[f'a{i:02}' for i in range(12)], columns=['x', 'y'])
    model = RobustRollingKMeans(3, 1, random_state=2, n_init=1)
    first, second = model.update(frame), model.update(frame)
    assert adjusted_rand_score(first.labels, second.labels) < 1
    assert second.inertia < first.inertia
