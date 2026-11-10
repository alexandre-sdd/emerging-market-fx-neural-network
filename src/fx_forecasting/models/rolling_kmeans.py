"""Robust Rolling K-Means, Algorithm 2 of Holmes et al. (2024).

Each update consumes one cross section (assets x features), not a matrix whose
columns are dates. Scaling is the caller's responsibility. Cluster identities
persist; sorting by cluster size changes seed-selection priority, never IDs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ClusteringStep:
    """An update's diagnostics; labels follow the input asset order.

    ``labels`` use the rolling bank; ``lloyd_labels`` use current centroids only.
    Counts and inertia refer to the latter, before rolling classification.
    """

    labels: pd.Series
    lloyd_labels: pd.Series
    centers: np.ndarray
    counts: np.ndarray
    initial_centers: np.ndarray | None
    selection_order: np.ndarray
    inertia: float
    n_iter: int
    step: int


def validate_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and order assets for deterministic distance ties."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a DataFrame indexed by unique string asset IDs")
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Cross section must be nonempty with unique IDs and features")
    if not all(isinstance(x, str) for x in (*frame.index, *frame.columns)):
        raise ValueError("Asset IDs and feature names must be strings")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cross section must contain finite values; do not impute in the model")
    return frame.astype(float).sort_index()


def select_initial_centers(
    values: np.ndarray, previous: np.ndarray, counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap small clusters first to distinct *locations* in the current data.

    Input rows must already be sorted by asset ID. Equal-size clusters choose
    in ID order, equal-distance points in asset order. All duplicates at a
    claimed location are excluded, rather than just the selected row.
    """
    k = len(previous)
    if len(np.unique(values, axis=0)) < k:
        raise ValueError(f"Need at least {k} distinct feature locations")
    order = np.argsort(counts, kind="stable")
    available = np.ones(len(values), dtype=bool)
    seeds = np.empty_like(previous)
    for cluster in order:
        distances = np.sum((values - previous[cluster]) ** 2, axis=1)
        chosen = np.argmin(np.where(available, distances, np.inf))
        seeds[cluster] = values[chosen]
        available &= ~np.all(values == values[chosen], axis=1)
    return seeds, order


def is_lloyd_fixed_point(values: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> bool:
    """Check the fitted assignments and means, including a last-iteration fit.

    Reaching ``max_iter`` does not itself prove nonconvergence: the last allowed
    assignment/update can already be a Lloyd fixed point. Tied nearest centers
    are accepted when the returned assignments also reproduce their means.
    """
    if len(np.unique(labels)) != len(centers):
        return False
    means = np.array([values[labels == cluster].mean(axis=0) for cluster in range(len(centers))])
    if not np.allclose(centers, means, rtol=1e-10, atol=1e-12):
        return False
    distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assigned = distances[np.arange(len(values)), labels]
    return bool(np.allclose(assigned, distances.min(axis=1), rtol=1e-10, atol=1e-12))


class RobustRollingKMeans:
    """Incrementally fit current centroids and classify with their last W states.

    Randomness is used only in the first cross section (seeded k-means++).
    Subsequent Lloyd fits use snapped seeds and ``n_init=1``. The rolling rule
    minimizes distance over all centroid/age pairs, rather than averaging the
    centroids or voting over labels. Ties favor the newest bank, then cluster ID.
    Missing assets can leave/re-enter; K and the feature schema stay fixed.
    Invalid or degenerate updates raise before modifying fitted state.
    """

    def __init__(self, n_clusters: int = 3, window: int = 6, *,
                 random_state: int = 42, n_init: int = 10, max_iter: int = 300):
        for name, value in (("n_clusters", n_clusters), ("window", window),
                            ("n_init", n_init), ("max_iter", max_iter)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.n_clusters = n_clusters
        self.window = window
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.step_count_ = 0
        self._history: deque[np.ndarray] = deque(maxlen=window)
        self._features: tuple[str, ...] | None = None
        self._counts: np.ndarray | None = None

    @property
    def cluster_centers_(self) -> np.ndarray:
        if not self._history:
            raise ValueError("Model has not been updated yet")
        return self._history[-1].copy()

    @property
    def centroid_history_(self) -> np.ndarray:
        """Oldest to newest, shape (min(t, W), K, D); returned as a copy."""
        if not self._history:
            raise ValueError("Model has not been updated yet")
        return np.stack(self._history)

    def _validated(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = validate_cross_section(frame)
        if self._features is not None and tuple(result.columns) != self._features:
            raise ValueError("Feature names/order changed; start a new model")
        return result

    @staticmethod
    def _classify(values: np.ndarray, history: list[np.ndarray]) -> np.ndarray:
        bank = np.concatenate(history[::-1], axis=0)  # newest wins distance ties
        distances = np.sum((values[:, None, :] - bank[None, :, :]) ** 2, axis=2)
        return distances.argmin(axis=1) % history[0].shape[0]

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Classify against the current rolling bank without advancing time."""
        if not self._history:
            raise ValueError("Model has not been updated yet")
        ordered = self._validated(frame)
        labels = self._classify(ordered.to_numpy(), list(self._history))
        return pd.Series(labels, index=ordered.index, name="cluster").reindex(frame.index)

    @threadpool_limits.wrap(limits=1)
    def update(self, frame: pd.DataFrame) -> ClusteringStep:
        """Advance one observation period; the caller enforces chronological order."""
        ordered = self._validated(frame)
        values = ordered.to_numpy()
        if len(np.unique(values, axis=0)) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} distinct feature locations")
        seeds, priority = None, np.arange(self.n_clusters)
        if self._history:
            seeds, priority = select_initial_centers(values, self._history[-1], self._counts)
        fitted = KMeans(
            n_clusters=self.n_clusters, init="k-means++" if seeds is None else seeds,
            n_init=self.n_init if seeds is None else 1, random_state=self.random_state,
            max_iter=self.max_iter, tol=0.0, algorithm="lloyd",
        ).fit(values)
        if fitted.n_iter_ >= self.max_iter and not is_lloyd_fixed_point(
            values, fitted.cluster_centers_, fitted.labels_,
        ):
            raise RuntimeError("Lloyd iteration limit reached; update was not committed")
        centers = fitted.cluster_centers_.copy()
        lloyd_labels = fitted.labels_.copy()
        counts = np.bincount(lloyd_labels, minlength=self.n_clusters)
        if np.any(counts == 0) or len(np.unique(centers, axis=0)) < self.n_clusters:
            raise ValueError("Collapsed/empty current cluster; update was not committed")
        if not self._history:
            # Initial IDs are arbitrary; order once. Never renumber subsequent IDs.
            priority = np.argsort(counts, kind="stable")
            inverse = np.argsort(priority)
            centers, counts = centers[priority], counts[priority]
            lloyd_labels = inverse[lloyd_labels]
        history = (list(self._history) + [centers])[-self.window:]
        rolling_labels = self._classify(values, history)
        result = ClusteringStep(
            labels=pd.Series(rolling_labels, index=ordered.index, name="cluster").reindex(frame.index),
            lloyd_labels=pd.Series(lloyd_labels, index=ordered.index, name="lloyd_cluster").reindex(frame.index),
            centers=centers.copy(), counts=counts.copy(),
            initial_centers=None if seeds is None else seeds.copy(),
            selection_order=priority.copy(), inertia=float(fitted.inertia_),
            n_iter=int(fitted.n_iter_), step=self.step_count_ + 1,
        )
        self._history.append(centers)
        self._counts = counts
        self._features = tuple(ordered.columns)
        self.step_count_ += 1
        return result
