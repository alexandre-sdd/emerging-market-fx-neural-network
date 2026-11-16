"""Matplotlib figures for the two-feature rolling clustering experiment."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from fx_forecasting.evaluation.rolling_clustering import RollingExperiment

COLORS = ["#247ba0", "#e49336", "#31936a", "#9867af", "#cb5364", "#657387"]


def _label_currencies(ax, cross: pd.DataFrame):
    """Place short currency labels without collisions in dense cross sections."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(size=7)
    pixels_per_point = ax.figure.dpi / 72
    points = ax.transData.transform(cross.to_numpy())
    point_boxes = [Bbox.from_bounds(x-3, y-3, 6, 6) for x, y in points]
    occupied = []
    offsets = [(5, 4)] + [(x, y) for y in (4, 13, -13, 24, -24, 36, -36, 50, -50)
                          for x in (5, 17, -20, 30, -35, 45, -50)]
    for (currency, values), (px, py) in zip(cross.iterrows(), points):
        width, height, descent = renderer.get_text_width_height_descent(currency, font, False)
        choices = []
        for dx, dy in offsets:
            box = Bbox.from_bounds(px + dx*pixels_per_point - 1,
                                   py + dy*pixels_per_point - 1, width+3, height+descent+3)
            collisions = sum(box.overlaps(other) for other in occupied)
            point_collisions = sum(box.overlaps(other) for other in point_boxes)
            outside = not ax.bbox.contains(box.x0, box.y0) or not ax.bbox.contains(box.x1, box.y1)
            score = 10000 * outside + 1000 * collisions + 80 * point_collisions + np.hypot(dx, dy)
            choices.append((score, dx, dy, box))
        _, dx, dy, box = min(choices, key=lambda item: item[0])
        occupied.append(box)
        arrow = {"arrowstyle": "-", "color": "#666666", "lw": .4} if np.hypot(dx, dy) > 12 else None
        ax.annotate(currency, values, xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha="left", va="bottom", arrowprops=arrow)


def plot_diagnostics(result: RollingExperiment, start: str = "2010-01"):
    rows = result.metrics.loc[result.metrics.period.ge(start)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for name, group in rows.groupby("model", sort=True):
        dates = pd.PeriodIndex(group.period, freq="M").to_timestamp()
        for ax, field in zip(axes, ("switch_rate", "ari", "silhouette")):
            ax.plot(dates, group[field].rolling(12, min_periods=1).mean(), label=name, lw=1.4)
    for ax, label in zip(axes, ("Asset switch rate", "Adjusted Rand index", "Silhouette (Euclidean)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].set_title("Rolling 12-month means: identity changes, partition changes, separation")
    axes[0].legend(ncol=3, fontsize=9)
    return fig


def plot_membership(result: RollingExperiment, model: str = "r2k_w6", start: str = "2010-01"):
    selected = result.labels.loc[result.labels.model.eq(model) & result.labels.period.ge(start)]
    if selected.empty:
        raise ValueError(f"No labels for {model}")
    table = selected.pivot(index="currency", columns="period", values="cluster").sort_index()
    periods = pd.period_range(table.columns.min(), table.columns.max(), freq="M").astype(str)
    table = table.reindex(columns=periods)
    k = int(selected.cluster.max()) + 1
    cmap = ListedColormap(COLORS[:k])
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(14, max(5, .22 * len(table))), constrained_layout=True)
    mesh = ax.imshow(np.ma.masked_invalid(table.to_numpy(dtype=float)), aspect="auto",
                     interpolation="nearest", cmap=cmap, vmin=-.5, vmax=k-.5)
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    ticks = np.arange(0, len(periods), 24)
    ax.set_xticks(ticks, periods[ticks], rotation=45, ha="right")
    ax.set_title(f"{model}: cluster identity at each origin (gray = ineligible)")
    fig.colorbar(mesh, ax=ax, ticks=range(k), label="Cluster ID")
    return fig


def plot_snapshot(result: RollingExperiment, features: pd.DataFrame, *,
                  model: str = "r2k_w6", window: int = 6, period: str | None = None):
    """Plot the true two-feature classification boundary and centroid paths.

    Centroids remain in their original normalized coordinates. This is not a
    projection of a higher-dimensional boundary, nor a re-fit at the last date.
    """
    period = period or result.labels.period.max()
    month = pd.Period(period, "M")
    raw = features.xs(month, level="period").dropna()
    if raw.shape[1] != 2:
        raise ValueError("Boundary plots require exactly two features")
    stats = result.normalization.loc[result.normalization.period.eq(period)].set_index("feature")
    if not stats.index.is_unique or not set(raw.columns).issubset(stats.index):
        raise ValueError("Missing or duplicate recorded normalization for requested period")
    stats = stats.reindex(raw.columns)
    if not np.isfinite(stats[["mean", "scale"]].to_numpy()).all() or stats["scale"].le(0).any():
        raise ValueError("Recorded normalization must have finite means and positive scales")
    cross = (raw - stats["mean"]) / stats["scale"]
    records = result.centroids.loc[result.centroids.model.eq(model) & result.centroids.period.le(period)]
    dates = sorted(records.period.unique())[-window:]
    if not dates or dates[-1] != period:
        raise ValueError("No fitted centroids at requested period")
    records = records.loc[records.period.isin(dates)].sort_values(["period", "cluster"], ascending=[False, True])
    bank = records[list(raw.columns)].to_numpy()
    bank_ids = records.cluster.to_numpy()
    k = int(bank_ids.max()) + 1
    all_points = np.vstack([cross.to_numpy(), bank])
    low, high = all_points.min(axis=0) - .5, all_points.max(axis=0) + .5
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 260), np.linspace(low[1], high[1], 220))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    distances = ((grid[:, None, :] - bank[None, :, :]) ** 2).sum(axis=2)
    labels = bank_ids[distances.argmin(axis=1)].reshape(xx.shape)
    cmap = ListedColormap(COLORS[:k])
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.pcolormesh(xx, yy, labels, cmap=cmap, norm=BoundaryNorm(np.arange(k+1)-.5, k), alpha=.15, shading="auto")
    selected = result.labels.loc[result.labels.model.eq(model) & result.labels.period.eq(period)].set_index("currency")
    for cluster in range(k):
        members = cross.loc[selected.cluster.reindex(cross.index).eq(cluster)]
        ax.scatter(members.iloc[:, 0], members.iloc[:, 1], color=COLORS[cluster], s=34, label=f"Cluster {cluster}")
        path = records.loc[records.cluster.eq(cluster)].sort_values("period")
        ax.plot(path[raw.columns[0]], path[raw.columns[1]], color=COLORS[cluster], marker="x", lw=1.5)
        ax.scatter(path[raw.columns[0]].iloc[-1], path[raw.columns[1]].iloc[-1],
                   marker="*", s=200, color=COLORS[cluster], edgecolor="black", linewidth=.5)
    names = {"annual_log_change": "Annualized mean FX log change",
             "annualized_volatility": "Annualized FX volatility"}
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           xlabel=f"{names.get(raw.columns[0], raw.columns[0])} (cross-sectional z-score)",
           ylabel=f"{names.get(raw.columns[1], raw.columns[1])} (cross-sectional z-score)",
           title=f"{model} at {period}: {len(dates)} centroid states; stars mark current centers")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=.15)
    _label_currencies(ax, cross)
    return fig
