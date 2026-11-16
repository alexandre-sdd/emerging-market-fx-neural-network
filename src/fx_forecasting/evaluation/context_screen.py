"""Matched, frozen-cutoff pooled Ridge screen for BIS-only context features.

This consumes currency-month rows, not joint neural sequences. No result here
selects the final architecture or establishes confirmatory forecast evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from fx_forecasting.data.panel import FeaturePanel, LOCAL_FEATURES, GLOBAL_FEATURES, OPTIONAL_LOCAL
from fx_forecasting.evaluation.grouping_screen import make_screen_panel

BASE_FEATURES = (*LOCAL_FEATURES, *GLOBAL_FEATURES, "rate_diff_carried", "inflation_diff_carried")
CORRELATION_FEATURES = ("corr_eur", "corr_cny", "corr_peers")
POLICY_FEATURES = ("domestic_policy_change_1m", "rate_diff_change_3m", "us_policy_change_1m")


@dataclass
class ContextScreen:
    predictions: pd.DataFrame
    validation_grid: pd.DataFrame
    fits: pd.DataFrame
    preprocessing: pd.DataFrame
    summary: pd.DataFrame
    currency_metrics: pd.DataFrame


def make_context_screen_table(panel: FeaturePanel, local: pd.DataFrame,
                              global_features: pd.DataFrame, *, horizon: int = 3) -> pd.DataFrame:
    """Join unscaled monthly inputs and contiguous targets without filling FX."""
    if not local.index.equals(panel.local.index) or not global_features.index.equals(panel.global_features.index):
        raise ValueError("Context must preserve the canonical local and monthly axes")
    if set(local.columns) & set(panel.local.columns) or set(global_features.columns) & set(panel.global_features.columns):
        raise ValueError("Context must not overwrite canonical features")
    data = panel.local.join(local)
    periods = data.index.get_level_values("period")
    for source in (panel.global_features, global_features):
        for column in source:
            data[column] = source[column].reindex(periods).to_numpy()
    for feature, family in (("rate_diff", "policy"), ("inflation_diff", "inflation")):
        flags = panel.provenance[[f"{family}_carried_domestic", f"{family}_carried_us"]]
        data[f"{feature}_carried"] = flags.any(axis=1).astype(float)
    target = make_screen_panel(panel.fx, (horizon,))[f"target_{horizon}m"]
    data["target"] = target.reindex(data.index)
    data["target_endpoint"] = periods + horizon
    mandatory = [name for name in LOCAL_FEATURES if name not in OPTIONAL_LOCAL] + list(GLOBAL_FEATURES)
    data["eligible_input"] = np.isfinite(data[mandatory].to_numpy(dtype=float)).all(axis=1)
    return data


def _metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = predictions.assign(square=lambda x: (x.actual-x.prediction)**2,
                              absolute=lambda x: (x.actual-x.prediction).abs(),
                              rw_square=lambda x: x.actual**2)
    keys = ["split", "model", "currency"]
    by_currency = rows.groupby(keys).agg(n=("actual", "size"), mse=("square", "mean"),
                                        mae=("absolute", "mean"), rw_mse=("rw_square", "mean")).reset_index()
    by_currency["rmse"] = np.sqrt(by_currency.mse)
    by_currency["r2_vs_zero"] = 1-by_currency.mse/by_currency.rw_mse.where(by_currency.rw_mse.gt(0))
    summary = by_currency.groupby(["split", "model"]).agg(
        currencies=("currency", "nunique"), observations=("n", "sum"),
        equal_currency_rmse=("rmse", "mean"), equal_currency_mae=("mae", "mean"),
        equal_currency_r2_vs_zero=("r2_vs_zero", "mean"),
        currencies_beating_zero=("r2_vs_zero", lambda x: int(x.gt(0).sum())),
    ).reset_index()
    return summary, by_currency


def _prepare(data: pd.DataFrame, train: np.ndarray, features: tuple[str, ...]):
    """Training-only median/scaling, explicit masks, and fixed currency indicators."""
    raw = data[list(features)].to_numpy(dtype=float)
    observed = np.isfinite(raw)
    medians = np.array([np.median(raw[train & observed[:, j], j]) if (train & observed[:, j]).any() else 0.
                        for j in range(raw.shape[1])])
    filled = np.where(observed, raw, medians)
    means, scales = filled[train].mean(axis=0), filled[train].std(axis=0)
    scales = np.where(scales > np.finfo(float).eps, scales, 1.)
    # Currency identities are fixed design metadata, not learned from outcomes.
    currency = data.index.get_level_values("currency")
    axis = sorted(currency.unique())
    identity = (currency.to_numpy()[:, None] == np.array(axis)[None, :]).astype(float)
    x = np.column_stack(((filled-means)/scales, (~observed).astype(float), identity))
    stats = pd.DataFrame({"feature": features, "median": medians, "mean": means, "scale": scales,
                          "observed_training_cells": observed[train].sum(axis=0)})
    return x, stats


@threadpool_limits.wrap(limits=1)
def run_context_screen(
    table: pd.DataFrame, *, fit_end: str = "2019-12", validation_end: str = "2022-12",
    l2_grid: tuple[float, ...] = (.001, .01, .1, 1.), minimum_training_months: int = 36,
) -> ContextScreen:
    """Fit through fit_end; select l2 on equal-currency validation RMSE only.

    All comparisons share exact train/validation/test target cells. A validation
    target must mature by validation_end; test targets must already be observed.
    No post-training refit or monthly updating is performed in this prototype.
    Ridge alpha=n_train*l2 gives comparable mean-loss regularization across fits.
    """
    if table.index.names != ["period", "currency"] or not table.index.is_unique:
        raise ValueError("Screen needs unique (period, currency) rows")
    if not l2_grid or any(not np.isfinite(x) or x <= 0 for x in l2_grid):
        raise ValueError("Ridge penalties must be positive finite values")
    if isinstance(minimum_training_months, bool) or not isinstance(minimum_training_months, int) or minimum_training_months < 1:
        raise ValueError("minimum_training_months must be a positive integer")
    end, val_end = pd.Period(fit_end, "M"), pd.Period(validation_end, "M")
    if end >= val_end:
        raise ValueError("Training cutoff must precede validation cutoff")
    data = table.loc[table.eligible_input & np.isfinite(table.target)].sort_index().copy()
    periods = data.index.get_level_values("period")
    endpoint = pd.PeriodIndex(data.target_endpoint, freq="M")
    train = np.asarray((periods <= end) & (endpoint <= end))
    masks = {"validation": np.asarray((periods > end) & (periods <= val_end) & (endpoint <= val_end)),
             "test": np.asarray(periods > val_end)}
    if periods[train].nunique() < minimum_training_months or any(not mask.any() for mask in masks.values()):
        raise ValueError("Need sufficient mature training months and nonempty validation/test samples")
    y = data.target.to_numpy(dtype=float)
    currency = data.index.get_level_values("currency")
    models = {
        "A_existing": BASE_FEATURES,
        "B_correlation": (*BASE_FEATURES, *CORRELATION_FEATURES),
        "C_policy_only": (*BASE_FEATURES, *POLICY_FEATURES),
        "D_correlation_policy": (*BASE_FEATURES, *CORRELATION_FEATURES, *POLICY_FEATURES),
        "own_fx_ridge": ("fx_change_1m", "fx_change_3m", "fx_change_12m", "volatility_12m"),
    }
    predictions, grids, fits, preprocessing = [], [], [], []

    def forecast_rows(name, split, values):
        mask = masks[split]
        return pd.DataFrame({"origin": periods[mask].astype(str), "currency": currency[mask],
                             "split": split, "model": name, "actual": y[mask], "prediction": values})

    for name, features in models.items():
        x, stats = _prepare(data, train, features)
        stats["model"] = name
        preprocessing.append(stats)
        candidates = []
        for l2 in l2_grid:
            model = Ridge(alpha=float(l2)*int(train.sum()), solver="svd").fit(x[train], y[train])
            errors = (model.predict(x[masks["validation"]])-y[masks["validation"]])**2
            score = float(pd.Series(errors, index=currency[masks["validation"]]).groupby(level=0).mean().pow(.5).mean())
            grids.append({"model": name, "l2": l2, "validation_equal_currency_rmse": score})
            candidates.append((score, float(l2), model))
        _, selected_l2, fitted = min(candidates, key=lambda item: (item[0], item[1]))
        fits.append({"model": name, "selected_l2": selected_l2, "training_rows": int(train.sum()),
                     "training_months": periods[train].nunique(), "training_first_origin": str(periods[train].min()),
                     "training_last_origin": str(periods[train].max()),
                     "latest_training_endpoint": str(endpoint[train].max()),
                     "numeric_features": len(features), "design_columns": x.shape[1], "fit_end": fit_end})
        for split, mask in masks.items():
            predictions.append(forecast_rows(name, split, fitted.predict(x[mask])))
    means = pd.Series(y[train], index=currency[train]).groupby(level=0).mean()
    for split, mask in masks.items():
        predictions.append(forecast_rows("zero_change", split, np.zeros(mask.sum())))
        predictions.append(forecast_rows("currency_mean", split, means.reindex(currency[mask]).fillna(y[train].mean()).to_numpy()))
    forecasts = pd.concat(predictions, ignore_index=True)
    summary, currency_metrics = _metrics(forecasts)
    return ContextScreen(forecasts, pd.DataFrame(grids), pd.DataFrame(fits),
                         pd.concat(preprocessing, ignore_index=True), summary, currency_metrics)
