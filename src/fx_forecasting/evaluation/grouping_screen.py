"""Exploratory, FX-only test of grouping's role in forecast pooling.

This is a small Ridge screen, not the project's production macro/NN backtest.
Every fit sees only targets realized by its month-end origin. Grouped fits use
each historical row's own origin-time membership, never a current map backfill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fx_forecasting.data.rolling_features import make_rolling_features

PREDICTORS = ("annual_log_change", "annualized_volatility", "fx_change_1m", "fx_change_3m")


@dataclass
class GroupingScreen:
    forecasts: pd.DataFrame
    fits: pd.DataFrame
    summary: pd.DataFrame
    currency_metrics: pd.DataFrame


def make_screen_panel(fx: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 12)) -> pd.DataFrame:
    """Contiguous FX-only predictors/targets; excluded FX must already be NaN."""
    panel = make_rolling_features(fx)
    positive = fx.where(np.isfinite(fx) & fx.gt(0))
    logs = np.log(positive)
    for lag in (1, 3):
        move = 100 * logs.diff(lag)
        valid = positive.notna().rolling(lag+1, min_periods=lag+1).sum().eq(lag+1)
        panel[f"fx_change_{lag}m"] = move.where(valid).rename_axis(index="period", columns="currency").stack().reindex(panel.index)
    for h in horizons:
        if isinstance(h, bool) or not isinstance(h, int) or h < 1:
            raise ValueError("Horizons must be positive integers")
        valid = positive.notna().rolling(h+1, min_periods=h+1).sum().shift(-h).eq(h+1)
        target = (100 * (logs.shift(-h) - logs)).where(valid)
        panel[f"target_{h}m"] = target.rename_axis(index="period", columns="currency").stack().reindex(panel.index)
    return panel


def summarize_forecasts(forecasts: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report both observation-pooled and equal-currency forecast comparisons."""
    data = forecasts.copy()
    data["squared_error"] = (data.actual - data.prediction) ** 2
    data["rw_squared_error"] = data.actual ** 2
    by_currency = data.groupby(["model", "horizon", "currency"]).agg(
        n=("actual", "size"), mse=("squared_error", "mean"),
        rw_mse=("rw_squared_error", "mean"), fallback_rate=("fallback", "mean"),
    ).reset_index()
    by_currency["r2_os_pct"] = 100 * (1 - by_currency.mse / by_currency.rw_mse.where(by_currency.rw_mse.gt(0)))
    by_currency["reporting_group"] = by_currency.currency.map(metadata["group"])
    data["reporting_group"] = data.currency.map(metadata["group"])
    rows = []
    for (model, horizon), all_rows in data.groupby(["model", "horizon"], sort=True):
        for group in ("all", *sorted(data.reporting_group.dropna().unique())):
            selected = all_rows if group == "all" else all_rows.loc[all_rows.reporting_group.eq(group)]
            currency = by_currency.loc[by_currency.model.eq(model) & by_currency.horizon.eq(horizon)]
            if group != "all":
                currency = currency.loc[currency.reporting_group.eq(group)]
            if selected.empty:
                continue
            rw_mse = selected.rw_squared_error.mean()
            mse = selected.squared_error.mean()
            rows.append({"model": model, "horizon": horizon, "reporting_group": group,
                         "n": len(selected), "n_currencies": selected.currency.nunique(),
                         "first_origin": selected.origin.min(), "last_origin": selected.origin.max(),
                         "rmse": np.sqrt(mse), "pooled_r2_os_pct": 100*(1-mse/rw_mse) if rw_mse > 0 else np.nan,
                         "equal_currency_r2_os_pct": currency.r2_os_pct.mean(),
                         "median_currency_r2_os_pct": currency.r2_os_pct.median(),
                         "fraction_currencies_beating_rw": currency.r2_os_pct.gt(0).mean(),
                         "fallback_rate": selected.fallback.mean()})
    return pd.DataFrame(rows), by_currency


@threadpool_limits.wrap(limits=1)
def run_grouping_screen(
    panel: pd.DataFrame, labels: dict[str, pd.Series], metadata: pd.DataFrame, *,
    evaluation_start: str = "2010-01", horizons: tuple[int, ...] = (1, 3, 12),
    l2: float = 1.0, minimum_rows: int = 120, minimum_months: int = 36,
) -> GroupingScreen:
    """Monthly expanding fits with common samples and identical regularization.

    Each label series is indexed by (period, currency), with causal memberships.
    The Ridge objective is mean squared error + l2 * sum(coef**2), by setting
    sklearn's alpha = n_training_rows * l2. All models share pooled-training-only
    standardization. Small groups fall back explicitly to the pooled model.
    No settings are selected on post-2010 forecast performance.
    """
    if l2 <= 0 or minimum_rows < 1 or minimum_months < 1:
        raise ValueError("Regularization and training minima must be positive")
    if panel.index.names != ["period", "currency"] or not panel.index.is_unique:
        raise ValueError("Expected unique (period, currency) rows")
    shared = panel.index
    for name, series in labels.items():
        if not series.index.is_unique or series.index.names != panel.index.names:
            raise ValueError(f"{name}: labels require unique (period, currency) rows")
        shared = shared.intersection(series.dropna().index)
    data = panel.reindex(shared).sort_index()
    finite_x = np.isfinite(data[list(PREDICTORS)].to_numpy(dtype=float)).all(axis=1)
    data = data.loc[finite_x]
    groups = {name: series.reindex(data.index) for name, series in labels.items()}
    period = data.index.get_level_values("period")
    currencies = data.index.get_level_values("currency")
    x = data[list(PREDICTORS)].to_numpy()
    forecasts, fits = [], []
    for h in horizons:
        y = data[f"target_{h}m"].to_numpy(dtype=float)
        endpoint = period + h
        for origin in period.unique().sort_values():
            if origin < pd.Period(evaluation_start, "M"):
                continue
            train = (endpoint <= origin) & np.isfinite(y)
            test = (period == origin) & np.isfinite(y)
            if not test.any():
                continue
            if train.sum() < minimum_rows or period[train].nunique() < minimum_months:
                raise ValueError(f"Insufficient pooled training history at {origin}, h={h}")
            scaler = StandardScaler().fit(x[train])
            x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])
            y_train, y_test = y[train], y[test]
            pooled = Ridge(alpha=l2*len(y_train), solver="svd").fit(x_train, y_train)
            pooled_pred = pooled.predict(x_test)
            predictions = {"random_walk": (np.zeros(test.sum()), np.zeros(test.sum(), dtype=bool)),
                           "pooled_ridge": (pooled_pred, np.zeros(test.sum(), dtype=bool))}
            # Currency-specific realized historical mean is a useful drift control.
            historical_mean = pd.Series(y_train, index=currencies[train]).groupby(level=0).mean()
            mean_pred = historical_mean.reindex(currencies[test]).fillna(y_train.mean()).to_numpy()
            predictions["currency_mean"] = (mean_pred, np.zeros(test.sum(), dtype=bool))
            for name, series in groups.items():
                train_labels = series.to_numpy()[train]
                test_labels = series.to_numpy()[test]
                predicted, fallback = pooled_pred.copy(), np.zeros(test.sum(), dtype=bool)
                for cluster in np.unique(test_labels):
                    membership = train_labels == cluster
                    route = test_labels == cluster
                    n, months = int(membership.sum()), int(period[train][membership].nunique())
                    use_fallback = n < minimum_rows or months < minimum_months
                    if not use_fallback:
                        model = Ridge(alpha=l2*n, solver="svd").fit(x_train[membership], y_train[membership])
                        predicted[route] = model.predict(x_test[route])
                    else:
                        fallback[route] = True
                    fits.append({"origin": str(origin), "horizon": h, "model": name, "cluster": int(cluster),
                                 "n_train": n, "n_train_months": months,
                                 "n_train_currencies": int(currencies[train][membership].nunique()),
                                 "latest_training_endpoint": str(endpoint[train][membership].max()) if n else None,
                                 "n_predicted": int(route.sum()), "fallback": use_fallback})
                predictions[name] = (predicted, fallback)
            for name, (predicted, fallback) in predictions.items():
                for currency, actual, prediction, used_fallback in zip(currencies[test], y_test, predicted, fallback):
                    forecasts.append({"origin": str(origin), "currency": currency, "horizon": h,
                                      "model": name, "actual": float(actual), "prediction": float(prediction),
                                      "fallback": bool(used_fallback)})
    if not forecasts:
        raise ValueError("No evaluable forecasts")
    forecasts = pd.DataFrame(forecasts)
    summary, currency = summarize_forecasts(forecasts, metadata)
    return GroupingScreen(forecasts, pd.DataFrame(fits), summary, currency)
