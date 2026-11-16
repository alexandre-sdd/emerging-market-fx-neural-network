"""Trailing currency relationships for exploratory forecast context.

Correlations use only observed, contiguous returns. Fixed training-fitted peer
groups may describe training rows, but do not support forecasts before their
fit cutoff. Thresholds are learned separately from the trailing descriptors.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _calendar(fx: pd.DataFrame):
    if fx.empty or not isinstance(fx.index, pd.PeriodIndex) or fx.index.freqstr != "M":
        raise ValueError("FX requires a nonempty monthly PeriodIndex")
    if (not fx.index.is_unique or not fx.columns.is_unique
            or not fx.index.equals(pd.period_range(fx.index.min(), fx.index.max(), freq="M"))):
        raise ValueError("FX requires unique currencies and a complete ascending monthly calendar")
    if len(fx.columns) < 2 or not all(isinstance(c, str) for c in fx.columns):
        raise ValueError("Need at least two string currency identifiers")


def rolling_pair_correlations(
    fx: pd.DataFrame, *, windows: tuple[int, ...] = (6, 9, 12, 18, 60),
    return_views: tuple[str, ...] = ("raw", "median_adjusted"),
) -> pd.DataFrame:
    """All unordered pairs, including unavailable rows and paired observation counts.

    Each estimate needs exactly window consecutive observed monthly returns
    (hence window+1 FX levels). Counts include paired observations in the trailing
    calendar window even where insufficient/constant data prevent correlation.
    Sample selection/peg masks must already be applied by the caller.
    """
    _calendar(fx)
    if (not windows or len(set(windows)) != len(windows)
            or any(isinstance(w, bool) or not isinstance(w, int) or w < 2 for w in windows)):
        raise ValueError("Windows must be distinct integers of at least two months")
    if (not return_views or len(set(return_views)) != len(return_views)
            or set(return_views) - {"raw", "median_adjusted"}):
        raise ValueError("Return views must be distinct raw/median_adjusted values")
    currencies = sorted(fx.columns)
    positive = fx.reindex(columns=currencies).where(lambda x: np.isfinite(x) & x.gt(0))
    returns = 100 * np.log(positive).diff()
    dates = returns.index
    left, right = np.triu_indices(len(currencies), 1)
    pair_count = len(left)
    joint = returns.notna().to_numpy()[:, left] & returns.notna().to_numpy()[:, right]
    expanded_index = pd.MultiIndex.from_product([dates, currencies])
    frames = []
    for view in return_views:
        values = returns if view == "raw" else returns.sub(returns.median(axis=1), axis=0)
        for window in windows:
            rolling = values.rolling(window, min_periods=window)
            correlations = rolling.corr(pairwise=True).reindex(index=expanded_index, columns=currencies)
            estimates = correlations.to_numpy().reshape(len(dates), len(currencies), len(currencies))[:, left, right]
            scales = rolling.std(ddof=1).to_numpy()
            variable = (scales[:, left] > 1e-12) & (scales[:, right] > 1e-12)
            counts = pd.DataFrame(joint).rolling(window, min_periods=1).sum().to_numpy(dtype=np.int16)
            valid = variable & (counts == window) & np.isfinite(estimates)
            estimates = np.where(valid, np.clip(estimates, -1, 1), np.nan)
            frames.append(pd.DataFrame({
                "period": np.repeat(dates, pair_count),
                "currency_a": np.tile(np.asarray(currencies)[left], len(dates)),
                "currency_b": np.tile(np.asarray(currencies)[right], len(dates)),
                "window": np.int16(window), "return_view": view,
                "observed_pairs": counts.ravel(), "correlation": estimates.ravel(),
            }))
    result = pd.concat(frames, ignore_index=True)
    for column in ("currency_a", "currency_b", "return_view"):
        result[column] = result[column].astype("category")
    return result


def compact_correlations(
    pairs: pd.DataFrame, currencies: list[str], peer_groups: Mapping[str, int], *,
    window: int = 9, return_view: str = "raw", minimum_peers: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EUR, CNY and mean fixed-peer correlations on a fixed currency axis.

    Self-correlations are never used. Missing/constant pairs remain unavailable;
    the peer mean requires minimum_peers and is accompanied by its denominator.
    """
    if set(peer_groups) != set(currencies) or len(set(currencies)) != len(currencies):
        raise ValueError("Peer groups must map the exact unique currency axis")
    if isinstance(minimum_peers, bool) or not isinstance(minimum_peers, int) or minimum_peers < 1:
        raise ValueError("minimum_peers must be a positive integer")
    selected = pairs.loc[pairs.window.eq(window) & pairs.return_view.eq(return_view)]
    if selected.empty:
        raise ValueError("Requested correlation window/view is absent")
    dates = pd.PeriodIndex(selected.period.unique(), freq="M").sort_values()
    index = pd.MultiIndex.from_product([dates, sorted(currencies)], names=["period", "currency"])
    features = pd.DataFrame(np.nan, index=index, columns=["corr_eur", "corr_cny", "corr_peers"])
    counts = pd.DataFrame(0, index=index, columns=["peer_count"], dtype=int)
    lookup = selected.set_index(["period", "currency_a", "currency_b"]).correlation
    if not lookup.index.is_unique:
        raise ValueError("Duplicate period/currency-pair correlation")
    available_pairs = {(str(a), str(b)): group.droplevel([1, 2]).reindex(dates)
                       for (a, b), group in lookup.groupby(level=[1, 2], observed=True)}

    def pair(a, b):
        return available_pairs.get(tuple(sorted((a, b))), pd.Series(np.nan, index=dates))

    for currency in sorted(currencies):
        rows = pd.MultiIndex.from_product([dates, [currency]], names=index.names)
        for anchor, name in (("EUR", "corr_eur"), ("CNY", "corr_cny")):
            if currency != anchor and anchor in currencies:
                features.loc[rows, name] = pair(currency, anchor).to_numpy()
        peers = [c for c in currencies if c != currency and peer_groups[c] == peer_groups[currency]]
        if peers:
            histories = pd.concat([pair(currency, c) for c in peers], axis=1)
            number = histories.notna().sum(axis=1)
            features.loc[rows, "corr_peers"] = histories.mean(axis=1).where(number >= minimum_peers).to_numpy()
            counts.loc[rows, "peer_count"] = number.to_numpy()
    return features, counts


def correlation_regime_flags(
    features: pd.DataFrame, *, fit_end: str = "2019-12", minimum_months: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-currency training quartiles; unknown and degenerate states stay NaN.

    Flags are train-fitted encodings for evaluation after fit_end. They must be
    refitted inside each training fold for an earlier-origin backtest.
    """
    if (features.index.names != ["period", "currency"] or not features.index.is_unique
            or not isinstance(minimum_months, int) or isinstance(minimum_months, bool) or minimum_months < 2):
        raise ValueError("Need unique (period, currency) features and minimum_months >= 2")
    cutoff = pd.Period(fit_end, "M")
    flags = pd.DataFrame(np.nan, index=features.index,
                         columns=[f"{name}_{state}" for name in features for state in ("low", "high")])
    records = []
    for currency, history in features.groupby(level="currency", sort=True):
        training = history.loc[history.index.get_level_values("period") <= cutoff]
        for feature in features:
            observed = training[feature].replace([np.inf, -np.inf], np.nan).dropna()
            lower, upper = observed.quantile([.25, .75]) if len(observed) else (np.nan, np.nan)
            usable = len(observed) >= minimum_months and upper-lower > 1e-12
            records.append({"currency": currency, "feature": feature, "fit_end": fit_end,
                            "training_months": len(observed), "lower_quartile": lower,
                            "upper_quartile": upper, "usable": bool(usable)})
            if usable:
                values = history[feature]
                flags.loc[history.index, f"{feature}_low"] = values.le(lower).astype(float).where(values.notna())
                flags.loc[history.index, f"{feature}_high"] = values.ge(upper).astype(float).where(values.notna())
    return flags, pd.DataFrame(records)
