"""Fixed-channel, framework-neutral sequences for joint currency forecasting.

One shared currency axis is stored once. Frozen memberships select channels;
neither changing membership nor missing values can silently replace a channel.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fx_forecasting.data.panel import FeaturePanel, LOCAL_FEATURES, GLOBAL_FEATURES, OPTIONAL_LOCAL


@dataclass
class NeuralDataset:
    arrays: dict[str, np.ndarray]
    metadata: dict
    groups: dict


@dataclass
class NeuralBatch:
    X_local: np.ndarray
    X_local_mask: np.ndarray
    X_local_carried: np.ndarray
    X_global: np.ndarray
    X_global_mask: np.ndarray
    Y: np.ndarray
    Y_mask: np.ndarray
    origins: np.ndarray
    input_periods: np.ndarray
    target_endpoints: np.ndarray
    currencies: tuple[str, ...]
    local_features: tuple[str, ...]
    global_features: tuple[str, ...]
    horizons: tuple[int, ...]


def frozen_groups(memberships: pd.DataFrame, currencies: list[str]) -> dict:
    """Convert membership rows to non-ragged JSON channel-index lists."""
    positions = {c: i for i, c in enumerate(currencies)}
    groups = {"all:k1": {"method": "all", "k": 1, "clusters": [
        {"cluster": 0, "currencies": currencies, "indices": list(range(len(currencies))),
         "representative": None}]}}
    for (method, k), table in memberships.groupby(["method", "k"], sort=True):
        if table.currency.duplicated().any() or set(table.currency) != set(currencies):
            raise ValueError(f"{method}/K={k}: membership must contain the exact currency axis once")
        clusters = []
        for cluster, part in table.groupby("cluster", sort=True):
            names = sorted(part.currency)
            representative = part.representative.iloc[0] if "representative" in part else names[0]
            clusters.append({"cluster": int(cluster), "currencies": names,
                             "indices": [positions[c] for c in names], "representative": representative})
        if len(clusters) != int(k):
            raise ValueError(f"{method}/K={k}: expected K nonempty groups")
        groups[f"{method}:k{int(k)}"] = {"method": method, "k": int(k), "clusters": clusters}
    return groups


def _fit_transform(values: np.ndarray, dates: pd.PeriodIndex, cutoff: pd.Period,
                   names: tuple[str, ...], optional: tuple[str, ...] = ()):
    mask = np.isfinite(values)
    transformed = np.empty_like(values, dtype=float)
    stats = {}
    for j, name in enumerate(names):
        raw = values[..., j]
        training = raw[dates <= cutoff]
        observed = training[np.isfinite(training)]
        if not len(observed):
            raise ValueError(f"{name} is entirely absent in training; cannot learn a replacement or scale")
        median = float(np.median(observed)) if name in optional else None
        training_values = np.where(np.isfinite(training), training, median) if median is not None else observed
        mean, scale = float(np.mean(training_values)), float(np.std(training_values))
        constant = scale <= np.finfo(float).eps
        if constant:
            scale = 1.0
        filled = np.where(np.isfinite(raw), raw, median) if median is not None else raw
        transformed[..., j] = np.where(np.isfinite(filled), (filled-mean)/scale, 0.0)
        stats[name] = {"mean": mean, "scale": scale, "median": median, "constant": bool(constant),
                       "observed_training_cells": int(len(observed))}
    return transformed.astype(np.float32), mask, stats


def build_neural_dataset(panel: FeaturePanel, groups: dict, *, lookback: int = 12,
                         horizons: tuple[int, ...] = (1, 3, 12),
                         fit_end: str = "2019-12", validation_end: str = "2022-12") -> NeuralDataset:
    """Build one full-history bundle; masks enforce split-specific target maturity.

    Learned transforms use only monthly feature cells available by fit_end.
    Optional macro missing values use a pooled training median. All other
    missing numerical inputs are zero placeholders with false masks.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    if not horizons or len(set(horizons)) != len(horizons) or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons must be distinct positive integers")
    dates = panel.fx.index
    if len(dates) < lookback or not dates.equals(pd.period_range(dates.min(), dates.max(), freq="M")):
        raise ValueError("Need a complete ascending monthly calendar at least as long as lookback")
    train_end, val_end = pd.Period(fit_end, "M"), pd.Period(validation_end, "M")
    if not dates.min() <= train_end <= dates.max() or train_end >= val_end:
        raise ValueError("Require a training cutoff inside the data calendar and an ensuing validation period")
    currencies = sorted(panel.fx.columns)
    for definition in groups.values():
        flattened = [c for g in definition["clusters"] for c in g["currencies"]]
        if sorted(flattened) != currencies:
            raise ValueError("Every grouping must partition the exact currency axis")
        for group in definition["clusters"]:
            if [currencies[i] for i in group["indices"]] != group["currencies"]:
                raise ValueError("Group channel indices do not match currency IDs")
    index = pd.MultiIndex.from_product([dates, currencies], names=["period", "currency"])
    local = panel.local.reindex(index)[list(LOCAL_FEATURES)].to_numpy(dtype=float).reshape(len(dates), len(currencies), -1)
    global_values = panel.global_features.reindex(dates)[list(GLOBAL_FEATURES)].to_numpy(dtype=float)
    local_values, local_mask, local_stats = _fit_transform(local, dates, train_end, LOCAL_FEATURES, OPTIONAL_LOCAL)
    global_scaled, global_mask, global_stats = _fit_transform(global_values, dates, train_end, GLOBAL_FEATURES)
    carried = np.zeros(local.shape, dtype=bool)
    provenance = panel.provenance.reindex(index)
    for feature, flags in (("rate_diff", ("policy_carried_domestic", "policy_carried_us")),
                           ("inflation_diff", ("inflation_carried_domestic", "inflation_carried_us"))):
        flag = np.zeros(len(index), dtype=bool)
        for name in flags:
            if name in provenance:
                flag |= provenance[name].fillna(False).to_numpy(dtype=bool)
        j = LOCAL_FEATURES.index(feature)
        carried[..., j] = flag.reshape(len(dates), len(currencies)) & local_mask[..., j]
    origin_positions = np.arange(lookback-1, len(dates))
    origins = dates[origin_positions]
    windows = lambda a: np.stack([a[i-lookback+1:i+1] for i in origin_positions])
    split = np.where(origins <= train_end, "train", np.where(origins <= val_end, "validation", "test"))
    split_cutoffs = pd.PeriodIndex([train_end if s == "train" else min(val_end, dates.max()) if s == "validation" else dates.max()
                                  for s in split], freq="M")
    logfx = np.log(panel.fx.reindex(columns=currencies).where(panel.fx.reindex(columns=currencies).gt(0)))
    target_values, target_masks, endpoints = [], [], []
    for horizon in horizons:
        valid = logfx.notna().rolling(horizon+1, min_periods=horizon+1).sum().shift(-horizon).eq(horizon+1)
        target = (100*(logfx.shift(-horizon)-logfx)).to_numpy()[origin_positions]
        ends = origins + horizon
        available = valid.to_numpy()[origin_positions] & (ends <= split_cutoffs)[:, None]
        target_values.append(np.where(available, target, 0.0).astype(np.float32))
        target_masks.append(available)
        endpoints.append(ends.astype(str).to_numpy())
    arrays = {
        "X_local": windows(local_values), "X_local_mask": windows(local_mask),
        "X_local_carried": windows(carried), "X_global": windows(global_scaled),
        "X_global_mask": windows(global_mask), "Y": np.stack(target_values, axis=-1),
        "Y_mask": np.stack(target_masks, axis=-1), "origins": origins.astype(str).to_numpy(dtype="U7"),
        "input_periods": windows(dates.astype(str).to_numpy(dtype="U7")),
        "target_endpoints": np.stack(endpoints, axis=-1).astype("U7"), "split": np.asarray(split, dtype="U10"),
    }
    if not all(np.isfinite(arrays[name]).all() for name in ("X_local", "X_global", "Y")):
        raise ValueError("Nonfinite numeric tensor encountered")
    metadata = {
        "schema_version": 1, "currencies": currencies, "local_features": list(LOCAL_FEATURES),
        "global_features": list(GLOBAL_FEATURES), "optional_local": list(OPTIONAL_LOCAL),
        "horizons": list(horizons), "lookback": lookback,
        "fit_end": fit_end, "validation_end": validation_end, "data_end": str(dates.max()),
        "preprocessing": {"local": local_stats, "global": global_stats, "fit_end": fit_end,
                          "scaling": "Pooled per-feature training mean/std; optional macro median before scaling",
                          "target_scaling": "None: original percent cumulative FX log changes"},
        "source_absent": {c["currency"]: {"policy": c["source_series"]["policy"] is None,
                                          "cpi": c["source_series"]["cpi"] is None} for c in panel.config["currencies"]},
        "mandatory_missing": "Scaled zero placeholder, false availability mask; never an observed/imputed economic value",
        "target_mask": "Every intervening FX level eligible, target endpoint observed and within its split cutoff",
    }
    return NeuralDataset(arrays, metadata, groups)


def _validate_dataset(dataset: NeuralDataset):
    """Protect channel identities and array axes at the public load boundary."""
    a, meta = dataset.arrays, dataset.metadata
    if meta.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema_version")
    currencies = meta["currencies"]
    if len(set(currencies)) != len(currencies) or currencies != sorted(currencies):
        raise ValueError("Dataset currency axis must be sorted and unique")
    n, length, c = len(a["origins"]), meta["lookback"], len(currencies)
    expected = {"X_local": (n, length, c, len(meta["local_features"])),
                "X_global": (n, length, len(meta["global_features"])),
                "Y": (n, c, len(meta["horizons"])), "input_periods": (n, length),
                "target_endpoints": (n, len(meta["horizons"])), "split": (n,)}
    for key, shape in expected.items():
        if key not in a or a[key].shape != shape:
            raise ValueError(f"Invalid {key} shape; expected {shape}")
    for key, parent in (("X_local_mask", "X_local"), ("X_local_carried", "X_local"),
                        ("X_global_mask", "X_global"), ("Y_mask", "Y")):
        if key not in a or a[key].shape != a[parent].shape or a[key].dtype != np.bool_:
            raise ValueError(f"Invalid {key} mask shape or dtype")
    for definition in dataset.groups.values():
        flattened = []
        for group in definition["clusters"]:
            ids = group["indices"]
            if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < c for i in ids):
                raise ValueError("Invalid group channel index")
            if [currencies[i] for i in ids] != group["currencies"]:
                raise ValueError("Group channel indices do not match currency IDs")
            flattened.extend(group["currencies"])
        if sorted(flattened) != currencies:
            raise ValueError("Group definitions must partition the exact currency axis")


def save_neural_dataset(dataset: NeuralDataset, output_dir: Path) -> list[str]:
    _validate_dataset(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "dataset.npz", **dataset.arrays)
    (output_dir / "dataset_metadata.json").write_text(json.dumps(dataset.metadata, indent=2)+"\n")
    (output_dir / "groups.json").write_text(json.dumps(dataset.groups, indent=2)+"\n")
    return ["dataset.npz", "dataset_metadata.json", "groups.json"]


def _view(metadata: dict, feature_view: str, with_credit: bool):
    if feature_view not in ("fx", "macro"):
        raise ValueError("feature_view must be 'fx' or 'macro'")
    local = [i for i, name in enumerate(metadata["local_features"]) if feature_view == "macro" or name.startswith("fx_change_") or name == "volatility_12m"]
    global_columns = [] if feature_view == "fx" else list(range(len(metadata["global_features"]))) if with_credit else [metadata["global_features"].index("usd_neer_change")]
    mandatory = [i for i in local if metadata["local_features"][i] not in metadata["optional_local"]]
    return local, global_columns, mandatory


def select_cluster_dataset(dataset: NeuralDataset, *, method: str = "adjusted_average", k: int = 2,
                           cluster: int = 0, split: str = "train", horizon: int | None = None,
                           feature_view: str = "macro", with_credit: bool = True,
                           complete_inputs: bool = True, require_all_targets: bool = False) -> NeuralBatch:
    """Select stable channels and eligible origins without refitting anything.

    Defaults require complete mandatory inputs and at least one observed target
    cell. Optional macro missingness remains explicit. require_all_targets=True
    additionally requires every selected currency/horizon target to be observed.
    """
    if split not in ("train", "validation", "test", "all"):
        raise ValueError("Unknown split")
    _validate_dataset(dataset)
    definition = dataset.groups[f"{method}:k{k}"]
    group = next((g for g in definition["clusters"] if g["cluster"] == cluster), None)
    if group is None:
        raise ValueError("Unknown cluster")
    a, meta = dataset.arrays, dataset.metadata
    if horizon is not None and horizon not in meta["horizons"]:
        raise ValueError("Unknown forecast horizon")
    hs = list(range(len(meta["horizons"]))) if horizon is None else [meta["horizons"].index(horizon)]
    loc, glob, mandatory = _view(meta, feature_view, with_credit)
    ids = group["indices"]
    take = np.ones(len(a["origins"]), dtype=bool) if split == "all" else a["split"] == split
    if complete_inputs:
        take &= a["X_local_mask"][:, :, ids, :][:, :, :, mandatory].all(axis=(1, 2, 3))
        if glob:
            take &= a["X_global_mask"][:, :, glob].all(axis=(1, 2))
    target_mask = a["Y_mask"][:, ids, :][:, :, hs]
    take &= target_mask.all(axis=(1, 2)) if require_all_targets else target_mask.any(axis=(1, 2))
    return NeuralBatch(
        X_local=a["X_local"][take][:, :, ids, :][:, :, :, loc],
        X_local_mask=a["X_local_mask"][take][:, :, ids, :][:, :, :, loc],
        X_local_carried=a["X_local_carried"][take][:, :, ids, :][:, :, :, loc],
        X_global=a["X_global"][take][:, :, glob], X_global_mask=a["X_global_mask"][take][:, :, glob],
        Y=a["Y"][take][:, ids, :][:, :, hs], Y_mask=target_mask[take], origins=a["origins"][take],
        input_periods=a["input_periods"][take], target_endpoints=a["target_endpoints"][take][:, hs],
        currencies=tuple(group["currencies"]), local_features=tuple(meta["local_features"][i] for i in loc),
        global_features=tuple(meta["global_features"][i] for i in glob), horizons=tuple(meta["horizons"][i] for i in hs),
    )


def load_cluster_dataset(output_dir: Path, **selection) -> NeuralBatch:
    """Load an NPZ with allow_pickle=False and select a documented group/view."""
    output_dir = Path(output_dir)
    with np.load(output_dir / "dataset.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    dataset = NeuralDataset(arrays, json.loads((output_dir / "dataset_metadata.json").read_text()),
                            json.loads((output_dir / "groups.json").read_text()))
    return select_cluster_dataset(dataset, **selection)


def eligibility_tables(dataset: NeuralDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Origin-level input/target eligibility plus true sequence counts by split."""
    a, meta = dataset.arrays, dataset.metadata
    rows = []
    for definition in dataset.groups.values():
        for group in definition["clusters"]:
            ids = group["indices"]
            target = a["Y_mask"][:, ids, :]
            for view, with_credit, label in (("fx", False, "fx"), ("macro", False, "macro_without_credit"), ("macro", True, "macro")):
                _, global_columns, mandatory = _view(meta, view, with_credit)
                eligible = a["X_local_mask"][:, :, ids, :][:, :, :, mandatory].all(axis=(1, 2, 3))
                if global_columns:
                    eligible &= a["X_global_mask"][:, :, global_columns].all(axis=(1, 2))
                for j, horizon in enumerate(meta["horizons"]):
                    frame = pd.DataFrame({"method": definition["method"], "k": definition["k"],
                                          "cluster": group["cluster"], "n_currencies": len(ids), "view": label,
                                          "horizon": horizon, "origin": a["origins"], "split": a["split"],
                                          "inputs_eligible": eligible, "observed_target_cells": target[:, :, j].sum(axis=1),
                                          "all_targets_observed": target[:, :, j].all(axis=1)})
                    rows.append(frame)
    details = pd.concat(rows, ignore_index=True)
    details["usable_origin"] = details.inputs_eligible & details.observed_target_cells.gt(0)
    details["complete_target_origin"] = details.inputs_eligible & details.all_targets_observed
    summary = details.groupby(["method", "k", "cluster", "n_currencies", "view", "horizon", "split"], as_index=False).agg(
        calendar_origins=("origin", "size"), complete_input_origins=("inputs_eligible", "sum"),
        usable_origins=("usable_origin", "sum"), complete_target_origins=("complete_target_origin", "sum"))
    return details, summary
