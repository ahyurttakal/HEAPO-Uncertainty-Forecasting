from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import atomic_json


@dataclass(frozen=True)
class TemporalCutoffs:
    train_end: pd.Timestamp
    calibration_end: pd.Timestamp


def _stratified_ids(
    audit: pd.DataFrame, fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if fraction <= 0:
        return audit["household_id"].astype(int).tolist(), []
    ids = audit["household_id"].astype(int).to_numpy()
    labels = audit["hp_type"].fillna("unknown").to_numpy()
    counts = pd.Series(labels).value_counts()
    stratify = labels if len(counts) > 1 and counts.min() >= 2 else None
    keep, held = train_test_split(
        ids, test_size=fraction, random_state=seed, stratify=stratify
    )
    return sorted(keep.tolist()), sorted(held.tolist())


def create_splits(
    audit: pd.DataFrame, cfg: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    eligible = audit[audit["eligible"]].copy()
    seed = int(cfg["project"]["seed"])
    remaining, unseen_test = _stratified_ids(
        eligible, cfg["split"]["unseen_household_fraction"], seed
    )
    remaining_frame = eligible[eligible["household_id"].isin(remaining)]
    adjusted = cfg["split"]["validation_household_fraction"] / max(
        1.0 - cfg["split"]["unseen_household_fraction"], 1e-9
    )
    train_ids, unseen_validation = _stratified_ids(
        remaining_frame, min(adjusted, 0.5), seed + 1
    )

    start = pd.to_datetime(eligible["start_utc"], utc=True).min()
    end = pd.to_datetime(eligible["end_utc"], utc=True).max()
    span = end - start
    train_end = start + cfg["split"]["temporal_train_fraction"] * span
    cal_end = train_end + cfg["split"]["temporal_calibration_fraction"] * span
    split = {
        "training_households": train_ids,
        "unseen_validation_households": unseen_validation,
        "unseen_test_households": unseen_test,
        "train_end": train_end.isoformat(),
        "calibration_end": cal_end.isoformat(),
        "global_start": start.isoformat(),
        "global_end": end.isoformat(),
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(split, output_dir / "split_manifest.json")
    return split


def assign_split(rows: pd.DataFrame, split: dict[str, Any]) -> pd.Series:
    house = rows["Household_ID"].astype(int)
    origin = pd.to_datetime(rows["origin_time"], utc=True)
    result = pd.Series("seen_test", index=rows.index, dtype="object")
    result.loc[origin <= pd.Timestamp(split["train_end"])] = "train"
    result.loc[
        (origin > pd.Timestamp(split["train_end"]))
        & (origin <= pd.Timestamp(split["calibration_end"]))
    ] = "calibration"
    result.loc[house.isin(split["unseen_validation_households"])] = "unseen_validation"
    result.loc[house.isin(split["unseen_test_households"])] = "unseen_test"
    return result


def rolling_origin_cutoffs(
    start: pd.Timestamp,
    end: pd.Timestamp,
    folds: int,
    validation_fraction: float = 0.10,
) -> list[TemporalCutoffs]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    span = end - start
    cutoffs: list[TemporalCutoffs] = []
    for fold in range(folds):
        train_fraction = 0.55 + fold * (0.25 / max(folds - 1, 1))
        train_end = start + train_fraction * span
        cal_end = min(train_end + validation_fraction * span, end)
        cutoffs.append(TemporalCutoffs(train_end, cal_end))
    return cutoffs
