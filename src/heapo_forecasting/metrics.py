from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


EPS = 1e-12


def pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    error = y - q
    return float(np.mean(np.maximum(level * error, (level - 1) * error)))


def winkler(y: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float = 0.20) -> float:
    width = high - low
    penalty_low = (2 / alpha) * (low - y) * (y < low)
    penalty_high = (2 / alpha) * (y - high) * (y > high)
    return float(np.mean(width + penalty_low + penalty_high))


def point_probabilistic_metrics(group: pd.DataFrame, alpha: float = 0.20) -> dict[str, float]:
    y = group["y_true"].to_numpy(dtype=float)
    pred = group["y_pred"].to_numpy(dtype=float)
    q10 = group["q10"].to_numpy(dtype=float)
    q50 = group["q50"].to_numpy(dtype=float)
    q90 = group["q90"].to_numpy(dtype=float)
    mae = mean_absolute_error(y, pred)
    if "mase_scale" in group:
        scale = group["mase_scale"].to_numpy(dtype=float)
        valid_scale = np.isfinite(scale) & (scale > 0)
        mase = (
            float(np.mean(np.abs(y[valid_scale] - pred[valid_scale]) / scale[valid_scale]))
            if valid_scale.any()
            else np.nan
        )
    else:
        mase = np.nan
    losses = [pinball(y, q10, 0.10), pinball(y, q50, 0.50), pinball(y, q90, 0.90)]
    return {
        "n": int(len(group)),
        "mae": float(mae),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "nmae_mean": float(mae / max(np.mean(np.abs(y)), EPS)),
        "mase": mase,
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "pinball_q10": losses[0],
        "pinball_q50": losses[1],
        "pinball_q90": losses[2],
        "picp": float(np.mean((y >= q10) & (y <= q90))),
        "mean_interval_width": float(np.mean(q90 - q10)),
        "normalized_interval_width": float(np.mean(q90 - q10) / max(np.mean(np.abs(y)), EPS)),
        "winkler": winkler(y, q10, q90, alpha),
        "approx_crps": float(2.0 * np.mean(losses)),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    group_columns: Iterable[str] = ("model", "split", "HP_Type", "horizon"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = list(group_columns)
    for keys, group in predictions.groupby(columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, keys))
        row.update(point_probabilistic_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _event_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, EPS),
        "false_alarm_rate": fp / max(fp + tn, 1),
    }


def peak_metrics(
    predictions: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    threshold_quantile: float = 0.95,
    top_peak_fraction: float = 0.05,
) -> pd.DataFrame:
    household_thresholds: dict[int, float] = {}
    type_thresholds: dict[str, float] = {}
    if reference is not None and not reference.empty:
        unique_reference = reference.drop_duplicates(["Household_ID", "target_time"])
        household_thresholds = (
            unique_reference.groupby("Household_ID")["y_true"]
            .quantile(threshold_quantile)
            .to_dict()
        )
        type_thresholds = (
            unique_reference.groupby("HP_Type")["y_true"]
            .quantile(threshold_quantile)
            .to_dict()
        )
    rows: list[dict[str, Any]] = []
    keys = ["model", "split", "HP_Type", "Household_ID", "origin_time"]
    for group_keys, group in predictions.groupby(keys, dropna=False):
        group = group.sort_values("horizon")
        y = group["y_true"].to_numpy(dtype=float)
        pred = group["y_pred"].to_numpy(dtype=float)
        if not len(y):
            continue
        model, split, hp_type, household, origin = group_keys
        threshold = household_thresholds.get(
            int(household), type_thresholds.get(hp_type, np.quantile(y, threshold_quantile))
        )
        event = _event_metrics(y > threshold, pred > threshold)
        top_k = max(1, int(np.ceil(len(y) * top_peak_fraction)))
        actual_top = set(np.argpartition(y, -top_k)[-top_k:].tolist())
        predicted_top = set(np.argpartition(pred, -top_k)[-top_k:].tolist())
        row = {
            "model": model,
            "split": split,
            "HP_Type": hp_type,
            "Household_ID": household,
            "origin_time": origin,
            "peak_magnitude_error": float(abs(np.max(pred) - np.max(y))),
            "peak_time_error_hours": float(abs(np.argmax(pred) - np.argmax(y)) * 0.25),
            "ramp_mae": float(np.mean(np.abs(np.diff(pred) - np.diff(y)))) if len(y) > 1 else np.nan,
            "top_peak_capture": len(actual_top & predicted_top) / top_k,
            **event,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    return (
        detail.groupby(["model", "split", "HP_Type"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )


def _seasonal_scale(
    panel: pd.DataFrame,
    target: str,
    timestamp: str,
    season_steps: int,
) -> tuple[float, int]:
    """Return a one-day naive MAE without crossing gaps or visit segments."""
    values: list[np.ndarray] = []
    groups = panel.groupby("Segment_ID", sort=False) if "Segment_ID" in panel else [(0, panel)]
    expected_delta = pd.Timedelta(minutes=15 * season_steps)
    for _, block in groups:
        ordered = block.sort_values(timestamp)
        y = pd.to_numeric(ordered[target], errors="coerce")
        time = pd.to_datetime(ordered[timestamp], utc=True)
        difference = (y - y.shift(season_steps)).abs()
        contiguous = (time - time.shift(season_steps)) == expected_delta
        clean = difference[contiguous & difference.notna()].to_numpy(dtype=float)
        if clean.size:
            values.append(clean)
    if not values:
        return np.nan, 0
    joined = np.concatenate(values)
    return float(np.mean(joined)), int(joined.size)


def compute_mase_scales(
    panel_manifest: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute household daily-naive scales using only data before train_end.

    Held-out households use their own pre-evaluation history where available.
    A scale that cannot be estimated is replaced by the corresponding heat-pump
    type median calculated only from training households, then by the global
    training-household median.  Evaluation targets are never used.
    """
    from .features import load_panel

    cutoff = pd.Timestamp(split_manifest["train_end"])
    target = cfg["data"]["target"]
    timestamp = cfg["data"]["timestamp"]
    season_steps = int(cfg.get("evaluation", {}).get("mase_season_steps", 96))
    minimum = int(cfg.get("evaluation", {}).get("mase_min_differences", 96))
    training_ids = {int(value) for value in split_manifest["training_households"]}
    rows: list[dict[str, Any]] = []
    for _, record in panel_manifest.iterrows():
        panel = load_panel(record)
        history = panel[pd.to_datetime(panel[timestamp], utc=True) <= cutoff]
        scale, count = _seasonal_scale(history, target, timestamp, season_steps)
        rows.append(
            {
                "Household_ID": int(record["household_id"]),
                "HP_Type": str(record["hp_type"]),
                "mase_scale": scale if count >= minimum and scale > 0 else np.nan,
                "mase_scale_n": count,
                "mase_scale_source": "own_pre_train_history",
                "is_training_household": int(record["household_id"]) in training_ids,
            }
        )
    result = pd.DataFrame(rows)
    training = result[
        result["is_training_household"] & result["mase_scale"].notna()
    ]
    type_fallback = training.groupby("HP_Type")["mase_scale"].median().to_dict()
    global_fallback = float(training["mase_scale"].median()) if not training.empty else np.nan
    missing = result["mase_scale"].isna()
    replacement = result["HP_Type"].map(type_fallback).fillna(global_fallback)
    result.loc[missing, "mase_scale"] = replacement[missing]
    result.loc[missing & result["HP_Type"].isin(type_fallback), "mase_scale_source"] = (
        "training_type_fallback"
    )
    result.loc[missing & ~result["HP_Type"].isin(type_fallback), "mase_scale_source"] = (
        "training_global_fallback"
    )
    return result


def add_mase_scale(predictions: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """Attach precomputed household scales without duplicating existing columns."""
    result = predictions.drop(
        columns=["mase_scale", "mase_scale_source", "mase_scale_n"], errors="ignore"
    ).copy()
    columns = ["Household_ID", "mase_scale", "mase_scale_source", "mase_scale_n"]
    available = [column for column in columns if column in scales]
    lookup = scales[available].drop_duplicates("Household_ID").copy()
    lookup["Household_ID"] = lookup["Household_ID"].astype(int)
    result["Household_ID"] = result["Household_ID"].astype(int)
    return result.merge(lookup, on="Household_ID", how="left", validate="many_to_one")
