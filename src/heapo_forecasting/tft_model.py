from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import load_panel
from .utils import frame_write


def _continuous_sequences(panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    target = cfg["data"]["target"]
    timestamp = cfg["data"]["timestamp"]
    valid = panel[target].notna()
    gap_group = (~valid).cumsum()
    frame = panel[valid].copy()
    frame["Sequence_ID"] = (
        frame["Household_ID"].astype(str)
        + "_"
        + frame["Segment_ID"].astype(str)
        + "_"
        + gap_group[valid].astype(str)
    )
    timestamps = pd.to_datetime(frame[timestamp], utc=True)
    frame["time_idx"] = (timestamps.astype("int64") // pd.Timedelta(minutes=15).value).astype(int)
    minimum = (
        int(cfg["models"]["tft"]["max_encoder_length"])
        + int(cfg["models"]["tft"]["max_prediction_length"])
    )
    sizes = frame.groupby("Sequence_ID")["time_idx"].transform("size")
    frame = frame[sizes >= minimum].copy()
    frame["HP_Type"] = frame["HP_Type"].fillna("unknown").astype(str)
    for col in frame.columns:
        if col.startswith("static__"):
            if pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
            else:
                frame[col] = frame[col].fillna("unknown").astype(str)
    weather = [col for col in frame if col.startswith("weather_operational__")]
    for col in weather:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[col] = frame.groupby("Sequence_ID")[col].ffill().bfill().fillna(frame[col].median())
    frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True)
    core = [timestamp, target, "Household_ID", "HP_Type", "Segment_ID", "Sequence_ID", "time_idx"]
    features = [
        col
        for col in frame
        if col.startswith("cal_")
        or col.startswith("static__")
        or col.startswith("weather_operational__")
        or (cfg["data"]["weather_scenario"] == "oracle" and "hourly" in col.lower())
    ]
    keep = list(dict.fromkeys([col for col in core + features if col in frame]))
    return frame[keep].copy()


def _bounded_sequence(
    frame: pd.DataFrame,
    timestamp: str,
    end_time: pd.Timestamp | None,
    maximum_rows: int,
    minimum_rows: int,
    stage: str,
) -> pd.DataFrame:
    """Return the latest eligible continuous sequence ending by a cutoff."""
    eligible = frame if end_time is None else frame[frame[timestamp] <= end_time]
    if eligible.empty:
        return pd.DataFrame()
    sequence_order = (
        eligible.groupby("Sequence_ID", observed=True)[timestamp]
        .max()
        .sort_values(ascending=False)
        .index
    )
    for sequence_id in sequence_order:
        block = eligible[eligible["Sequence_ID"] == sequence_id].sort_values(timestamp)
        if len(block) < minimum_rows:
            continue
        block = block.tail(max(maximum_rows, minimum_rows)).copy()
        block["Sequence_ID"] = block["Sequence_ID"].astype(str) + f"__{stage}"
        return block.reset_index(drop=True)
    return pd.DataFrame()


def build_common_evaluation_index(
    predictions: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Select deterministic origin times shared by tabular models and TFT."""
    reference_model = str(cfg["models"]["tft"].get("reference_model", "lightgbm"))
    splits = ["calibration", "seen_test", "unseen_test"]
    reference = predictions[
        (predictions["model"] == reference_model)
        & (predictions["horizon"] == 1)
        & predictions["split"].isin(splits)
    ][["Household_ID", "HP_Type", "split", "origin_time"]].drop_duplicates()
    maximum = int(cfg["models"]["tft"].get("max_eval_origins_per_household", 12))
    parts: list[pd.DataFrame] = []
    for _, group in reference.groupby(["Household_ID", "split"], sort=False):
        ordered = group.sort_values("origin_time")
        if len(ordered) > maximum:
            positions = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
            ordered = ordered.iloc[positions]
        parts.append(ordered)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=reference.columns)
    result["origin_time"] = pd.to_datetime(result["origin_time"], utc=True)
    return result


def _evaluation_sequences(
    continuous: pd.DataFrame,
    origins: pd.DataFrame,
    timestamp: str,
    encoder_length: int,
    prediction_length: int,
) -> pd.DataFrame:
    """Build one exact encoder/decoder sequence for every requested origin."""
    if origins.empty or continuous.empty:
        return pd.DataFrame()
    ordered = continuous.sort_values(timestamp).reset_index(drop=True)
    time_to_position = pd.Series(
        np.arange(len(ordered), dtype=int),
        index=pd.DatetimeIndex(pd.to_datetime(ordered[timestamp], utc=True)),
    )
    parts: list[pd.DataFrame] = []
    for _, origin in origins.drop_duplicates("origin_time").iterrows():
        origin_time = pd.Timestamp(origin["origin_time"])
        if origin_time not in time_to_position.index:
            continue
        position_value = time_to_position.loc[origin_time]
        position = int(position_value.iloc[-1] if isinstance(position_value, pd.Series) else position_value)
        start = position - encoder_length + 1
        stop = position + prediction_length + 1
        if start < 0 or stop > len(ordered):
            continue
        block = ordered.iloc[start:stop].copy()
        if len(block) != encoder_length + prediction_length:
            continue
        if block["Sequence_ID"].nunique() != 1:
            continue
        expected = pd.date_range(
            block[timestamp].iloc[0], periods=len(block), freq="15min"
        )
        if not pd.DatetimeIndex(block[timestamp]).equals(expected):
            continue
        household = int(block["Household_ID"].iloc[0])
        token = str(origin_time.value)
        block["Sequence_ID"] = f"eval__{household}__{origin['split']}__{token}"
        parts.append(block)
    return pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()


def _harmonize_tft_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align household-specific weather schemas using training-only statistics."""
    core = {
        "Timestamp", "kWh_received_HeatPump", "Household_ID", "HP_Type",
        "Segment_ID", "Sequence_ID", "time_idx",
    }
    feature_columns = [col for col in train.columns if col not in core]
    categorical = {
        col
        for col in feature_columns
        if col.startswith("static__")
        and (
            pd.api.types.is_object_dtype(train[col])
            or isinstance(train[col].dtype, pd.CategoricalDtype)
        )
    }
    usable: list[str] = []
    medians: dict[str, float] = {}
    for col in feature_columns:
        if col in categorical:
            usable.append(col)
            continue
        numeric = pd.to_numeric(train[col], errors="coerce")
        median = numeric.median()
        if pd.isna(median):
            continue
        usable.append(col)
        medians[col] = float(median)

    frames: list[pd.DataFrame] = []
    base_columns = [
        col
        for col in [
            "Timestamp", "kWh_received_HeatPump", "Household_ID", "HP_Type",
            "Segment_ID", "Sequence_ID", "time_idx",
        ]
        if col in train.columns
    ]
    for source in [train, validation, test]:
        aligned = source.copy()
        aligned["HP_Type"] = aligned["HP_Type"].fillna("unknown").astype(str)
        aligned["Sequence_ID"] = aligned["Sequence_ID"].astype(str)
        for col in usable:
            if col not in aligned:
                aligned[col] = np.nan
            if col in categorical:
                aligned[col] = aligned[col].fillna("unknown").astype(str)
            else:
                aligned[col] = (
                    pd.to_numeric(aligned[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(medians[col])
                    .astype("float32")
                )
        frames.append(aligned[base_columns + usable].copy())
    return frames[0], frames[1], frames[2]


def build_tft_frames(
    manifest: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
    evaluation_index: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    training_ids = set(int(value) for value in split_manifest["training_households"])
    all_ids = training_ids | set(int(value) for value in split_manifest["unseen_validation_households"]) | set(
        int(value) for value in split_manifest["unseen_test_households"]
    )
    selected = manifest[manifest["household_id"].astype(int).isin(all_ids)]
    timestamp = cfg["data"]["timestamp"]
    options = cfg["models"]["tft"]
    minimum = int(options["max_encoder_length"]) + int(options["max_prediction_length"])
    max_train = int(options.get("max_train_rows_per_household", 5000))
    max_eval = int(options.get("max_eval_rows_per_household", minimum))
    train_end = pd.Timestamp(split_manifest["train_end"])
    calibration_end = pd.Timestamp(split_manifest["calibration_end"])
    for _, record in selected.iterrows():
        household_id = int(record["household_id"])
        continuous = _continuous_sequences(load_panel(record), cfg)
        if household_id in training_ids:
            train_block = _bounded_sequence(
                continuous, timestamp, train_end, max_train, minimum, "train"
            )
            validation_block = _bounded_sequence(
                continuous, timestamp, calibration_end, max_eval, minimum, "validation"
            )
            if not train_block.empty:
                train_parts.append(train_block)
            if not validation_block.empty:
                validation_parts.append(validation_block)
        if evaluation_index is not None:
            requested = evaluation_index[
                evaluation_index["Household_ID"].astype(int) == household_id
            ]
            test_block = _evaluation_sequences(
                continuous,
                requested,
                timestamp,
                int(options["max_encoder_length"]),
                int(options["max_prediction_length"]),
            )
        else:
            test_block = _bounded_sequence(
                continuous, timestamp, None, max_eval, minimum, "test"
            )
        if not test_block.empty:
            test_parts.append(test_block)

    train = pd.concat(train_parts, ignore_index=True, copy=False) if train_parts else pd.DataFrame()
    validation = (
        pd.concat(validation_parts, ignore_index=True, copy=False)
        if validation_parts else pd.DataFrame()
    )
    test = pd.concat(test_parts, ignore_index=True, copy=False) if test_parts else pd.DataFrame()
    if train.empty or validation.empty or test.empty:
        return train, validation, test
    return _harmonize_tft_frames(train, validation, test)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prediction_components(predicted: Any) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Normalize pytorch-forecasting 1.x Prediction API variants."""
    output = getattr(predicted, "output", predicted[0] if isinstance(predicted, tuple) else predicted)
    y_value = getattr(predicted, "y", predicted[2] if isinstance(predicted, tuple) and len(predicted) > 2 else None)
    index = getattr(predicted, "index", predicted[1] if isinstance(predicted, tuple) and len(predicted) > 1 else None)
    if y_value is None or index is None:
        raise RuntimeError("TFT prediction result did not include return_y and return_index")
    if isinstance(y_value, (tuple, list)):
        y_value = y_value[0]
    return _as_numpy(output), _as_numpy(y_value), pd.DataFrame(index).reset_index(drop=True)


def train_tft(
    manifest: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
    evaluation_index: pd.DataFrame | None = None,
) -> Path:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import EncoderNormalizer, NaNLabelEncoder
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:
        raise RuntimeError("Install requirements-tft.txt to run TFT") from exc

    options = cfg["models"]["tft"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, test_frame = build_tft_frames(
        manifest, split_manifest, cfg, evaluation_index=evaluation_index
    )
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise RuntimeError("No continuous training sequences are long enough for TFT")
    timestamp = cfg["data"]["timestamp"]
    target = cfg["data"]["target"]

    known_reals = [col for col in train_frame if col.startswith("cal_")]
    scenario_prefix = "weather_operational__" if cfg["data"]["weather_scenario"] == "operational" else ""
    weather = [
        col for col in train_frame
        if (col.startswith(scenario_prefix) if scenario_prefix else "hourly" in col.lower())
        and pd.api.types.is_numeric_dtype(train_frame[col])
    ]
    known_reals += [col for col in weather if col not in known_reals]
    static_reals = [
        col for col in train_frame if col.startswith("static__") and pd.api.types.is_numeric_dtype(train_frame[col])
    ]
    static_categoricals = ["HP_Type"] + [
        col for col in train_frame if col.startswith("static__") and not pd.api.types.is_numeric_dtype(train_frame[col])
    ]
    encoder_frame = pd.concat(
        [train_frame, validation_frame, test_frame], ignore_index=True, copy=False
    )
    categorical_encoders = {}
    for name in ["Sequence_ID"] + static_categoricals:
        encoder = NaNLabelEncoder(add_nan=True)
        encoder.fit(encoder_frame[name].fillna("unknown").astype(str))
        categorical_encoders[name] = encoder
    dataset = TimeSeriesDataSet(
        train_frame,
        time_idx="time_idx",
        target=target,
        group_ids=["Sequence_ID"],
        max_encoder_length=int(options["max_encoder_length"]),
        max_prediction_length=int(options["max_prediction_length"]),
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=["time_idx"] + known_reals,
        time_varying_unknown_reals=[target],
        # Normalize each sample from its own encoder history.  This supports
        # cold-start households and avoids coupling the target normalizer to a
        # static category that is not one of TimeSeriesDataSet's group_ids.
        target_normalizer=EncoderNormalizer(transformation=None),
        categorical_encoders=categorical_encoders,
        allow_missing_timesteps=False,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        dataset, validation_frame, predict=True, stop_randomization=True
    )
    train_loader = dataset.to_dataloader(train=True, batch_size=int(options["batch_size"]), num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=int(options["batch_size"]), num_workers=0)
    existing_checkpoints = sorted(output_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    reuse_existing = bool(options.get("reuse_existing_checkpoint", True)) and bool(existing_checkpoints)
    best_score = np.nan
    if reuse_existing:
        best = existing_checkpoints[-1]
    else:
        checkpoint = ModelCheckpoint(
            dirpath=output_dir,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        trainer = pl.Trainer(
            max_epochs=int(options["max_epochs"]),
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[
                EarlyStopping("val_loss", patience=int(options["patience"]), mode="min"),
                checkpoint,
            ],
            logger=True,
            deterministic=True,
        )
        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=float(options.get("learning_rate", 0.03)),
            hidden_size=int(options["hidden_size"]),
            attention_head_size=int(options["attention_head_size"]),
            hidden_continuous_size=int(options["hidden_continuous_size"]),
            dropout=float(options["dropout"]),
            loss=QuantileLoss(quantiles=cfg["models"]["quantiles"]),
            reduce_on_plateau_patience=3,
        )
        resume_path = (
            str(existing_checkpoints[-1])
            if existing_checkpoints and bool(options.get("resume_training_from_checkpoint", False))
            else None
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_path,
        )
        best = Path(checkpoint.best_model_path)
        if checkpoint.best_model_score is not None:
            best_score = float(checkpoint.best_model_score)
    pd.DataFrame(
        [{
            "best_checkpoint": str(best),
            "best_val_loss": best_score,
            "checkpoint_reused": reuse_existing,
        }]
    ).to_csv(output_dir / "tft_training_summary.csv", index=False)
    best_model = TemporalFusionTransformer.load_from_checkpoint(str(best))
    test_dataset = TimeSeriesDataSet.from_dataset(
        dataset,
        test_frame,
        predict=True,
        stop_randomization=True,
    )
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=int(options["batch_size"]), num_workers=0
    )
    predicted = best_model.predict(
        test_loader,
        mode="quantiles",
        return_index=True,
        return_y=True,
        trainer_kwargs={"accelerator": "auto", "devices": 1},
    )
    quantile_values, actual, index = _prediction_components(predicted)
    quantiles = [float(value) for value in cfg["models"]["quantiles"]]
    q10_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.10)))
    q50_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.50)))
    q90_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.90)))
    metadata: dict[str, dict[str, Any]] = {}
    prediction_length = int(options["max_prediction_length"])
    for sequence, block in test_frame.groupby("Sequence_ID", sort=False):
        ordered = block.sort_values(timestamp)
        household = int(ordered["Household_ID"].iloc[0])
        origin_time = pd.Timestamp(ordered[timestamp].iloc[-prediction_length - 1])
        if str(sequence).startswith("eval__"):
            split = str(sequence).split("__", 3)[2]
        else:
            split = (
                "unseen_test"
                if household in split_manifest["unseen_test_households"]
                else "unseen_validation"
                if household in split_manifest["unseen_validation_households"]
                else "seen_test"
            )
        metadata[str(sequence)] = {
            "Household_ID": household,
            "HP_Type": str(ordered["HP_Type"].iloc[0]),
            "split": split,
            "origin_time": origin_time,
        }
    tidy: list[dict[str, Any]] = []
    for sample_idx in range(quantile_values.shape[0]):
        sequence = str(index.loc[sample_idx, "Sequence_ID"])
        meta = metadata.get(sequence)
        if meta is None:
            raise RuntimeError(f"TFT returned unknown sequence: {sequence}")
        household = int(meta["Household_ID"])
        first_idx_col = "time_idx_first_prediction" if "time_idx_first_prediction" in index else "time_idx"
        first_time_idx = int(index.loc[sample_idx, first_idx_col])
        split = str(meta["split"])
        hp_type = str(meta["HP_Type"])
        for horizon in range(1, quantile_values.shape[1] + 1):
            time_idx = first_time_idx + horizon - 1
            target_time = pd.Timestamp(time_idx * pd.Timedelta(minutes=15).value, tz="UTC")
            tidy.append(
                {
                    "origin_time": meta["origin_time"],
                    "target_time": target_time,
                    "horizon": horizon,
                    "Household_ID": household,
                    "HP_Type": hp_type,
                    "y_true": float(actual[sample_idx, horizon - 1]),
                    "split": split,
                    "model": "tft",
                    "y_pred": float(quantile_values[sample_idx, horizon - 1, q50_idx]),
                    "q10": float(quantile_values[sample_idx, horizon - 1, q10_idx]),
                    "q50": float(quantile_values[sample_idx, horizon - 1, q50_idx]),
                    "q90": float(quantile_values[sample_idx, horizon - 1, q90_idx]),
                }
            )
    frame_write(pd.DataFrame(tidy), output_dir / "tft_predictions.parquet")
    return best
