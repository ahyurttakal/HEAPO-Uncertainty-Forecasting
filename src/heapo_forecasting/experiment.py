from __future__ import annotations

import gc
import json
import math
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import run_audit
from .config import run_dir
from .conformal import ConditionalConformalizer
from .features import (
    load_panel,
    make_supervised_rows,
    prepare_panels,
    temperature_feature_name,
    valid_origin_positions,
)
from .metrics import add_mase_scale, compute_mase_scales, evaluate_predictions, peak_metrics
from .models import (
    ElasticNetBundle,
    LightGBMBundle,
    TabularEncoder,
    baseline_predictions,
    prediction_frame,
)
from .portfolio import run_portfolio_analysis
from .statistics import bootstrap_metric_intervals, paired_block_bootstrap
from .splits import assign_split, create_splits, rolling_origin_cutoffs
from .utils import (
    atomic_json,
    ensure_dirs,
    environment_manifest,
    frame_read,
    frame_write,
    input_fingerprint,
    seed_everything,
)


def _read_artifact(path: Path) -> pd.DataFrame:
    return frame_read(path)


def _existing_artifact(path: Path) -> bool:
    return path.exists() or path.with_suffix(".csv.gz").exists()


def _artifact_row_count(path: Path) -> int:
    """Read a stored frame's row count without materializing parquet columns."""
    if path.exists() and path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        except (ImportError, ModuleNotFoundError):
            pass
    return int(len(frame_read(path)))


def initialize_run(cfg: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    root = run_dir(cfg)
    dirs = ensure_dirs(
        root,
        [
            "audit",
            "panel",
            "splits",
            "supervised",
            "models",
            "predictions",
            "metrics",
            "figures",
            "portfolios",
            "tft",
        ],
    )
    seed_everything(int(cfg["project"]["seed"]))
    data_root = Path(cfg["data"]["root"]).expanduser().resolve()
    fingerprint_paths = [
        data_root / "meta_data/households.csv",
        data_root / "meta_data/meta_data.csv",
        data_root / "smart_meter_data/overview/smart_meter_data_15min_overview.csv",
        data_root / "weather_data/overview/weather_variables_availability.csv",
        data_root / "reports/protocols.csv",
    ]
    manifest = {
        **environment_manifest(),
        "configuration": {key: value for key, value in cfg.items() if not key.startswith("_")},
        "config_path": cfg.get("_config_path"),
        "heapo_repository_commit": "38768a8092552e9e0f0927a51f30ad31a27bc002",
        "input_fingerprints": input_fingerprint(fingerprint_paths),
    }
    atomic_json(manifest, root / "run_manifest.json")
    return root, dirs


def ensure_audit(cfg: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    path = dirs["audit"] / "household_audit.parquet"
    if _existing_artifact(path):
        return _read_artifact(path)
    return run_audit(cfg, dirs["audit"])


def ensure_panels(
    cfg: dict[str, Any], dirs: dict[str, Path], audit: pd.DataFrame
) -> pd.DataFrame:
    path = dirs["panel"] / "panel_manifest.parquet"
    if _existing_artifact(path):
        return _read_artifact(path)
    return prepare_panels(cfg, audit, dirs["panel"])


def ensure_splits(
    cfg: dict[str, Any], dirs: dict[str, Path], audit: pd.DataFrame
) -> dict[str, Any]:
    path = dirs["splits"] / "split_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return create_splits(audit, cfg, dirs["splits"])


def _subsample_positions(positions: np.ndarray, maximum: int) -> np.ndarray:
    if len(positions) <= maximum:
        return positions
    indices = np.linspace(0, len(positions) - 1, maximum, dtype=int)
    return positions[indices]


def build_supervised_shards(
    manifest: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    # v2 changes the held-out-household time windows.  Use new artifact names
    # so an older completed run is never silently reused with the revised code.
    manifest_path = output_dir / "supervised_manifest_v2.parquet"
    if _existing_artifact(manifest_path):
        return frame_read(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = set(int(v) for v in split_manifest["training_households"])
    validation_ids = set(int(v) for v in split_manifest["unseen_validation_households"])
    test_ids = set(int(v) for v in split_manifest["unseen_test_households"])
    max_eval = int(cfg["sampling"]["max_eval_origins_per_household"])
    forecast_steps = int(cfg["features"]["forecast_steps"])
    horizons = range(1, forecast_steps + 1)
    # Expanding every candidate origin to every forecast horizon can create
    # millions of rows for a single household before the later model-sampling
    # limit is applied.  Cap origins here, distribute the configured global
    # row budget evenly across training households, and sample them across the
    # full time span.  This preserves horizon balance and temporal coverage
    # while keeping preparation memory bounded.
    max_train_rows = int(cfg["sampling"]["max_train_rows"])
    max_train_origins_per_household = max(
        1,
        int(math.ceil(max_train_rows / max(len(train_ids), 1) / forecast_steps)),
    )
    records: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        household_id = int(item["household_id"])
        panel = load_panel(item)
        timestamp = cfg["data"]["timestamp"]
        if household_id in train_ids:
            train_positions = valid_origin_positions(
                panel, cfg, int(cfg["sampling"]["train_origin_stride"])
            )
            train_positions = train_positions[
                panel.iloc[train_positions][timestamp].to_numpy()
                <= pd.Timestamp(split_manifest["train_end"])
            ]
            train_positions = _subsample_positions(
                train_positions, max_train_origins_per_household
            )
            eval_positions = valid_origin_positions(
                panel, cfg, int(cfg["sampling"]["eval_origin_stride"])
            )
            eval_positions = eval_positions[
                panel.iloc[eval_positions][timestamp].to_numpy()
                > pd.Timestamp(split_manifest["train_end"])
            ]
            eval_positions = _subsample_positions(eval_positions, max_eval)
            positions = np.unique(np.concatenate([train_positions, eval_positions]))
        elif household_id in validation_ids or household_id in test_ids:
            positions = valid_origin_positions(
                panel, cfg, int(cfg["sampling"]["eval_origin_stride"])
            )
            origin_times = pd.to_datetime(
                panel.iloc[positions][timestamp], utc=True
            )
            if household_id in validation_ids:
                positions = positions[
                    (origin_times > pd.Timestamp(split_manifest["train_end"]))
                    & (origin_times <= pd.Timestamp(split_manifest["calibration_end"]))
                ]
            else:
                positions = positions[
                    origin_times > pd.Timestamp(split_manifest["calibration_end"])
                ]
            positions = _subsample_positions(positions, max_eval)
        else:
            continue
        rows = make_supervised_rows(panel, positions, horizons, cfg)
        if rows.empty:
            continue
        rows["split"] = assign_split(rows, split_manifest)
        path = frame_write(rows, output_dir / f"household_{household_id}_v2.parquet")
        counts = rows.groupby("split").size().to_dict()
        records.append(
            {
                "household_id": household_id,
                "hp_type": item["hp_type"],
                "path": str(path.resolve()),
                "rows": int(len(rows)),
                **{f"rows_{key}": int(value) for key, value in counts.items()},
            }
        )
    result = pd.DataFrame(records)
    frame_write(result, manifest_path)
    return result


def _stratified_row_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    per_horizon = max(1, int(math.ceil(maximum / frame["horizon"].nunique())))
    sampled = pd.concat(
        [
            group.sample(min(len(group), per_horizon), random_state=seed)
            for _, group in frame.groupby("horizon", sort=False)
        ],
        ignore_index=True,
    )
    return sampled.sample(min(len(sampled), maximum), random_state=seed).reset_index(drop=True)


def load_model_samples(
    supervised_manifest: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maximum = int(cfg["sampling"]["max_train_rows"])
    seed = int(cfg["project"]["seed"])
    train_counts = (
        supervised_manifest["rows_train"].fillna(0)
        if "rows_train" in supervised_manifest
        else pd.Series(0, index=supervised_manifest.index)
    )
    train_files = supervised_manifest[train_counts > 0]
    cap = max(1000, int(math.ceil(maximum / max(len(train_files), 1))))
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    for _, item in supervised_manifest.iterrows():
        frame = frame_read(Path(item["path"]))
        if (frame["split"] == "train").any():
            train_parts.append(_stratified_row_sample(frame[frame["split"] == "train"], cap, seed))
        if (frame["split"] == "unseen_validation").any():
            validation_parts.append(
                _stratified_row_sample(frame[frame["split"] == "unseen_validation"], 10000, seed + 1)
            )
        if (frame["split"] == "calibration").any():
            calibration_parts.append(
                _stratified_row_sample(frame[frame["split"] == "calibration"], 10000, seed + 2)
            )
    if not train_parts:
        raise RuntimeError("No supervised training rows; review inclusion and split settings")
    train = _stratified_row_sample(pd.concat(train_parts, ignore_index=True), maximum, seed)
    calibration = pd.concat(calibration_parts, ignore_index=True) if calibration_parts else pd.DataFrame()
    validation = pd.concat(validation_parts, ignore_index=True) if validation_parts else pd.DataFrame()
    if validation.empty:
        if calibration.empty:
            raise RuntimeError("Neither validation nor calibration rows are available")
        validation = calibration.sample(min(len(calibration), 100000), random_state=seed)
    return train, validation, calibration


def _prediction_temperature(prediction: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    result = prediction.copy()
    if "evaluation_temperature_c" in rows:
        result["temperature_c"] = rows["evaluation_temperature_c"].to_numpy(dtype=float)
    else:
        temp = temperature_feature_name(rows.columns)
        result["temperature_c"] = rows[temp].to_numpy(dtype=float) if temp else np.nan
    return result


def train_and_predict(
    supervised_manifest: pd.DataFrame,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    mase_scales: pd.DataFrame,
) -> tuple[pd.DataFrame, LightGBMBundle, pd.DataFrame, pd.DataFrame]:
    train, validation, calibration_sample = load_model_samples(supervised_manifest, cfg)
    elastic = ElasticNetBundle.fit(train, cfg)
    joblib_path = dirs["models"] / "elastic_net.joblib"
    import joblib
    joblib.dump(elastic, joblib_path)
    lightgbm = LightGBMBundle.fit(train, validation, cfg)
    lightgbm.save(dirs["models"] / "lightgbm.joblib")

    prediction_parts: list[pd.DataFrame] = []
    shap_parts: list[pd.DataFrame] = []
    for _, item in supervised_manifest.iterrows():
        rows = frame_read(Path(item["path"]))
        rows = rows[rows["split"] != "train"].reset_index(drop=True)
        if rows.empty:
            continue
        prediction_parts.append(_prediction_temperature(baseline_predictions(rows), pd.concat([rows] * 3, ignore_index=True)))
        prediction_parts.append(_prediction_temperature(elastic.predict(rows), rows))
        prediction_parts.append(_prediction_temperature(lightgbm.predict(rows), rows))
        test_rows = rows[rows["split"].isin(["seen_test", "unseen_test"])]
        if not test_rows.empty:
            shap_parts.append(test_rows.sample(min(len(test_rows), 1000), random_state=int(cfg["project"]["seed"])))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    conformal = ConditionalConformalizer.from_config(cfg).fit(
        predictions[predictions["split"] == "calibration"]
    )
    predictions = conformal.transform(predictions)
    conformal.save(dirs["models"] / "conformal.joblib")
    frame_write(
        conformal.corrections_frame(),
        dirs["models"] / "conformal_corrections.parquet",
    )
    predictions = add_mase_scale(predictions, mase_scales)
    frame_write(predictions, dirs["predictions"] / "household_predictions.parquet")
    frame_write(train, dirs["models"] / "training_sample.parquet")
    shap_rows = pd.concat(shap_parts, ignore_index=True) if shap_parts else pd.DataFrame()
    return predictions, lightgbm, shap_rows, train


def evaluate_all(
    predictions: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    training_rows: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    report_horizons = set(int(value) for value in cfg["evaluation"]["report_horizons"])
    selected = predictions[predictions["horizon"].isin(report_horizons)]
    metrics = evaluate_predictions(selected)
    raw_selected = selected.copy()
    if {"q10_raw", "q90_raw"}.issubset(raw_selected.columns):
        raw_selected["q10"] = raw_selected["q10_raw"]
        raw_selected["q90"] = raw_selected["q90_raw"]
    raw_metrics = evaluate_predictions(raw_selected)
    peak = peak_metrics(
        predictions[predictions["horizon"] <= cfg["features"]["forecast_steps"]],
        reference=training_rows,
        threshold_quantile=float(cfg["evaluation"]["peak_threshold_quantile"]),
        top_peak_fraction=float(cfg["evaluation"]["top_peak_fraction"]),
    )
    calibration_thresholds = (
        predictions[predictions["split"] == "calibration"]
        .dropna(subset=["temperature_c"])
        .groupby("HP_Type")["temperature_c"]
        .quantile(float(cfg["evaluation"]["cold_temperature_quantile"]))
        .to_dict()
    )
    test = predictions[predictions["split"].isin(["seen_test", "unseen_test"])]
    cold_parts: list[pd.DataFrame] = []
    for (split, hp_type), group in test.groupby(["split", "HP_Type"], dropna=False):
        threshold = calibration_thresholds.get(hp_type)
        if threshold is None or not np.isfinite(threshold):
            continue
        cold = group[group["temperature_c"] <= threshold].copy()
        evaluated = evaluate_predictions(cold, group_columns=("model", "split", "HP_Type", "horizon"))
        evaluated["cold_threshold_c"] = threshold
        cold_parts.append(evaluated)
    cold_metrics = pd.concat(cold_parts, ignore_index=True) if cold_parts else pd.DataFrame()
    frame_write(metrics, output_dir / "forecast_metrics.parquet")
    frame_write(raw_metrics, output_dir / "forecast_metrics_raw_intervals.parquet")
    frame_write(peak, output_dir / "peak_metrics.parquet")
    frame_write(cold_metrics, output_dir / "extreme_cold_metrics.parquet")
    statistical = cfg.get("statistics", {})
    test_selected = selected[selected["split"].isin(["seen_test", "unseen_test"])]
    repetitions = int(statistical.get("bootstrap_repetitions", 1000))
    confidence = float(statistical.get("confidence", 0.95))
    seed = int(cfg["project"]["seed"])
    intervals = bootstrap_metric_intervals(
        test_selected, repetitions=repetitions, confidence=confidence, seed=seed
    )
    configured_pairs = statistical.get("comparisons")
    pairs = [tuple(pair) for pair in configured_pairs] if configured_pairs else None
    paired = paired_block_bootstrap(
        test_selected,
        comparisons=pairs,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
    )
    frame_write(intervals, output_dir / "forecast_metric_bootstrap_intervals.parquet")
    frame_write(paired, output_dir / "paired_model_tests.parquet")
    atomic_json(
        {
            "revision": 2,
            "mase": "household daily-naive scale from pre-train panel history",
            "intervals": "hierarchical model/type/horizon/cold conformal calibration",
            "bootstrap_unit": "household-day",
            "bootstrap_repetitions": repetitions,
        },
        output_dir / "revision_v2_complete.json",
    )
    return {
        "forecast": metrics,
        "forecast_raw_intervals": raw_metrics,
        "peak": peak,
        "cold": cold_metrics,
        "bootstrap": intervals,
        "paired": paired,
    }


def _fit_mean_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[TabularEncoder, Any]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to run transfer experiments") from exc
    encoder = TabularEncoder()
    x_train = encoder.fit_transform(train)
    x_valid = encoder.transform(validation)
    options = dict(cfg["models"]["lightgbm"])
    options["n_estimators"] = max(200, int(options["n_estimators"] // 2))
    options["random_state"] = int(cfg["project"]["seed"])
    model = lgb.LGBMRegressor(objective="regression_l1", **options)
    model.fit(
        x_train,
        train["y_true"],
        eval_set=[(x_valid, validation["y_true"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return encoder, model


def run_transfer_experiments(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    mase_scales: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    scenarios = [
        ("ASHP_only", ["ASHP"]),
        ("GSHP_only", ["GSHP"]),
        ("combined", ["ASHP", "GSHP"]),
    ]
    for name, train_types in scenarios:
        train_subset = train[train["HP_Type"].isin(train_types)]
        valid_subset = validation[validation["HP_Type"].isin(train_types)]
        if train_subset.empty or valid_subset.empty:
            continue
        encoder, model = _fit_mean_lightgbm(train_subset, valid_subset, cfg)
        values = np.clip(model.predict(encoder.transform(test)), 0.0, None)
        pred = prediction_frame(test, name, values, values, values, values)
        pred["train_types"] = "+".join(train_types)
        parts.append(pred)
    predictions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if predictions.empty:
        return predictions
    predictions = add_mase_scale(predictions, mase_scales)
    metrics = evaluate_predictions(predictions, ("model", "train_types", "split", "HP_Type", "horizon"))
    frame_write(predictions, output_dir / "transfer_predictions_v2.parquet")
    frame_write(metrics, output_dir / "transfer_metrics_v2.parquet")
    return metrics


def run_rolling_validation(
    all_rows: pd.DataFrame,
    cfg: dict[str, Any],
    split_manifest: dict[str, Any],
    output_dir: Path,
    mase_scales: pd.DataFrame,
) -> pd.DataFrame:
    folds = rolling_origin_cutoffs(
        pd.Timestamp(split_manifest["global_start"]),
        pd.Timestamp(split_manifest["calibration_end"]),
        int(cfg["split"]["rolling_folds"]),
    )
    parts: list[pd.DataFrame] = []
    origin = pd.to_datetime(all_rows["origin_time"], utc=True)
    for index, fold in enumerate(folds):
        train = all_rows[origin <= fold.train_end]
        validation = all_rows[(origin > fold.train_end) & (origin <= fold.calibration_end)]
        if train.empty or validation.empty:
            continue
        train = _stratified_row_sample(train, min(len(train), 250000), int(cfg["project"]["seed"]) + index)
        validation = _stratified_row_sample(validation, min(len(validation), 100000), int(cfg["project"]["seed"]) + index)
        encoder, model = _fit_mean_lightgbm(train, validation, cfg)
        values = np.clip(model.predict(encoder.transform(validation)), 0.0, None)
        pred = prediction_frame(validation, f"rolling_fold_{index + 1}", values, values, values, values)
        pred["fold"] = index + 1
        parts.append(pred)
    predictions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not predictions.empty:
        predictions = add_mase_scale(predictions, mase_scales)
    metrics = evaluate_predictions(predictions, ("fold", "model", "HP_Type", "horizon")) if not predictions.empty else pd.DataFrame()
    frame_write(metrics, output_dir / "rolling_origin_metrics_v2.parquet")
    return metrics


def prepare_stage(cfg: dict[str, Any]) -> tuple[Path, dict[str, Path], pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    root, dirs = initialize_run(cfg)
    audit = ensure_audit(cfg, dirs)
    panels = ensure_panels(cfg, dirs, audit)
    splits = ensure_splits(cfg, dirs, audit)
    supervised = build_supervised_shards(panels, splits, cfg, dirs["supervised"])
    return root, dirs, audit, panels, splits, supervised


def run_pilot(cfg: dict[str, Any]) -> dict[str, Any]:
    root, dirs, audit, panels, splits, supervised = prepare_stage(cfg)
    mase_scales = compute_mase_scales(panels, splits, cfg)
    frame_write(mase_scales, dirs["metrics"] / "mase_scales.parquet")
    predictions, model, shap_rows, train = train_and_predict(
        supervised, cfg, dirs, mase_scales
    )
    evaluated = evaluate_all(predictions, cfg, dirs["metrics"], train)
    if not shap_rows.empty:
        from .explain import run_shap
        run_shap(model, shap_rows, dirs["figures"] / "shap", cfg)
    return {
        "run_dir": root,
        "eligible_households": int(audit["eligible"].sum()),
        "prediction_rows": int(len(predictions)),
        "metrics": evaluated,
    }


def run_full(cfg: dict[str, Any]) -> dict[str, Any]:
    root, dirs = initialize_run(cfg)
    prediction_path = dirs["predictions"] / "household_predictions.parquet"
    training_path = dirs["models"] / "training_sample.parquet"
    pilot_ready = all(
        [
            _existing_artifact(prediction_path),
            _existing_artifact(training_path),
            (dirs["models"] / "lightgbm.joblib").exists(),
            _existing_artifact(dirs["metrics"] / "forecast_metrics.parquet"),
            (dirs["metrics"] / "revision_v2_complete.json").exists(),
        ]
    )
    if pilot_ready:
        audit = ensure_audit(cfg, dirs)
        predictions = frame_read(prediction_path)
        result = {
            "run_dir": root,
            "eligible_households": int(audit["eligible"].sum()),
            "prediction_rows": int(len(predictions)),
            "metrics": {},
        }
    else:
        result = run_pilot(cfg)
        root, dirs = initialize_run(cfg)
        predictions = frame_read(prediction_path)

    splits = json.loads((dirs["splits"] / "split_manifest.json").read_text(encoding="utf-8"))
    mase_path = dirs["metrics"] / "mase_scales.parquet"
    if _existing_artifact(mase_path):
        mase_scales = frame_read(mase_path)
    else:
        panels = frame_read(dirs["panel"] / "panel_manifest.parquet")
        mase_scales = compute_mase_scales(panels, splits, cfg)
        frame_write(mase_scales, mase_path)
    transfer_ready = _existing_artifact(dirs["metrics"] / "transfer_metrics_v2.parquet")
    rolling_ready = _existing_artifact(dirs["metrics"] / "rolling_origin_metrics_v2.parquet")
    if not (transfer_ready and rolling_ready):
        supervised = frame_read(dirs["supervised"] / "supervised_manifest_v2.parquet")
        train, validation, _ = load_model_samples(supervised, cfg)
        if not transfer_ready:
            test_parts: list[pd.DataFrame] = []
            for _, item in supervised.iterrows():
                rows = frame_read(Path(item["path"]))
                test_parts.append(rows[rows["split"].isin(["seen_test", "unseen_test"])])
            test = pd.concat(test_parts, ignore_index=True)
            run_transfer_experiments(
                train, validation, test, cfg, dirs["metrics"], mase_scales
            )
            del test, test_parts
        if not rolling_ready:
            rolling_parts: list[pd.DataFrame] = [train]
            for _, item in supervised.iterrows():
                rows = frame_read(Path(item["path"]))
                rolling_parts.append(rows[rows["split"].isin(["calibration", "seen_test"])])
            run_rolling_validation(
                pd.concat(rolling_parts, ignore_index=True),
                cfg,
                splits,
                dirs["metrics"],
                mase_scales,
            )
            del rolling_parts
        del supervised, train, validation
        gc.collect()

    portfolio_path = dirs["portfolios"] / "portfolio_predictions_v2.parquet"
    peak_summary_path = dirs["portfolios"] / "peak_management_summary_v2.parquet"
    if _existing_artifact(portfolio_path) and _existing_artifact(peak_summary_path):
        portfolio_rows = _artifact_row_count(portfolio_path)
        peak_scenarios = _artifact_row_count(peak_summary_path)
    else:
        portfolios, peak_summary = run_portfolio_analysis(
            predictions, splits, cfg, dirs["portfolios"]
        )
        portfolio_rows = int(len(portfolios))
        peak_scenarios = int(len(peak_summary))
        del portfolios, peak_summary
    tft_prediction_rows = 0
    if cfg["models"]["tft"]["enabled"]:
        from .tft_model import build_common_evaluation_index, train_tft

        panels = frame_read(dirs["panel"] / "panel_manifest.parquet")
        common_index = build_common_evaluation_index(predictions, cfg)
        frame_write(common_index, dirs["tft"] / "tft_common_evaluation_index.parquet")
        train_tft(
            panels,
            splits,
            cfg,
            dirs["tft"],
            evaluation_index=common_index,
        )
        tft_predictions = frame_read(dirs["tft"] / "tft_predictions.parquet")
        temperature_lookup = predictions[predictions["model"] == "lightgbm"][[
            "Household_ID", "origin_time", "target_time", "horizon", "split", "temperature_c"
        ]].drop_duplicates([
            "Household_ID", "origin_time", "target_time", "horizon", "split"
        ])
        tft_predictions = tft_predictions.merge(
            temperature_lookup,
            on=["Household_ID", "origin_time", "target_time", "horizon", "split"],
            how="left",
            validate="one_to_one",
        )
        tft_calibration = tft_predictions[tft_predictions["split"] == "calibration"]
        if not tft_calibration.empty:
            tft_conformal = ConditionalConformalizer.from_config(cfg).fit(tft_calibration)
            tft_predictions = tft_conformal.transform(tft_predictions)
            tft_conformal.save(dirs["tft"] / "tft_conformal.joblib")
            frame_write(
                tft_conformal.corrections_frame(),
                dirs["tft"] / "tft_conformal_corrections.parquet",
            )
        else:
            tft_predictions["interval_calibrated"] = False
        tft_predictions = add_mase_scale(tft_predictions, mase_scales)
        frame_write(tft_predictions, dirs["tft"] / "tft_predictions.parquet")
        tft_metrics = evaluate_predictions(tft_predictions)
        frame_write(tft_metrics, dirs["metrics"] / "tft_metrics.parquet")
        common_keys = tft_predictions[
            tft_predictions["split"].isin(["seen_test", "unseen_test"])
        ][["Household_ID", "origin_time", "split"]].drop_duplicates()
        tabular_common = predictions.merge(
            common_keys,
            on=["Household_ID", "origin_time", "split"],
            how="inner",
            validate="many_to_one",
        )
        tft_common = tft_predictions[
            tft_predictions["split"].isin(["seen_test", "unseen_test"])
        ]
        common_predictions = pd.concat(
            [tabular_common, tft_common], ignore_index=True, sort=False
        )
        report_horizons = set(int(value) for value in cfg["evaluation"]["report_horizons"])
        common_selected = common_predictions[
            common_predictions["horizon"].isin(report_horizons)
        ]
        frame_write(common_predictions, dirs["tft"] / "common_index_predictions.parquet")
        frame_write(
            evaluate_predictions(common_selected),
            dirs["metrics"] / "common_index_model_metrics.parquet",
        )
        statistical = cfg.get("statistics", {})
        repetitions = int(statistical.get("bootstrap_repetitions", 1000))
        confidence = float(statistical.get("confidence", 0.95))
        frame_write(
            bootstrap_metric_intervals(
                common_selected,
                repetitions=repetitions,
                confidence=confidence,
                seed=int(cfg["project"]["seed"]),
            ),
            dirs["metrics"] / "common_index_bootstrap_intervals.parquet",
        )
        frame_write(
            paired_block_bootstrap(
                common_selected,
                comparisons=[("tft", model) for model in [
                    "lightgbm", "seasonal_daily", "seasonal_weekly", "persistence"
                ]],
                repetitions=repetitions,
                confidence=confidence,
                seed=int(cfg["project"]["seed"]),
            ),
            dirs["metrics"] / "common_index_tft_paired_tests.parquet",
        )
        tft_prediction_rows = int(len(tft_predictions))
    del predictions
    gc.collect()
    result.update(
        {
            "portfolio_rows": portfolio_rows,
            "peak_scenarios": peak_scenarios,
            "tft_prediction_rows": tft_prediction_rows,
        }
    )
    from .reporting import build_validation_report

    result["validation"] = build_validation_report(root, cfg)
    return result


def run_scenario_suite(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run operational and oracle-weather experiments in isolated run folders."""
    base_name = str(cfg["project"]["run_name"])
    scenarios = list(cfg.get("scenarios", {}).get("weather", ["operational", "oracle"]))
    results: dict[str, Any] = {}
    metric_parts: list[pd.DataFrame] = []
    for scenario in scenarios:
        scenario_cfg = copy.deepcopy(cfg)
        scenario_cfg["data"]["weather_scenario"] = str(scenario)
        scenario_cfg["project"]["run_name"] = f"{base_name}_{scenario}"
        if scenario == "oracle" and not bool(
            cfg.get("scenarios", {}).get("run_tft_for_oracle", False)
        ):
            scenario_cfg["models"]["tft"]["enabled"] = False
        scenario_result = run_full(scenario_cfg)
        results[str(scenario)] = {
            key: value for key, value in scenario_result.items() if key != "metrics"
        }
        scenario_root = run_dir(scenario_cfg)
        metrics = frame_read(scenario_root / "metrics" / "forecast_metrics.parquet")
        metrics["weather_scenario"] = str(scenario)
        metric_parts.append(metrics)
    combined = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    comparison_dir = run_dir(
        {
            **cfg,
            "project": {**cfg["project"], "run_name": f"{base_name}_scenario_comparison"},
        }
    )
    frame_write(combined, comparison_dir / "scenario_forecast_metrics.parquet")
    value_columns = [
        "mae", "rmse", "mase", "r2", "picp", "mean_interval_width", "winkler"
    ]
    index_columns = ["model", "split", "HP_Type", "horizon"]
    if {"operational", "oracle"}.issubset(set(combined.get("weather_scenario", []))):
        operational = combined[combined["weather_scenario"] == "operational"]
        oracle = combined[combined["weather_scenario"] == "oracle"]
        comparison = operational[index_columns + value_columns].merge(
            oracle[index_columns + value_columns],
            on=index_columns,
            how="inner",
            suffixes=("_operational", "_oracle"),
        )
        for value in value_columns:
            comparison[f"{value}_operational_minus_oracle"] = (
                comparison[f"{value}_operational"] - comparison[f"{value}_oracle"]
            )
    else:
        comparison = pd.DataFrame()
    frame_write(comparison, comparison_dir / "operational_oracle_differences.parquet")
    atomic_json(
        {"scenarios": scenarios, "run_names": [f"{base_name}_{value}" for value in scenarios]},
        comparison_dir / "scenario_manifest.json",
    )
    return {"comparison_dir": comparison_dir, "scenarios": results}
