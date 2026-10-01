from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet


NON_FEATURES = {
    "origin_time",
    "target_time",
    "y_true",
    "Household_ID",
    "HP_Type",
    "Segment_ID",
    "split",
    "baseline_persistence",
    "baseline_daily",
    "baseline_weekly",
    "conformal_temperature_c",
    "evaluation_temperature_c",
}


def fit_lightgbm_with_validation(
    model: Any,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    eval_metric: str,
    callbacks: list[Any],
) -> Any:
    """Use the installed LightGBM validation API without deprecation noise."""
    parameters = inspect.signature(model.fit).parameters
    kwargs: dict[str, Any] = {
        "eval_metric": eval_metric,
        "callbacks": callbacks,
    }
    if {"eval_X", "eval_y"}.issubset(parameters):
        kwargs["eval_X"] = x_valid
        kwargs["eval_y"] = y_valid
    else:
        kwargs["eval_set"] = [(x_valid, y_valid)]
    return model.fit(x_train, y_train, **kwargs)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in NON_FEATURES]


@dataclass
class TabularEncoder:
    feature_columns: list[str] = field(default_factory=list)
    categorical_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, feature_columns: list[str] | None = None) -> "TabularEncoder":
        self.feature_columns = feature_columns or select_feature_columns(frame)
        self.categorical_maps = {}
        self.medians = {}
        for col in self.feature_columns:
            series = frame[col]
            if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
                values = sorted(series.dropna().astype(str).unique().tolist())
                self.categorical_maps[col] = {value: idx for idx, value in enumerate(values)}
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                median = numeric.median()
                self.medians[col] = 0.0 if pd.isna(median) else float(median)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for col in self.feature_columns:
            # HEAPO households can be linked to weather stations with slightly
            # different variable availability.  Keep the training schema and
            # treat an unavailable prediction-time feature as missing data;
            # numeric values use the training median and categorical values
            # use the encoder's unknown-category code.
            series = (
                frame[col]
                if col in frame.columns
                else pd.Series(np.nan, index=frame.index, name=col)
            )
            if col in self.categorical_maps:
                mapping = self.categorical_maps[col]
                result[col] = series.astype(str).map(mapping).fillna(-1).astype("int32")
            else:
                result[col] = (
                    pd.to_numeric(series, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(self.medians[col])
                    .astype("float32")
                )
        return result

    def fit_transform(self, frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(frame, feature_columns).transform(frame)


def baseline_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model, column in {
        "persistence": "baseline_persistence",
        "seasonal_daily": "baseline_daily",
        "seasonal_weekly": "baseline_weekly",
    }.items():
        block = rows[
            ["origin_time", "target_time", "horizon", "Household_ID", "HP_Type", "y_true", "split"]
        ].copy()
        values = rows[column].to_numpy(dtype=float)
        block["model"] = model
        block["y_pred"] = values
        block["q10"] = values
        block["q50"] = values
        block["q90"] = values
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


@dataclass
class ElasticNetBundle:
    encoder: TabularEncoder
    model: ElasticNet

    @classmethod
    def fit(cls, train: pd.DataFrame, cfg: dict[str, Any]) -> "ElasticNetBundle":
        encoder = TabularEncoder()
        x = encoder.fit_transform(train)
        options = cfg["models"]["elastic_net"]
        model = ElasticNet(
            alpha=float(options["alpha"]),
            l1_ratio=float(options["l1_ratio"]),
            random_state=int(cfg["project"]["seed"]),
            max_iter=5000,
            selection="cyclic",
        )
        model.fit(x, train["y_true"].to_numpy(dtype=float))
        return cls(encoder, model)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        values = np.clip(self.model.predict(self.encoder.transform(rows)), 0.0, None)
        return prediction_frame(rows, "elastic_net", values, values, values, values)


@dataclass
class LightGBMBundle:
    encoder: TabularEncoder
    models: dict[str, Any]
    quantiles: list[float]

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        cfg: dict[str, Any],
    ) -> "LightGBMBundle":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to train LightGBM") from exc
        encoder = TabularEncoder()
        x_train = encoder.fit_transform(train)
        x_valid = encoder.transform(validation)
        y_train = train["y_true"].to_numpy(dtype=float)
        y_valid = validation["y_true"].to_numpy(dtype=float)
        options = dict(cfg["models"]["lightgbm"])
        options["random_state"] = int(cfg["project"]["seed"])
        options.setdefault("verbosity", -1)
        models: dict[str, Any] = {}
        mean_model = lgb.LGBMRegressor(objective="regression_l1", **options)
        fit_lightgbm_with_validation(
            mean_model,
            x_train,
            y_train,
            x_valid,
            y_valid,
            eval_metric="mae",
            callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
        )
        models["mean"] = mean_model
        quantiles = [float(q) for q in cfg["models"]["quantiles"]]
        for quantile in quantiles:
            model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **options)
            fit_lightgbm_with_validation(
                model,
                x_train,
                y_train,
                x_valid,
                y_valid,
                eval_metric="quantile",
                callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
            )
            models[f"q{quantile:g}"] = model
        return cls(encoder, models, quantiles)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        x = self.encoder.transform(rows)
        mean = np.clip(self.models["mean"].predict(x), 0.0, None)
        predictions = {
            q: np.clip(self.models[f"q{q:g}"].predict(x), 0.0, None)
            for q in self.quantiles
        }
        q10 = predictions[min(self.quantiles, key=lambda value: abs(value - 0.10))]
        q50 = predictions[min(self.quantiles, key=lambda value: abs(value - 0.50))]
        q90 = predictions[min(self.quantiles, key=lambda value: abs(value - 0.90))]
        stacked = np.sort(np.column_stack([q10, q50, q90]), axis=1)
        return prediction_frame(rows, "lightgbm", mean, stacked[:, 0], stacked[:, 1], stacked[:, 2])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LightGBMBundle":
        return joblib.load(path)


def prediction_frame(
    rows: pd.DataFrame,
    model: str,
    point: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> pd.DataFrame:
    result = rows[
        ["origin_time", "target_time", "horizon", "Household_ID", "HP_Type", "y_true", "split"]
    ].copy()
    result["model"] = model
    result["y_pred"] = point
    result["q10"] = q10
    result["q50"] = q50
    result["q90"] = q90
    return result
