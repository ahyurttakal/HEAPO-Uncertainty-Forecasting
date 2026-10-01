from __future__ import annotations

import copy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or inconsistent."""


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Configuration not found: {path}")
    # The default configuration is package data, so configuration loading also
    # works from an installed wheel rather than only from a source checkout.
    default_config = files("heapo_forecasting").joinpath("configs/default.yaml")
    with default_config.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    cfg = _deep_update(base, override)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any], require_data: bool = False) -> None:
    required = [
        "project", "pipeline", "data", "features", "split", "sampling",
        "models", "evaluation",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ConfigError(f"Missing configuration sections: {missing}")
    if cfg["data"]["weather_scenario"] not in {"operational", "oracle"}:
        raise ConfigError("data.weather_scenario must be operational or oracle")
    if cfg["features"]["forecast_steps"] < max(cfg["evaluation"]["report_horizons"]):
        raise ConfigError("forecast_steps must cover every report horizon")
    if int(cfg["pipeline"].get("schema_version", 0)) != 1:
        raise ConfigError(
            "This release requires pipeline.schema_version: 1; use the "
            "configuration supplied with this repository."
        )
    clock_hours = [int(value) for value in cfg["sampling"].get("eval_clock_hours", [])]
    if not clock_hours or len(clock_hours) != len(set(clock_hours)):
        raise ConfigError("sampling.eval_clock_hours must contain unique local hours")
    if any(value < 0 or value > 23 for value in clock_hours):
        raise ConfigError("sampling.eval_clock_hours values must be between 0 and 23")
    if int(cfg["sampling"].get("max_eval_origins_per_household", 0)) < len(clock_hours):
        raise ConfigError(
            "max_eval_origins_per_household must be at least the number of clock strata"
        )
    rolling_monitor_fraction = float(
        cfg["sampling"].get("rolling_monitor_fraction", 0.10)
    )
    if not 0 < rolling_monitor_fraction < 0.5:
        raise ConfigError("sampling.rolling_monitor_fraction must be in (0, 0.5)")
    if int(cfg["evaluation"].get("mase_season_steps", 96)) <= 0:
        raise ConfigError("evaluation.mase_season_steps must be positive")
    fractions = (
        cfg["split"]["temporal_train_fraction"],
        cfg["split"]["temporal_calibration_fraction"],
    )
    if fractions[0] <= 0 or fractions[1] <= 0 or sum(fractions) >= 1:
        raise ConfigError("Temporal fractions must be positive and sum to less than one")
    alpha = float(cfg["models"]["conformal"]["alpha"])
    if not 0 < alpha < 1:
        raise ConfigError("models.conformal.alpha must be between zero and one")
    early_stop_fraction = float(cfg["models"]["tft"].get("early_stop_fraction", 0.10))
    if not 0 < early_stop_fraction < 0.5:
        raise ConfigError("models.tft.early_stop_fraction must be in (0, 0.5)")
    if bool(cfg["models"]["tft"].get("reuse_existing_checkpoint", False)):
        raise ConfigError(
            "The canonical analysis requires reuse_existing_checkpoint: false to prevent reuse "
            "of checkpoints trained with an incompatible validation boundary."
        )
    if bool(cfg["models"]["tft"].get("resume_training_from_checkpoint", False)):
        raise ConfigError(
            "The canonical analysis requires resume_training_from_checkpoint: false."
        )
    if any(int(size) <= 0 for size in cfg.get("portfolio", {}).get("sizes", [])):
        raise ConfigError("portfolio sizes must be positive")
    peak_thresholds = cfg.get("portfolio", {}).get(
        "peak_threshold_quantiles",
        [cfg["evaluation"]["peak_threshold_quantile"]],
    )
    if any(not 0 < float(value) < 1 for value in peak_thresholds):
        raise ConfigError("portfolio peak threshold quantiles must be in (0, 1)")
    confidence = float(cfg.get("statistics", {}).get("confidence", 0.95))
    if not 0 < confidence < 1:
        raise ConfigError("statistics.confidence must be between zero and one")
    repetitions = int(cfg.get("statistics", {}).get("bootstrap_repetitions", 1000))
    if repetitions <= 0:
        raise ConfigError("statistics.bootstrap_repetitions must be positive")
    testing_alpha = float(cfg.get("statistics", {}).get("multiple_testing_alpha", 0.05))
    if not 0 < testing_alpha < 1:
        raise ConfigError("statistics.multiple_testing_alpha must be between zero and one")
    minimum_households = int(
        cfg.get("statistics", {}).get("minimum_households_for_confirmatory", 20)
    )
    if minimum_households <= 1:
        raise ConfigError("minimum households for confirmatory inference must exceed one")
    sensitivity = cfg.get("statistics", {}).get(
        "portfolio_sensitivity_thresholds", [20, 30, 50]
    )
    if any(int(value) <= 0 for value in sensitivity):
        raise ConfigError("portfolio sensitivity thresholds must be positive")
    if require_data:
        root = Path(cfg["data"]["root"]).expanduser()
        if not root.is_dir():
            raise ConfigError(f"HEAPO data root does not exist: {root}")


def run_dir(cfg: dict[str, Any]) -> Path:
    base = Path(cfg["project"]["output_root"]).expanduser()
    if not base.is_absolute():
        base = Path(cfg["_config_path"]).parent.parent / base
    result = (base / cfg["project"]["run_name"]).resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result
