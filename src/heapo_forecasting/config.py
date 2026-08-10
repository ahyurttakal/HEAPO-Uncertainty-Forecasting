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
    required = ["project", "data", "features", "split", "models", "evaluation"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ConfigError(f"Missing configuration sections: {missing}")
    if cfg["data"]["weather_scenario"] not in {"operational", "oracle"}:
        raise ConfigError("data.weather_scenario must be operational or oracle")
    if cfg["features"]["forecast_steps"] < max(cfg["evaluation"]["report_horizons"]):
        raise ConfigError("forecast_steps must cover every report horizon")
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
    if any(int(size) <= 0 for size in cfg.get("portfolio", {}).get("sizes", [])):
        raise ConfigError("portfolio sizes must be positive")
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
