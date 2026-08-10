from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .utils import finite_sample_quantile


def _temperature_regime(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.Series:
    if "temperature_c" not in frame:
        return pd.Series("unknown", index=frame.index, dtype="object")
    temperature = pd.to_numeric(frame["temperature_c"], errors="coerce")
    threshold = frame["HP_Type"].astype(str).map(thresholds)
    result = pd.Series("normal", index=frame.index, dtype="object")
    result.loc[temperature.isna() | threshold.isna()] = "unknown"
    result.loc[temperature.notna() & threshold.notna() & (temperature <= threshold)] = "cold"
    return result


@dataclass
class ConditionalConformalizer:
    """Hierarchical Mondrian conformal calibration for every forecast model.

    The finest cell is model × heat-pump type × horizon × temperature regime.
    Small cells fall back deterministically through broader calibration pools.
    """

    alpha: float
    min_samples: int
    cold_quantile: float = 0.05
    temperature_thresholds: dict[str, float] = field(default_factory=dict)
    tables: list[tuple[tuple[str, ...], pd.DataFrame]] = field(default_factory=list)
    pooled_correction: float = np.nan

    def fit(self, calibration: pd.DataFrame) -> "ConditionalConformalizer":
        required = {"model", "HP_Type", "horizon", "y_true", "q10", "q90"}
        missing = sorted(required - set(calibration.columns))
        if missing:
            raise ValueError(f"Calibration frame is missing columns: {missing}")
        if calibration.empty:
            raise ValueError("Calibration frame is empty")
        finite_temperature = calibration.dropna(subset=["temperature_c"]) if "temperature_c" in calibration else pd.DataFrame()
        self.temperature_thresholds = (
            finite_temperature.groupby("HP_Type")["temperature_c"]
            .quantile(self.cold_quantile)
            .astype(float)
            .to_dict()
            if not finite_temperature.empty
            else {}
        )
        working = calibration.copy()
        working["temperature_regime"] = _temperature_regime(
            working, self.temperature_thresholds
        )
        score = np.maximum(
            working["q10"].to_numpy(dtype=float) - working["y_true"].to_numpy(dtype=float),
            working["y_true"].to_numpy(dtype=float) - working["q90"].to_numpy(dtype=float),
        )
        working["_score"] = np.maximum(score, 0.0)
        self.pooled_correction = finite_sample_quantile(
            working["_score"].to_numpy(dtype=float), 1.0 - self.alpha
        )
        levels = [
            ("model", "HP_Type", "horizon", "temperature_regime"),
            ("model", "HP_Type", "horizon"),
            ("model", "horizon"),
            ("model",),
        ]
        self.tables = []
        for columns in levels:
            rows: list[dict[str, Any]] = []
            grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
            for keys, group in working.groupby(grouper, dropna=False, observed=True):
                if len(group) < self.min_samples:
                    continue
                key_values = keys if isinstance(keys, tuple) else (keys,)
                row = dict(zip(columns, key_values))
                row.update(
                    {
                        "conformal_correction": finite_sample_quantile(
                            group["_score"].to_numpy(dtype=float), 1.0 - self.alpha
                        ),
                        "calibration_n": int(len(group)),
                    }
                )
                rows.append(row)
            self.tables.append((columns, pd.DataFrame(rows)))
        return self

    @staticmethod
    def _lookup(frame: pd.DataFrame, columns: tuple[str, ...], table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if table.empty:
            return np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        if len(columns) == 1:
            correction = frame[columns[0]].map(
                table.set_index(columns[0])["conformal_correction"]
            )
            count = frame[columns[0]].map(table.set_index(columns[0])["calibration_n"])
        else:
            table_index = pd.MultiIndex.from_frame(table[list(columns)])
            query_index = pd.MultiIndex.from_frame(frame[list(columns)])
            correction_map = pd.Series(table["conformal_correction"].to_numpy(), index=table_index)
            count_map = pd.Series(table["calibration_n"].to_numpy(), index=table_index)
            correction = pd.Series(correction_map.reindex(query_index).to_numpy(), index=frame.index)
            count = pd.Series(count_map.reindex(query_index).to_numpy(), index=frame.index)
        return correction.to_numpy(dtype=float), count.to_numpy(dtype=float)

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if not self.tables or not np.isfinite(self.pooled_correction):
            raise RuntimeError("Conformalizer must be fitted first")
        result = predictions.copy()
        result["temperature_regime"] = _temperature_regime(
            result, self.temperature_thresholds
        )
        result["q10_raw"] = result["q10"]
        result["q90_raw"] = result["q90"]
        correction = np.full(len(result), np.nan)
        sample_count = np.full(len(result), np.nan)
        level_name = np.full(len(result), "pooled", dtype=object)
        for columns, table in self.tables:
            values, counts = self._lookup(result, columns, table)
            take = ~np.isfinite(correction) & np.isfinite(values)
            correction[take] = values[take]
            sample_count[take] = counts[take]
            level_name[take] = "+".join(columns)
        correction = np.where(np.isfinite(correction), correction, self.pooled_correction)
        result["q10"] = np.clip(result["q10"] - correction, 0.0, None)
        result["q90"] = result["q90"] + correction
        result["conformal_correction"] = correction
        result["conformal_calibration_n"] = sample_count
        result["conformal_level"] = level_name
        result["interval_calibrated"] = True
        return result

    def corrections_frame(self) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for columns, table in self.tables:
            if table.empty:
                continue
            block = table.copy()
            block["calibration_level"] = "+".join(columns)
            parts.append(block)
        pooled = pd.DataFrame(
            [{
                "calibration_level": "pooled",
                "conformal_correction": self.pooled_correction,
                "calibration_n": np.nan,
            }]
        )
        return pd.concat(parts + [pooled], ignore_index=True, sort=False)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ConditionalConformalizer":
        options = cfg["models"]["conformal"]
        return cls(
            alpha=float(options["alpha"]),
            min_samples=int(options.get("min_samples_per_condition", options["min_samples_per_horizon"])),
            cold_quantile=float(cfg["evaluation"]["cold_temperature_quantile"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


@dataclass
class HorizonConformalizer:
    """Backward-compatible horizon-only calibrator used by older callers/tests."""

    alpha: float
    min_samples: int
    pooled_correction: float = 0.0
    corrections: dict[int, float] | None = None

    def fit(self, calibration: pd.DataFrame) -> "HorizonConformalizer":
        score = np.maximum(
            calibration["q10"].to_numpy() - calibration["y_true"].to_numpy(),
            calibration["y_true"].to_numpy() - calibration["q90"].to_numpy(),
        )
        score = np.maximum(score, 0.0)
        self.pooled_correction = finite_sample_quantile(score, 1.0 - self.alpha)
        self.corrections = {}
        working = calibration.assign(_score=score)
        for horizon, group in working.groupby("horizon"):
            if len(group) >= self.min_samples:
                self.corrections[int(horizon)] = finite_sample_quantile(
                    group["_score"].to_numpy(), 1.0 - self.alpha
                )
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if self.corrections is None:
            raise RuntimeError("Conformalizer must be fitted first")
        result = predictions.copy()
        correction = result["horizon"].map(self.corrections).fillna(self.pooled_correction)
        result["q10_raw"] = result["q10"]
        result["q90_raw"] = result["q90"]
        result["q10"] = np.clip(result["q10"] - correction, 0.0, None)
        result["q90"] = result["q90"] + correction
        result["conformal_correction"] = correction
        result["interval_calibrated"] = True
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
