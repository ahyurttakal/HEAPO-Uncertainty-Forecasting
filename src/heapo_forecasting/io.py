from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class HEAPODataError(RuntimeError):
    """Raised when the HEAPO on-disk structure is not usable."""


def read_semicolon(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise HEAPODataError(f"Missing HEAPO file: {path}")
    return pd.read_csv(path, sep=";", low_memory=False, **kwargs)


class HEAPOAdapter:
    """Thin reader for the file layout documented by the official loader."""

    REQUIRED = (
        "meta_data/households.csv",
        "meta_data/meta_data.csv",
        "smart_meter_data/overview/smart_meter_data_15min_overview.csv",
        "reports/protocols.csv",
    )

    def __init__(self, root: str | Path, cfg: dict[str, Any]):
        self.root = Path(root).expanduser().resolve()
        self.cfg = cfg
        self.timestamp = cfg["data"]["timestamp"]
        self.household_id = cfg["data"]["household_id"]
        self.target = cfg["data"]["target"]
        self.timezone = cfg["data"]["timezone"]
        self.validate_layout()
        self._metadata: pd.DataFrame | None = None

    def validate_layout(self) -> None:
        if not self.root.is_dir():
            raise HEAPODataError(f"Data directory does not exist: {self.root}")
        missing = [name for name in self.REQUIRED if not (self.root / name).exists()]
        if missing:
            raise HEAPODataError(f"Incomplete HEAPO extraction; missing: {missing}")

    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            households = read_semicolon(self.root / "meta_data/households.csv")
            metadata = read_semicolon(self.root / "meta_data/meta_data.csv")
            merged = households.merge(metadata, on=self.household_id, how="outer")
            merged[self.household_id] = pd.to_numeric(
                merged[self.household_id], errors="coerce"
            ).astype("Int64")
            self._metadata = merged.sort_values(self.household_id).reset_index(drop=True)
        return self._metadata.copy()

    def meter_overview(self) -> pd.DataFrame:
        return read_semicolon(
            self.root
            / "smart_meter_data/overview/smart_meter_data_15min_overview.csv"
        )

    def protocols(self) -> pd.DataFrame:
        frame = read_semicolon(self.root / "reports/protocols.csv")
        if "Visit_Date" in frame:
            frame["Visit_Date"] = pd.to_datetime(frame["Visit_Date"], utc=True, errors="coerce")
        return frame

    def meter_path(self, household_id: int) -> Path:
        return self.root / "smart_meter_data/15min" / f"{int(household_id)}.csv"

    def weather_path(self, weather_id: Any) -> Path:
        candidates = [self.root / "weather_data/hourly" / f"{weather_id}.csv"]
        if pd.notna(weather_id):
            try:
                candidates.append(
                    self.root / "weather_data/hourly" / f"{int(float(weather_id))}.csv"
                )
            except (TypeError, ValueError):
                pass
        for path in candidates:
            if path.exists():
                return path
        raise HEAPODataError(f"No hourly weather file for Weather_ID={weather_id}")

    def load_meter(self, household_id: int) -> pd.DataFrame:
        frame = read_semicolon(self.meter_path(household_id))
        if self.timestamp not in frame:
            raise HEAPODataError(f"{household_id}: missing {self.timestamp}")
        frame[self.timestamp] = pd.to_datetime(frame[self.timestamp], utc=True, errors="coerce")
        frame = frame.dropna(subset=[self.timestamp]).sort_values(self.timestamp)
        frame = frame.drop_duplicates(self.timestamp, keep="last")
        return frame.reset_index(drop=True)

    def load_weather_for_household(self, household_id: int) -> pd.DataFrame:
        meta = self.metadata()
        row = meta.loc[meta[self.household_id] == int(household_id)]
        if row.empty or "Weather_ID" not in row or pd.isna(row.iloc[0]["Weather_ID"]):
            return pd.DataFrame(columns=[self.timestamp])
        frame = read_semicolon(self.weather_path(row.iloc[0]["Weather_ID"]))
        frame[self.timestamp] = pd.to_datetime(frame[self.timestamp], utc=True, errors="coerce")
        return (
            frame.dropna(subset=[self.timestamp])
            .sort_values(self.timestamp)
            .drop_duplicates(self.timestamp, keep="last")
            .reset_index(drop=True)
        )

    def static_row(self, household_id: int) -> dict[str, Any]:
        frame = self.metadata()
        row = frame.loc[frame[self.household_id] == int(household_id)]
        return {} if row.empty else row.iloc[0].to_dict()

    def normalized_hp_type(self, household_id: int) -> str | None:
        row = self.static_row(household_id)
        raw = row.get(self.cfg["data"]["hp_type_column"])
        return self.cfg["data"]["allowed_hp_types"].get(str(raw))

    def visit_dates(self, household_id: int) -> list[pd.Timestamp]:
        frame = self.protocols()
        if self.household_id not in frame or "Visit_Date" not in frame:
            return []
        values = frame.loc[frame[self.household_id] == int(household_id), "Visit_Date"]
        return [value for value in values.dropna().sort_values().tolist()]


def infer_temperature_column(columns: list[str]) -> str | None:
    priority = (
        "Temperature_mean_hourly",
        "Temperature_avg_hourly",
        "Temperature_hourly",
    )
    for name in priority:
        if name in columns:
            return name
    candidates = [
        name
        for name in columns
        if "temperature" in name.lower()
        and "hourly" in name.lower()
        and not any(token in name.lower() for token in ("dew", "min", "max"))
    ]
    return candidates[0] if candidates else None


def merge_weather_15min(
    meter: pd.DataFrame,
    weather: pd.DataFrame,
    timestamp: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Interpolate only the three 15-minute slots between observed hours."""
    if weather.empty:
        return meter.copy(), []
    weather_cols = [
        col
        for col in weather.columns
        if col not in {timestamp, "Date", "Weather_ID"}
        and pd.api.types.is_numeric_dtype(weather[col])
    ]
    left = meter.copy().set_index(timestamp)
    right = weather[[timestamp] + weather_cols].copy().set_index(timestamp)
    union = left.index.union(right.index).sort_values()
    right = right.reindex(union).interpolate(
        method="time", limit=3, limit_area="inside", limit_direction="forward"
    )
    right = right.reindex(left.index)
    overlap = [col for col in weather_cols if col in left.columns]
    if overlap:
        warnings.warn(f"Dropping duplicate weather columns from meter input: {overlap}")
        left = left.drop(columns=overlap)
    merged = left.join(right, how="left").reset_index()
    return merged, weather_cols


def regularize_target(
    frame: pd.DataFrame,
    timestamp: str,
    target: str,
    short_gap_steps: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in frame:
        raise HEAPODataError(f"Target column not found: {target}")
    frame = frame.sort_values(timestamp).drop_duplicates(timestamp, keep="last").copy()
    full_index = pd.date_range(
        frame[timestamp].min(), frame[timestamp].max(), freq="15min", tz="UTC"
    )
    frame = frame.set_index(timestamp).reindex(full_index)
    frame.index.name = timestamp
    original_missing = frame[target].isna()
    if short_gap_steps > 0:
        interpolated = frame[target].interpolate(
            method="time",
            limit=short_gap_steps,
            limit_area="inside",
            limit_direction="both",
        )
        run_id = original_missing.ne(original_missing.shift()).cumsum()
        run_size = original_missing.groupby(run_id).transform("sum")
        fillable = original_missing & (run_size <= short_gap_steps)
        frame.loc[fillable, target] = interpolated.loc[fillable]
    frame["Target_OriginallyMissing"] = original_missing.astype("int8")
    return frame.reset_index(), original_missing
