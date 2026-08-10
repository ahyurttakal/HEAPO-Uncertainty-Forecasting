from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io import HEAPOAdapter, infer_temperature_column, merge_weather_15min, regularize_target
from .utils import frame_read, frame_write


STATIC_CANDIDATES = (
    "Survey_Building_LivingArea",
    "Survey_Building_Residents",
    "Survey_HeatDistribution_System_Radiator",
    "Survey_HeatDistribution_System_FloorHeating",
    "Survey_DHW_Production_ByHeatPump",
    "Survey_DHW_Production_ByElectricWaterHeater",
    "Survey_DHW_Production_BySolar",
    "Installation_HasPVSystem",
    "Survey_Installation_HasElectricVehicle",
    "Survey_HeatPump_Installation_Year",
    "Survey_HeatPump_Installation_HeatingCapacity",
    "Survey_HeatPump_Installation_Type",
    "Group",
)


def _calendar_features(timestamps: pd.Series, timezone: str) -> pd.DataFrame:
    local = timestamps.dt.tz_convert(timezone)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    day_of_year = local.dt.dayofyear
    day_of_week = local.dt.dayofweek
    result = pd.DataFrame(index=timestamps.index)
    result["cal_hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    result["cal_hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    result["cal_dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["cal_dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    result["cal_doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["cal_doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["cal_weekend"] = (day_of_week >= 5).astype("int8")
    result["cal_month"] = local.dt.month.astype("int8")
    result["cal_year"] = local.dt.year.astype("int16")
    try:
        import holidays

        years = sorted(local.dt.year.dropna().unique().tolist())
        zurich = holidays.country_holidays("CH", subdiv="ZH", years=years)
        result["cal_public_holiday"] = local.dt.date.isin(zurich).astype("int8")
    except ImportError:
        result["cal_public_holiday"] = 0
    return result


def _segment_by_visits(timestamps: pd.Series, visits: list[pd.Timestamp]) -> pd.Series:
    result = np.zeros(len(timestamps), dtype=np.int16)
    for visit in visits:
        result += (timestamps >= visit).to_numpy(dtype=np.int16)
    return pd.Series(result, index=timestamps.index, name="Segment_ID")


def prepare_household_panel(adapter: HEAPOAdapter, household_id: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = adapter.cfg
    meter = adapter.load_meter(household_id)
    weather = adapter.load_weather_for_household(household_id)
    merged, weather_cols = merge_weather_15min(meter, weather, adapter.timestamp)
    regular, _ = regularize_target(
        merged,
        adapter.timestamp,
        adapter.target,
        cfg["data"]["short_gap_steps"],
    )
    static = adapter.static_row(household_id)
    hp_type = adapter.normalized_hp_type(household_id)
    regular[adapter.household_id] = int(household_id)
    regular["HP_Type"] = hp_type
    regular["Segment_ID"] = _segment_by_visits(
        regular[adapter.timestamp], adapter.visit_dates(household_id)
    )
    regular = pd.concat(
        [regular, _calendar_features(regular[adapter.timestamp], adapter.timezone)], axis=1
    )
    numeric_weather: list[str] = []
    for col in weather_cols:
        regular[col] = pd.to_numeric(regular[col], errors="coerce")
        if regular[col].notna().any():
            numeric_weather.append(col)
            regular[f"weather_operational__{col}"] = regular[col].shift(96)
            exact_hour = regular[adapter.timestamp].dt.minute.eq(0)
            regular[f"weather_causal__{col}"] = regular[col].where(exact_hour).ffill()

    for name in STATIC_CANDIDATES:
        if name in static:
            regular[f"static__{name}"] = static[name]
    temperature = infer_temperature_column(numeric_weather)
    metadata = {
        "household_id": int(household_id),
        "hp_type": hp_type,
        "rows": int(len(regular)),
        "start_utc": regular[adapter.timestamp].min().isoformat(),
        "end_utc": regular[adapter.timestamp].max().isoformat(),
        "weather_columns": numeric_weather,
        "temperature_column": temperature,
        "static_columns": [col for col in regular if col.startswith("static__")],
    }
    return regular, metadata


def prepare_panels(
    cfg: dict[str, Any], audit: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    adapter = HEAPOAdapter(cfg["data"]["root"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    eligible = audit.loc[audit["eligible"], "household_id"].astype(int).tolist()
    for household_id in eligible:
        panel, meta = prepare_household_panel(adapter, household_id)
        path = frame_write(panel, output_dir / f"household_{household_id}.parquet")
        meta["path"] = str(path.resolve())
        rows.append(meta)
    manifest = pd.DataFrame(rows)
    frame_write(manifest, output_dir / "panel_manifest.parquet")
    return manifest


def load_panel(manifest_row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    path = Path(str(manifest_row["path"]))
    frame = frame_read(path)
    if "Timestamp" in frame:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
    return frame


def history_features(panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    target = cfg["data"]["target"]
    result = pd.DataFrame(index=panel.index)
    observed = panel[target]
    for lag in cfg["features"]["lag_steps"]:
        result[f"load_lag_{lag}"] = observed.shift(max(int(lag) - 1, 0))
    for window in cfg["features"]["rolling_steps"]:
        rolling = observed.rolling(window, min_periods=window)
        result[f"load_roll_mean_{window}"] = rolling.mean()
        result[f"load_roll_std_{window}"] = rolling.std(ddof=0)
    result["load_ramp_1"] = observed - observed.shift(1)
    result["load_ramp_4"] = observed - observed.shift(4)
    day = observed.rolling(96, min_periods=96)
    result["load_day_min"] = day.min()
    result["load_day_max"] = day.max()
    result["load_day_base"] = day.quantile(0.10)
    on = (observed > cfg["features"]["on_threshold_kwh"]).astype(float)
    result["load_on_fraction_24h"] = on.rolling(96, min_periods=96).mean()
    transitions = on.ne(on.shift(1)).astype(float)
    result["load_transition_count_24h"] = transitions.rolling(96, min_periods=96).sum()
    return result


def valid_origin_positions(
    panel: pd.DataFrame,
    cfg: dict[str, Any],
    stride: int,
    max_origins: int | None = None,
) -> np.ndarray:
    history = cfg["features"]["history_steps"]
    forecast = cfg["features"]["forecast_steps"]
    target = cfg["data"]["target"]
    positions = np.arange(history, len(panel) - forecast, stride, dtype=int)
    if not positions.size:
        return positions
    segment = panel["Segment_ID"].to_numpy()
    y = panel[target].notna().to_numpy()
    keep = []
    for pos in positions:
        same_segment = segment[pos - history + 1] == segment[pos] == segment[pos + forecast]
        history_ok = y[pos - history + 1 : pos + 1].all()
        target_ok = y[pos + 1 : pos + forecast + 1].all()
        keep.append(same_segment and history_ok and target_ok)
    result = positions[np.asarray(keep)]
    if max_origins is not None and len(result) > max_origins:
        indices = np.linspace(0, len(result) - 1, max_origins, dtype=int)
        result = result[indices]
    return result


def make_supervised_rows(
    panel: pd.DataFrame,
    positions: np.ndarray,
    horizons: Iterable[int],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Create tidy origin–horizon rows with only origin-known information."""
    if not len(positions):
        return pd.DataFrame()
    timestamp = cfg["data"]["timestamp"]
    target = cfg["data"]["target"]
    scenario = cfg["data"]["weather_scenario"]
    origin_features = history_features(panel, cfg)
    origin_columns = origin_features.columns.tolist()
    cal_columns = [col for col in panel if col.startswith("cal_")]
    static_columns = [col for col in panel if col.startswith("static__")]
    weather_columns = [
        col
        for col in panel
        if f"weather_operational__{col}" in panel.columns
    ]
    evaluation_temperature = infer_temperature_column(weather_columns)
    base = origin_features.iloc[positions].reset_index(drop=True)
    records: list[pd.DataFrame] = []
    for horizon in horizons:
        target_positions = positions + int(horizon)
        block = base.copy()
        block["origin_time"] = panel.iloc[positions][timestamp].to_numpy()
        block["target_time"] = panel.iloc[target_positions][timestamp].to_numpy()
        block["horizon"] = int(horizon)
        block["horizon_sqrt"] = np.sqrt(float(horizon))
        block["y_true"] = panel.iloc[target_positions][target].to_numpy(dtype=float)
        block["baseline_persistence"] = panel.iloc[positions][target].to_numpy(dtype=float)
        daily_positions = target_positions - 96
        weekly_positions = target_positions - 672
        block["baseline_daily"] = panel.iloc[daily_positions][target].to_numpy(dtype=float)
        block["baseline_weekly"] = panel.iloc[weekly_positions][target].to_numpy(dtype=float)
        block["Household_ID"] = panel.iloc[positions]["Household_ID"].to_numpy()
        block["HP_Type"] = panel.iloc[positions]["HP_Type"].to_numpy()
        block["Segment_ID"] = panel.iloc[positions]["Segment_ID"].to_numpy()
        for col in cal_columns:
            block[f"future__{col}"] = panel.iloc[target_positions][col].to_numpy()
        for col in weather_columns:
            source = col if scenario == "oracle" else f"weather_operational__{col}"
            values = panel.iloc[target_positions][source].to_numpy(dtype=float)
            if scenario == "operational":
                fallback = panel.iloc[positions][f"weather_causal__{col}"].to_numpy(dtype=float)
                values = np.where(np.isfinite(values), values, fallback)
            block[f"future_weather__{col}"] = values
        block["evaluation_temperature_c"] = (
            panel.iloc[target_positions][evaluation_temperature].to_numpy(dtype=float)
            if evaluation_temperature is not None
            else np.nan
        )
        for col in static_columns:
            block[col] = panel.iloc[positions][col].to_numpy()
        records.append(block)
    result = pd.concat(records, ignore_index=True)
    required = origin_columns + ["y_true", "baseline_persistence", "baseline_daily", "baseline_weekly"]
    return result.dropna(subset=required).reset_index(drop=True)


def temperature_feature_name(columns: Iterable[str]) -> str | None:
    raw = [col.removeprefix("future_weather__") for col in columns if col.startswith("future_weather__")]
    match = infer_temperature_column(raw)
    return None if match is None else f"future_weather__{match}"
