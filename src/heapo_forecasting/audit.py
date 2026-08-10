from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import HEAPOAdapter, infer_temperature_column, merge_weather_15min
from .utils import frame_write


@dataclass
class HouseholdAudit:
    household_id: int
    hp_type: str | None
    meter_file: bool
    separate_hp_meter: bool
    start_utc: str | None
    end_utc: str | None
    observed_rows: int
    expected_rows: int
    usable_days: float
    target_missing_fraction: float
    heating_seasons: int
    weather_id: str | None
    temperature_column: str | None
    weather_match_fraction: float
    metadata_nonmissing_fraction: float
    protocol_available: bool
    multiple_visits: bool
    eligible: bool
    exclusion_reasons: str


def _heating_seasons(timestamps: pd.Series, valid: pd.Series, months: list[int]) -> int:
    local = timestamps.dt.tz_convert("Europe/Zurich")
    season_year = local.dt.year + (local.dt.month >= 10).astype(int)
    table = pd.DataFrame({"season": season_year, "month": local.dt.month, "valid": valid})
    table = table[table["month"].isin(months) & table["valid"]]
    if table.empty:
        return 0
    days = table.assign(day=local[table.index].dt.date).groupby("season")["day"].nunique()
    return int((days >= 120).sum())


def audit_household(adapter: HEAPOAdapter, household_id: int) -> HouseholdAudit:
    cfg = adapter.cfg
    path = adapter.meter_path(household_id)
    hp_type = adapter.normalized_hp_type(household_id)
    reasons: list[str] = []
    if not path.exists():
        return HouseholdAudit(
            int(household_id), hp_type, False, False, None, None, 0, 0, 0.0,
            1.0, 0, None, None, 0.0, 0.0, False, False, False, "missing_meter_file"
        )
    meter = adapter.load_meter(household_id)
    target = adapter.target
    separate = target in meter and meter[target].notna().any()
    if hp_type is None:
        reasons.append("unknown_hp_type")
    if not separate:
        reasons.append("no_separate_hp_meter")

    start = meter[adapter.timestamp].min()
    end = meter[adapter.timestamp].max()
    expected = int((end - start) / pd.Timedelta(minutes=15)) + 1 if pd.notna(start) else 0
    target_values = meter[target] if target in meter else pd.Series(np.nan, index=meter.index)
    missing = 1.0 - target_values.notna().sum() / max(expected, 1)
    usable_days = float(target_values.notna().sum() / 96.0)
    seasons = _heating_seasons(
        meter[adapter.timestamp], target_values.notna(), cfg["inclusion"]["heating_months"]
    )

    static = adapter.static_row(household_id)
    weather_id = static.get("Weather_ID")
    weather_match = 0.0
    temp_col: str | None = None
    try:
        weather = adapter.load_weather_for_household(household_id)
        temp_col = infer_temperature_column(weather.columns.tolist())
        if temp_col:
            merged, _ = merge_weather_15min(
                meter[[adapter.timestamp] + ([target] if target in meter else [])],
                weather,
                adapter.timestamp,
            )
            weather_match = float(merged.loc[target_values.notna(), temp_col].notna().mean())
    except Exception:
        weather_match = 0.0

    metadata_keys = [key for key in static if key != adapter.household_id]
    meta_fraction = (
        float(pd.Series({key: static[key] for key in metadata_keys}).notna().mean())
        if metadata_keys
        else 0.0
    )
    visits = adapter.visit_dates(household_id)

    inc = cfg["inclusion"]
    if usable_days < inc["min_days"]:
        reasons.append("insufficient_days")
    if missing > inc["max_target_missing_fraction"]:
        reasons.append("target_missingness")
    if seasons < inc["min_heating_seasons"]:
        reasons.append("insufficient_heating_seasons")
    if weather_match < inc["min_weather_match_fraction"]:
        reasons.append("weather_match")
    return HouseholdAudit(
        household_id=int(household_id),
        hp_type=hp_type,
        meter_file=True,
        separate_hp_meter=separate,
        start_utc=start.isoformat() if pd.notna(start) else None,
        end_utc=end.isoformat() if pd.notna(end) else None,
        observed_rows=int(len(meter)),
        expected_rows=expected,
        usable_days=usable_days,
        target_missing_fraction=float(missing),
        heating_seasons=seasons,
        weather_id=None if pd.isna(weather_id) else str(weather_id),
        temperature_column=temp_col,
        weather_match_fraction=weather_match,
        metadata_nonmissing_fraction=meta_fraction,
        protocol_available=bool(visits),
        multiple_visits=len(visits) > 1,
        eligible=not reasons,
        exclusion_reasons="|".join(reasons),
    )


def run_audit(cfg: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    adapter = HEAPOAdapter(cfg["data"]["root"], cfg)
    metadata = adapter.metadata()
    ids = metadata[adapter.household_id].dropna().astype(int).unique().tolist()
    rows = [asdict(audit_household(adapter, hid)) for hid in ids]
    frame = pd.DataFrame(rows).sort_values(["eligible", "hp_type", "household_id"], ascending=[False, True, True])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_write(frame, output_dir / "household_audit.parquet")
    summary = (
        frame.groupby(["hp_type", "eligible"], dropna=False)
        .agg(households=("household_id", "nunique"), median_usable_days=("usable_days", "median"))
        .reset_index()
    )
    frame_write(summary, output_dir / "audit_summary.parquet")
    exclusions = (
        frame.assign(reason=frame["exclusion_reasons"].replace("", "eligible").str.split("\u007c"))
        .explode("reason")
        .groupby("reason", dropna=False)
        .size()
        .rename("households")
        .reset_index()
        .sort_values("households", ascending=False)
    )
    frame_write(exclusions, output_dir / "exclusion_counts.parquet")
    return frame
