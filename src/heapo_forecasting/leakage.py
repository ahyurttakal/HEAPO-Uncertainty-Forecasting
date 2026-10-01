from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_json, frame_read


class LeakageProtocolError(RuntimeError):
    """Raised when an artifact crosses a pre-specified information boundary."""


def validate_supervised_protocol(
    manifest: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Fail closed on household, temporal, origin, and horizon violations."""
    train_ids = set(map(int, split_manifest["training_households"]))
    validation_ids = set(map(int, split_manifest["unseen_validation_households"]))
    test_ids = set(map(int, split_manifest["unseen_test_households"]))
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise LeakageProtocolError("Household partitions are not disjoint")

    train_end = pd.Timestamp(split_manifest["train_end"])
    calibration_end = pd.Timestamp(split_manifest["calibration_end"])
    allowed_hours = set(map(int, cfg["sampling"]["eval_clock_hours"]))
    forecast_steps = int(cfg["features"]["forecast_steps"])
    max_origins = int(cfg["sampling"]["max_eval_origins_per_household"])
    checks: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    def require(condition: bool, message: str) -> None:
        checks.append({"check": message, "passed": bool(condition)})
        if not condition:
            raise LeakageProtocolError(message)

    for _, record in manifest.iterrows():
        declared_temperature = record.get("conformal_temperature_feature", "")
        temperature_feature = (
            ""
            if pd.isna(declared_temperature)
            else str(declared_temperature).strip()
        )
        columns = [
            "Household_ID", "origin_time", "target_time", "horizon", "split"
        ]
        if cfg["data"]["weather_scenario"] == "operational":
            require(
                bool(temperature_feature),
                "operational conformal temperature feature is declared",
            )
            columns.extend(["conformal_temperature_c", temperature_feature])
        rows = frame_read(
            Path(record["path"]),
            columns=list(dict.fromkeys(columns)),
        )
        if rows.empty:
            continue
        household = int(record["household_id"])
        require(rows["Household_ID"].astype(int).eq(household).all(), "shard household identity")
        require((pd.to_datetime(rows["target_time"], utc=True) > pd.to_datetime(rows["origin_time"], utc=True)).all(), "targets occur after origins")
        if cfg["data"]["weather_scenario"] == "operational":
            observed = pd.to_numeric(
                rows["conformal_temperature_c"], errors="coerce"
            ).to_numpy(dtype=float)
            origin_available = pd.to_numeric(
                rows[temperature_feature], errors="coerce"
            ).to_numpy(dtype=float)
            require(
                np.allclose(
                    observed,
                    origin_available,
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                ),
                "operational conformal temperature is origin-available",
            )
        for split_name, block in rows.groupby("split", observed=True):
            split_name = str(split_name)
            origins = pd.to_datetime(block["origin_time"], utc=True)
            targets = pd.to_datetime(block["target_time"], utc=True)
            unique_origins = origins.drop_duplicates()
            counts[split_name] = counts.get(split_name, 0) + int(len(block))
            require(block["horizon"].between(1, forecast_steps).all(), f"{split_name} horizons are bounded")
            per_origin = block.groupby("origin_time", observed=True)["horizon"].nunique()
            require(per_origin.eq(forecast_steps).all(), f"{split_name} origins contain complete forecast paths")
            if split_name == "train":
                require(household in train_ids, "training rows use training households only")
                require((origins <= train_end).all(), "training origins precede the training cutoff")
                require(
                    (targets <= train_end).all(),
                    "training target trajectories end by the training cutoff",
                )
                boundary_rows.append(
                    {
                        "household_id": household,
                        "split": split_name,
                        "origins": int(unique_origins.nunique()),
                        "rows": int(len(block)),
                        "origin_min": origins.min(),
                        "origin_max": origins.max(),
                        "target_min": targets.min(),
                        "target_max": targets.max(),
                        "permitted_target_end": train_end,
                        "boundary_passed": bool((targets <= train_end).all()),
                    }
                )
                continue
            local_hours = unique_origins.dt.tz_convert(str(cfg["data"]["timezone"])).dt.hour
            require(set(map(int, local_hours.unique())).issubset(allowed_hours), f"{split_name} origins obey clock strata")
            require(len(unique_origins) <= max_origins, f"{split_name} origin cap is respected")
            if split_name == "calibration":
                require(household in train_ids, "calibration rows use training households only")
                require(((origins > train_end) & (origins <= calibration_end)).all(), "calibration origins stay inside the calibration interval")
                require(
                    (targets <= calibration_end).all(),
                    "calibration target trajectories end by the calibration cutoff",
                )
            elif split_name == "seen_test":
                require(household in train_ids, "seen test rows use training households only")
                require((origins > calibration_end).all(), "seen test origins follow calibration")
            elif split_name == "unseen_validation":
                require(household in validation_ids, "validation rows use held-out validation households only")
                require(
                    (origins <= train_end).all(),
                    "validation origins stay inside the development period",
                )
                require(
                    (targets <= train_end).all(),
                    "validation target trajectories end by the training cutoff",
                )
            elif split_name == "unseen_test":
                require(household in test_ids, "unseen test rows use final-test households only")
                require((origins > calibration_end).all(), "unseen test origins follow calibration")
            else:
                raise LeakageProtocolError(f"Unknown split label: {split_name}")
            target_limit = (
                train_end
                if split_name == "unseen_validation"
                else calibration_end
                if split_name == "calibration"
                else pd.NaT
            )
            boundary_rows.append(
                {
                    "household_id": household,
                    "split": split_name,
                    "origins": int(unique_origins.nunique()),
                    "rows": int(len(block)),
                    "origin_min": origins.min(),
                    "origin_max": origins.max(),
                    "target_min": targets.min(),
                    "target_max": targets.max(),
                    "permitted_target_end": target_limit,
                    "boundary_passed": bool(
                        pd.isna(target_limit) or (targets <= target_limit).all()
                    ),
                }
            )

    report = {
        "schema_version": int(cfg["pipeline"]["schema_version"]),
        "passed": True,
        "household_sets_disjoint": True,
        "operational_conformal_temperature_origin_available": bool(
            cfg["data"]["weather_scenario"] != "operational"
            or any(
                item["check"]
                == "operational conformal temperature is origin-available"
                and item["passed"]
                for item in checks
            )
        ),
        "allowed_local_clock_hours": sorted(allowed_hours),
        "rows_by_split": counts,
        "trajectory_boundary_rule": (
            "train and unseen_validation target_time <= train_end; calibration "
            "target_time <= calibration_end; test target_time > origin_time"
        ),
        "checks": checks,
    }
    atomic_json(report, output_path)
    pd.DataFrame(boundary_rows).to_csv(
        output_path.with_name(f"{output_path.stem}_boundaries.csv"), index=False
    )
    return report
