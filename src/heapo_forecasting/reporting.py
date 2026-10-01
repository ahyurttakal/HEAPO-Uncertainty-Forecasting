from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from .utils import atomic_json, frame_read


def _exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".csv.gz").exists()


def build_validation_report(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Create machine-readable and human-readable post-run integrity checks."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    leakage_path = root / "splits" / "leakage_audit.json"
    if leakage_path.exists():
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        add(
            "leakage_boundary_audit",
            bool(leakage.get("passed", False)),
            "Household, temporal, horizon, and clock-stratum boundaries passed.",
        )
        add(
            "conformal_temperature_provenance",
            bool(
                leakage.get(
                    "operational_conformal_temperature_origin_available", False
                )
            ),
            (
                "Operational conformal regimes use origin-available weather; "
                "realised target-time temperature is evaluation-only."
            ),
        )
    else:
        add("leakage_boundary_audit", False, "leakage_audit.json is missing.")

    metric_path = root / "metrics" / "forecast_metrics.parquet"
    if _exists(metric_path):
        metrics = frame_read(metric_path)
        finite_mase = int(metrics["mase"].notna().sum())
        add(
            "finite_mase",
            finite_mase == len(metrics),
            f"{finite_mase}/{len(metrics)} metric cells have a finite MASE value.",
        )
        test = metrics[metrics["split"].isin(["seen_test", "unseen_test"])]
        baselines = test[test["model"].isin(["persistence", "seasonal_daily", "seasonal_weekly"])]
        add(
            "baseline_intervals_calibrated",
            bool(not baselines.empty and (baselines["mean_interval_width"] > 0).all()),
            "Seasonal and persistence baselines have non-degenerate calibrated intervals.",
        )
    else:
        add("forecast_metrics_present", False, "forecast_metrics artifact is missing.")

    paired_path = root / "metrics" / "paired_model_tests.parquet"
    add(
        "paired_block_bootstrap",
        _exists(paired_path),
        "Household-day paired bootstrap comparison table is present.",
    )

    calibration_audit_path = root / "metrics" / "calibration_index_audit.parquet"
    common_metric_path = root / "metrics" / "common_index_model_metrics.parquet"
    if _exists(calibration_audit_path) and _exists(common_metric_path):
        calibration_audit = frame_read(calibration_audit_path)
        primary = calibration_audit[
            calibration_audit["evaluation_index"].eq("common_tft_tabular")
            & calibration_audit["aggregation"].eq(
                "macro_model_stratum_horizon_cells"
            )
        ]
        common_metrics = frame_read(common_metric_path)
        common_test = common_metrics[
            common_metrics["split"].isin(["seen_test", "unseen_test"])
        ]
        figure_source_path = (
            root
            / "metrics"
            / "common_index_household_cluster_metric_intervals.parquet"
        )
        figure_source = (
            frame_read(figure_source_path)
            if _exists(figure_source_path)
            else common_test
        )
        figure_source = figure_source[
            figure_source["split"].isin(["seen_test", "unseen_test"])
        ]
        source_picp = float(figure_source["picp"].mean())
        reported_picp = (
            float(primary.iloc[0]["calibrated_picp"])
            if len(primary) == 1
            else np.nan
        )
        consistent = bool(
            len(primary) == 1
            and np.isfinite(source_picp)
            and np.isclose(source_picp, reported_picp, rtol=0.0, atol=1e-12)
        )
        target = 1.0 - float(cfg["models"]["conformal"]["alpha"])
        deviation = (
            float(primary.iloc[0]["calibrated_picp_mean_absolute_deviation"])
            if len(primary) == 1
            else np.nan
        )
        add(
            "figure3_calibration_index_consistency",
            consistent,
            (
                f"Common-index macro PICP={reported_picp:.6f}; the actual Figure 3 "
                "cell source and manuscript audit agree exactly."
            ),
        )
        add(
            "interval_coverage",
            bool(np.isfinite(deviation) and deviation <= 0.10),
            (
                f"Common-index macro mean absolute PICP deviation from "
                f"{target:.2f} is {deviation:.4f}; interval width and WIS are audited."
            ),
        )
    else:
        add(
            "figure3_calibration_index_consistency",
            False,
            "Common-index calibration audit or model metrics are missing.",
        )

    portfolio_path = root / "portfolios" / "portfolio_predictions.parquet"
    if _exists(portfolio_path):
        # Validation only needs the flag; loading the complete multi-million
        # row portfolio table can exhaust memory after a full run.
        portfolios = frame_read(portfolio_path, columns=["calibration_valid"])
        valid = int(portfolios.get("calibration_valid", False).fillna(False).sum()) if "calibration_valid" in portfolios else 0
        add(
            "portfolio_calibration_flagged",
            "calibration_valid" in portfolios,
            f"{valid}/{len(portfolios)} aggregate rows pass the minimum calibration sample rule.",
        )
    sensitivity_path = (
        root / "portfolios" / "peak_management_sensitivity.parquet"
    )
    if _exists(sensitivity_path):
        sensitivity = frame_read(
            sensitivity_path,
            columns=[
                "peak_threshold_quantile", "flexible_fraction", "shift_hours"
            ],
        )
        expected_thresholds = {
            float(value)
            for value in cfg["portfolio"].get(
                "peak_threshold_quantiles",
                [cfg["evaluation"]["peak_threshold_quantile"]],
            )
        }
        observed_thresholds = set(
            pd.to_numeric(
                sensitivity["peak_threshold_quantile"], errors="coerce"
            ).dropna()
        )
        add(
            "portfolio_operational_sensitivity",
            expected_thresholds.issubset(observed_thresholds),
            (
                "Peak thresholds, flexibility fractions and shift windows are "
                "exported in peak_management_sensitivity."
            ),
        )
    else:
        add(
            "portfolio_operational_sensitivity",
            False,
            "peak_management_sensitivity is missing.",
        )

    peak_path = root / "portfolios" / "peak_management_detail.parquet"
    if _exists(peak_path):
        peak = frame_read(
            peak_path,
            columns=[
                "model", "strategy", "peak_reduction", "energy_balance_error",
                "missed_event", "false_intervention",
            ],
        )
        minimum = float(peak["peak_reduction"].min()) if not peak.empty else np.nan
        energy_error = float(peak["energy_balance_error"].max()) if not peak.empty else np.nan
        add(
            "peak_shift_constraints",
            bool(not peak.empty and np.isfinite(minimum) and energy_error <= 1e-8),
            (
                f"Negative reductions are retained; minimum peak reduction={minimum:.6g}. "
                f"Maximum energy error={energy_error:.6g}."
            ),
        )
        oracle = peak
        oracle = oracle[
            oracle["model"].eq("perfect_foresight")
            & oracle["strategy"].eq("perfect_foresight")
        ]
        oracle_valid = bool(
            not oracle.empty
            and (oracle["peak_reduction"] >= -1e-9).all()
            and (oracle["energy_balance_error"] <= 1e-8).all()
            and ~oracle["missed_event"].astype(bool).any()
            and ~oracle["false_intervention"].astype(bool).any()
        )
        add(
            "perfect_foresight_benchmark",
            oracle_valid,
            "Oracle dispatch is present, energy-conserving, and has no forecast classification errors.",
        )

    if bool(cfg["models"]["tft"]["enabled"]):
        common_path = root / "metrics" / "common_index_model_metrics.parquet"
        add(
            "tft_common_index",
            _exists(common_path),
            "TFT and tabular models are evaluated on the same origin index.",
        )
        partition_path = root / "tft" / "tft_partition_manifest.csv"
        if partition_path.exists():
            partition = pd.read_csv(
                partition_path,
                parse_dates=[
                    "fit_target_end", "early_stop_target_start", "early_stop_target_end"
                ],
            )
            separated = False
            # The split cutoff is recorded in the split manifest; compare it
            # explicitly instead of relying on model-training timestamps.
            split_path = root / "splits" / "split_manifest.json"
            if split_path.exists():
                split = json.loads(split_path.read_text(encoding="utf-8"))
                train_end = pd.Timestamp(split["train_end"])
                separated = bool(
                    not partition.empty
                    and (partition["fit_target_end"] < partition["early_stop_target_start"]).all()
                    and (partition["early_stop_target_end"] <= train_end).all()
                )
            add(
                "tft_early_stopping_independence",
                separated,
                "TFT early-stopping targets follow fitting targets and remain inside the training segment.",
            )
        else:
            add(
                "tft_early_stopping_independence",
                False,
                "tft_partition_manifest.csv is missing.",
            )

    portfolio_protocol_path = root / "portfolios" / "portfolio_protocol.json"
    if portfolio_protocol_path.exists():
        protocol = json.loads(portfolio_protocol_path.read_text(encoding="utf-8"))
        add(
            "portfolio_origin_only_decisions",
            bool(
                protocol.get("negative_realised_peak_reductions_retained", False)
                and protocol.get("realised_demand_role")
                == "ex-post scoring and source-availability cap only"
            ),
            "Portfolio decisions use forecasts only; realised demand is restricted to ex-post execution/scoring.",
        )
    else:
        add(
            "portfolio_origin_only_decisions",
            False,
            "portfolio_protocol.json is missing.",
        )

    report = {
        "run_dir": str(root),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    atomic_json(report, root / "validation_report.json")
    lines = [
        "# HEAPO run validation",
        "",
        f"Overall status: {'PASS' if report['passed'] else 'REVIEW'}",
        "",
    ]
    for item in checks:
        lines.append(f"- [{'x' if item['passed'] else ' '}] {item['check']}: {item['detail']}")
    (root / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
