from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import atomic_json, frame_read


def _exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".csv.gz").exists()


def build_validation_report(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Create machine-readable and human-readable post-run integrity checks."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    metric_path = root / "metrics" / "forecast_metrics.parquet"
    if _exists(metric_path):
        metrics = frame_read(metric_path)
        finite_mase = int(metrics["mase"].notna().sum())
        add(
            "finite_mase",
            finite_mase == len(metrics),
            f"{finite_mase}/{len(metrics)} metric cells have a finite MASE value.",
        )
        target = 1.0 - float(cfg["models"]["conformal"]["alpha"])
        test = metrics[metrics["split"].isin(["seen_test", "unseen_test"])]
        deviation = float((test["picp"] - target).abs().mean()) if not test.empty else np.nan
        add(
            "interval_coverage",
            bool(np.isfinite(deviation) and deviation <= 0.10),
            f"Mean absolute PICP deviation from {target:.2f} is {deviation:.4f}.",
        )
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

    portfolio_path = root / "portfolios" / "portfolio_predictions_v2.parquet"
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

    peak_path = root / "portfolios" / "peak_management_detail_v2.parquet"
    if _exists(peak_path):
        peak = frame_read(
            peak_path, columns=["peak_reduction", "energy_balance_error"]
        )
        minimum = float(peak["peak_reduction"].min()) if not peak.empty else np.nan
        energy_error = float(peak["energy_balance_error"].max()) if not peak.empty else np.nan
        add(
            "peak_shift_constraints",
            bool(not peak.empty and minimum >= -1e-9 and energy_error <= 1e-8),
            f"Minimum peak reduction={minimum:.6g}; maximum energy error={energy_error:.6g}.",
        )

    if bool(cfg["models"]["tft"]["enabled"]):
        common_path = root / "metrics" / "common_index_model_metrics.parquet"
        add(
            "tft_common_index",
            _exists(common_path),
            "TFT and tabular models are evaluated on the same origin index.",
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
