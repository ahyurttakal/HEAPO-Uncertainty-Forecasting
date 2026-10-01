from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import run_dir
from .statistics import hierarchical_metric_intervals, hierarchical_paired_bootstrap
from .utils import atomic_json, frame_read, frame_write


def _artifact_exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".csv.gz").exists()


def _require_artifact(path: Path, description: str) -> None:
    if not _artifact_exists(path):
        raise FileNotFoundError(
            f"{description} is missing: {path}. Run the completed full stage first."
        )


def _read_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    selected = list(dict.fromkeys(columns))
    try:
        return frame_read(path, columns=selected)
    except TypeError:
        return frame_read(path).loc[:, selected]


def portfolio_calibration_sensitivity(
    corrections: pd.DataFrame,
    thresholds: Iterable[int] = (20, 30, 50),
) -> pd.DataFrame:
    """Report which portfolio type/size cells remain calibratable by threshold."""
    required = {
        "portfolio_type",
        "portfolio_size",
        "model",
        "horizon",
        "calibration_n",
        "calibration_rows",
    }
    missing = sorted(required.difference(corrections.columns))
    if missing:
        raise ValueError(f"Portfolio correction columns are missing: {missing}")
    rows: list[dict[str, Any]] = []
    keys = ["portfolio_type", "portfolio_size"]
    for threshold in sorted({int(value) for value in thresholds}):
        if threshold <= 0:
            raise ValueError("Portfolio sensitivity thresholds must be positive")
        for values, group in corrections.groupby(keys, observed=True, sort=True):
            valid = group["calibration_n"].astype(int) >= threshold
            rows.append(
                {
                    "min_calibration_samples": threshold,
                    "portfolio_type": values[0],
                    "portfolio_size": int(values[1]),
                    "model_horizon_cells": int(len(group)),
                    "valid_cells": int(valid.sum()),
                    "valid_cell_fraction": float(valid.mean()),
                    "all_cells_reportable": bool(valid.all()),
                    "calibration_n_min": int(group["calibration_n"].min()),
                    "calibration_n_median": float(group["calibration_n"].median()),
                    "calibration_n_max": int(group["calibration_n"].max()),
                    "calibration_rows_min": int(group["calibration_rows"].min()),
                    "calibration_rows_max": int(group["calibration_rows"].max()),
                    "exclusion_reason": (
                        ""
                        if valid.all()
                        else f"independent origins below {threshold}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _test_summary(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {
            "tests": 0,
            "holm_significant": 0,
            "candidate_better": 0,
            "candidate_worse": 0,
            "confirmatory": 0,
        }
    significant = frame["significant_holm"].fillna(False)
    return {
        "tests": int(len(frame)),
        "holm_significant": int(significant.sum()),
        "candidate_better": int(
            (significant & frame["effect_direction"].eq("candidate_better")).sum()
        ),
        "candidate_worse": int(
            (significant & frame["effect_direction"].eq("candidate_worse")).sum()
        ),
        "confirmatory": int(frame.get("confirmatory_holm", False).sum()),
    }


def _write_statistical_report(
    root: Path,
    cfg: dict[str, Any],
    outputs: dict[str, pd.DataFrame],
    sensitivity: pd.DataFrame,
) -> Path:
    current_threshold = int(cfg["portfolio"].get("min_calibration_samples", 100))
    minimum_households = int(
        cfg.get("statistics", {}).get("minimum_households_for_confirmatory", 20)
    )
    current = sensitivity[
        sensitivity["min_calibration_samples"] == current_threshold
    ]
    excluded = current[~current["all_cells_reportable"]]
    cluster_metrics = outputs["household_cluster_metric_intervals"]
    households = (
        cluster_metrics.groupby(["split", "HP_Type"], observed=True)["n_households"]
        .max()
        .reset_index()
    )
    lines = [
        "# HEAPO final statistical validation",
        "",
        "Status: COMPLETE",
        "",
        "## Resampling design",
        "",
        "- Primary inference resamples complete households with replacement.",
        "- Sensitivity inference resamples households and then local calendar days within each sampled household occurrence.",
        "- MAE and PICP remain row-weighted within each bootstrap replicate.",
        "- Paired tests use identical household/origin/horizon observations for both models.",
        "- Holm correction controls family-wise error separately for every pre-specified candidate/comparator family.",
        f"- Holm-significant results are labelled confirmatory only when at least {minimum_households} independent households are present.",
        "",
        "## Independent household counts",
        "",
        "| Split | HP type | Households |",
        "|---|---:|---:|",
    ]
    for row in households.itertuples(index=False):
        lines.append(f"| {row.split} | {row.HP_Type} | {int(row.n_households)} |")
    lines.extend(["", "## Paired-test summary", ""])
    for name in [
        "household_cluster_paired_tests",
        "nested_household_day_paired_tests",
        "common_index_household_cluster_tft_paired_tests",
        "common_index_nested_household_day_tft_paired_tests",
    ]:
        summary = _test_summary(outputs[name])
        lines.append(
            f"- {name}: {summary['holm_significant']}/{summary['tests']} Holm-significant; "
            f"candidate better={summary['candidate_better']}, candidate worse={summary['candidate_worse']}; "
            f"confirmatory with sufficient households={summary['confirmatory']}."
        )
    lines.extend(
        [
            "",
            "## Portfolio calibration sensitivity",
            "",
            f"Configured minimum independent origins: {current_threshold}.",
            "",
        ]
    )
    if excluded.empty:
        lines.append("Every configured portfolio type/size is reportable at this threshold.")
    else:
        lines.append("Excluded portfolio cells at the configured threshold:")
        for row in excluded.itertuples(index=False):
            lines.append(
                f"- {row.portfolio_type}, size {int(row.portfolio_size)}: "
                f"calibration_n={int(row.calibration_n_min)} ({row.exclusion_reason})."
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "Household-level results are the primary inferential evidence. The nested household/day results are a sensitivity analysis and do not increase the number of independent households.",
            "",
        ]
    )
    path = root / "final_statistical_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_statistical_finalize(cfg: dict[str, Any]) -> dict[str, Any]:
    """Recompute inferential outputs from cached predictions without refitting."""
    root = run_dir(cfg)
    metrics_dir = root / "metrics"
    portfolio_dir = root / "portfolios"
    predictions_path = root / "predictions" / "household_predictions.parquet"
    common_path = root / "tft" / "common_index_predictions.parquet"
    corrections_path = portfolio_dir / "portfolio_interval_corrections.parquet"
    _require_artifact(predictions_path, "Household predictions")
    _require_artifact(common_path, "Common-index predictions")
    _require_artifact(corrections_path, "Portfolio interval corrections")

    statistical = cfg.get("statistics", {})
    repetitions = int(statistical.get("bootstrap_repetitions", 1000))
    confidence = float(statistical.get("confidence", 0.95))
    testing_alpha = float(statistical.get("multiple_testing_alpha", 0.05))
    minimum_households = int(
        statistical.get("minimum_households_for_confirmatory", 20)
    )
    timezone = str(cfg["data"].get("timezone", "UTC"))
    seed = int(cfg["project"]["seed"])
    report_horizons = {int(value) for value in cfg["evaluation"]["report_horizons"]}
    test_splits = {"seen_test", "unseen_test"}
    prediction_columns = [
        "Household_ID",
        "origin_time",
        "target_time",
        "horizon",
        "split",
        "HP_Type",
        "model",
        "y_true",
        "y_pred",
        "q10",
        "q90",
    ]
    predictions = _read_columns(predictions_path, prediction_columns)
    predictions = predictions[
        predictions["split"].isin(test_splits)
        & predictions["horizon"].astype(int).isin(report_horizons)
    ].copy()
    comparisons = [
        (str(pair[0]), str(pair[1]))
        for pair in statistical.get(
            "comparisons",
            [
                ["lightgbm", "persistence"],
                ["lightgbm", "seasonal_daily"],
                ["lightgbm", "seasonal_weekly"],
                ["lightgbm", "elastic_net"],
            ],
        )
    ]
    outputs: dict[str, pd.DataFrame] = {}
    outputs["household_cluster_metric_intervals"] = hierarchical_metric_intervals(
        predictions,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
        timezone=timezone,
        nested_days=False,
        minimum_households_for_confirmatory=minimum_households,
    )
    outputs["nested_household_day_metric_intervals"] = hierarchical_metric_intervals(
        predictions,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
        timezone=timezone,
        nested_days=True,
        minimum_households_for_confirmatory=minimum_households,
    )
    outputs["household_cluster_paired_tests"] = hierarchical_paired_bootstrap(
        predictions,
        comparisons=comparisons,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
        timezone=timezone,
        nested_days=False,
        multiple_testing_alpha=testing_alpha,
        minimum_households_for_confirmatory=minimum_households,
    )
    outputs["nested_household_day_paired_tests"] = hierarchical_paired_bootstrap(
        predictions,
        comparisons=comparisons,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
        timezone=timezone,
        nested_days=True,
        multiple_testing_alpha=testing_alpha,
        minimum_households_for_confirmatory=minimum_households,
    )
    del predictions
    gc.collect()

    common = _read_columns(common_path, prediction_columns)
    common = common[
        common["split"].isin(test_splits)
        & common["horizon"].astype(int).isin(report_horizons)
    ].copy()
    tft_comparators = [
        model
        for model in [
            "lightgbm",
            "seasonal_daily",
            "seasonal_weekly",
            "persistence",
        ]
        if model in set(common["model"].astype(str))
    ]
    tft_pairs = [("tft", model) for model in tft_comparators]
    outputs["common_index_household_cluster_metric_intervals"] = (
        hierarchical_metric_intervals(
            common,
            repetitions=repetitions,
            confidence=confidence,
            seed=seed + 31,
            timezone=timezone,
            nested_days=False,
            minimum_households_for_confirmatory=minimum_households,
        )
    )
    outputs["common_index_nested_household_day_metric_intervals"] = (
        hierarchical_metric_intervals(
            common,
            repetitions=repetitions,
            confidence=confidence,
            seed=seed + 31,
            timezone=timezone,
            nested_days=True,
            minimum_households_for_confirmatory=minimum_households,
        )
    )
    outputs["common_index_household_cluster_tft_paired_tests"] = (
        hierarchical_paired_bootstrap(
            common,
            comparisons=tft_pairs,
            repetitions=repetitions,
            confidence=confidence,
            seed=seed + 31,
            timezone=timezone,
            nested_days=False,
            multiple_testing_alpha=testing_alpha,
            minimum_households_for_confirmatory=minimum_households,
        )
    )
    outputs["common_index_nested_household_day_tft_paired_tests"] = (
        hierarchical_paired_bootstrap(
            common,
            comparisons=tft_pairs,
            repetitions=repetitions,
            confidence=confidence,
            seed=seed + 31,
            timezone=timezone,
            nested_days=True,
            multiple_testing_alpha=testing_alpha,
            minimum_households_for_confirmatory=minimum_households,
        )
    )
    del common
    gc.collect()

    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        output_paths[name] = str(frame_write(frame, metrics_dir / f"{name}.parquet"))

    corrections = frame_read(corrections_path)
    thresholds = statistical.get("portfolio_sensitivity_thresholds", [20, 30, 50])
    if int(cfg["portfolio"].get("min_calibration_samples", 100)) not in thresholds:
        thresholds = [*thresholds, int(cfg["portfolio"]["min_calibration_samples"])]
    sensitivity = portfolio_calibration_sensitivity(corrections, thresholds)
    sensitivity_path = frame_write(
        sensitivity, portfolio_dir / "portfolio_calibration_sensitivity.parquet"
    )
    sensitivity_csv = portfolio_dir / "portfolio_calibration_sensitivity.csv"
    sensitivity.to_csv(sensitivity_csv, index=False)
    report_path = _write_statistical_report(root, cfg, outputs, sensitivity)

    result = {
        "run_dir": str(root),
        "bootstrap_repetitions": repetitions,
        "confidence": confidence,
        "multiple_testing_alpha": testing_alpha,
        "minimum_households_for_confirmatory": minimum_households,
        "metric_outputs": output_paths,
        "portfolio_sensitivity": str(sensitivity_path),
        "portfolio_sensitivity_csv": str(sensitivity_csv),
        "report": str(report_path),
        "test_summaries": {
            name: _test_summary(frame)
            for name, frame in outputs.items()
            if "paired_tests" in name
        },
    }
    from .reporting import build_validation_report

    result["validation"] = build_validation_report(root, cfg)
    atomic_json(result, root / "statistical_finalize.json")
    return result
