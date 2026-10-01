from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_json, finite_sample_quantile, frame_write


def sample_portfolios(
    predictions: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    house_type = (
        predictions[["Household_ID", "HP_Type"]]
        .drop_duplicates()
        .assign(Household_ID=lambda x: x["Household_ID"].astype(int))
    )
    cohorts = {
        "seen": set(split_manifest["training_households"]),
        "unseen": set(split_manifest["unseen_test_households"]),
    }
    rows: list[dict[str, Any]] = []
    for cohort, allowed in cohorts.items():
        pool = house_type[house_type["Household_ID"].isin(allowed)]
        for portfolio_type in cfg["portfolio"]["types"]:
            candidates = pool.loc[
                pool["HP_Type"] == portfolio_type, "Household_ID"
            ].to_numpy(dtype=int) if portfolio_type != "mixed" else pool["Household_ID"].to_numpy(dtype=int)
            for size in cfg["portfolio"]["sizes"]:
                if portfolio_type == "mixed":
                    ashp = pool.loc[pool["HP_Type"] == "ASHP", "Household_ID"].to_numpy(dtype=int)
                    gshp = pool.loc[pool["HP_Type"] == "GSHP", "Household_ID"].to_numpy(dtype=int)
                    n_ashp = int(math.ceil(size / 2))
                    n_gshp = int(size - n_ashp)
                    available = len(ashp) >= n_ashp and len(gshp) >= n_gshp
                else:
                    available = len(candidates) >= size
                if not available:
                    continue
                for rep in range(int(cfg["portfolio"]["repetitions"])):
                    if portfolio_type == "mixed":
                        selected = np.concatenate(
                            [
                                rng.choice(ashp, size=n_ashp, replace=False),
                                rng.choice(gshp, size=n_gshp, replace=False),
                            ]
                        )
                    else:
                        selected = rng.choice(candidates, size=int(size), replace=False)
                    portfolio_id = f"{cohort}_{portfolio_type}_{size}_{rep:03d}"
                    for household_id in selected:
                        rows.append(
                            {
                                "portfolio_id": portfolio_id,
                                "cohort": cohort,
                                "portfolio_type": portfolio_type,
                                "portfolio_size": int(size),
                                "repetition": rep,
                                "Household_ID": int(household_id),
                            }
                        )
    return pd.DataFrame(rows)


def aggregate_predictions(
    predictions: pd.DataFrame,
    memberships: pd.DataFrame,
    min_coverage: float,
) -> pd.DataFrame:
    if memberships.empty:
        return pd.DataFrame()
    # A single memberships-to-predictions merge duplicates the complete
    # prediction table for every portfolio containing a household.  With the
    # default repetitions this can create tens of millions of temporary rows.
    # Aggregate one portfolio at a time and retain only columns needed below.
    prediction_columns = [
        "Household_ID",
        "model",
        "split",
        "origin_time",
        "target_time",
        "horizon",
        "y_true",
        "y_pred",
        "q50",
    ]
    time_keys = ["model", "split", "origin_time", "target_time", "horizon"]
    parts: list[pd.DataFrame] = []
    for portfolio_id, members in memberships.groupby("portfolio_id", sort=False):
        metadata = members.iloc[0]
        household_ids = members["Household_ID"].astype(int).unique()
        selected = predictions.loc[
            predictions["Household_ID"].isin(household_ids), prediction_columns
        ]
        if selected.empty:
            continue
        block = (
            selected.groupby(time_keys, dropna=False, sort=False, observed=True)
            .agg(
                households_observed=("Household_ID", "nunique"),
                y_true=("y_true", "sum"),
                y_pred=("y_pred", "sum"),
                q50=("q50", "sum"),
            )
            .reset_index()
        )
        block.insert(0, "repetition", int(metadata["repetition"]))
        block.insert(0, "portfolio_size", int(metadata["portfolio_size"]))
        block.insert(0, "portfolio_type", metadata["portfolio_type"])
        block.insert(0, "cohort", metadata["cohort"])
        block.insert(0, "portfolio_id", portfolio_id)
        block["coverage"] = block["households_observed"] / block["portfolio_size"]
        block = block[block["coverage"] >= float(min_coverage)]
        if not block.empty:
            parts.append(block)
    if not parts:
        return pd.DataFrame()
    aggregate = pd.concat(parts, ignore_index=True, copy=False)
    aggregate["coverage"] = aggregate["households_observed"] / aggregate["portfolio_size"]
    # Dictionary-encode repeated identifiers.  Portfolio tables contain
    # millions of rows, while these columns only have a few hundred distinct
    # values.  Keeping them as Python objects needlessly consumes several GB.
    for column in ["portfolio_id", "cohort", "portfolio_type", "model", "split"]:
        aggregate[column] = aggregate[column].astype("category")
    return aggregate


def _dense_key_codes(
    frame: pd.DataFrame,
    keys: list[str],
    levels: list[pd.Index] | None = None,
) -> tuple[np.ndarray, list[pd.Index], np.ndarray]:
    """Encode a small Cartesian key without building a multi-million-row index."""
    if levels is None:
        levels = [pd.Index(pd.unique(frame[key].dropna())) for key in keys]
    codes = np.zeros(len(frame), dtype=np.int32)
    valid = np.ones(len(frame), dtype=bool)
    for key, categories in zip(keys, levels):
        if len(categories) == 0:
            valid[:] = False
            continue
        current = pd.Categorical(frame[key], categories=categories).codes
        valid &= current >= 0
        np.multiply(codes, len(categories), out=codes)
        np.add(codes, np.maximum(current, 0), out=codes)
    return codes, levels, valid


def calibrate_portfolio_intervals(
    aggregate: pd.DataFrame,
    alpha: float = 0.20,
    min_samples: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool coincident aggregate residuals by type/size/horizon on seen calibration."""
    if aggregate.empty:
        return aggregate, pd.DataFrame()
    keys = ["portfolio_type", "portfolio_size", "model", "horizon"]
    calibration_columns = [
        *keys,
        "origin_time",
        "target_time",
        "y_true",
        "q50",
    ]
    calibration = aggregate.loc[
        (aggregate["cohort"] == "seen") & (aggregate["split"] == "calibration"),
        calibration_columns,
    ].copy()
    calibration["residual"] = calibration["y_true"] - calibration["q50"]
    rows: list[dict[str, Any]] = []
    for values, group in calibration.groupby(
        keys, dropna=False, observed=True, sort=True
    ):
        residual = group["residual"].to_numpy(dtype=float)
        n_rows = len(residual)
        n_origins = int(
            group[["origin_time", "target_time"]].drop_duplicates().shape[0]
        )
        lower_level = max(
            0.0,
            np.floor((n_rows + 1) * (alpha / 2)) / max(n_rows, 1),
        )
        row = dict(zip(keys, values))
        row.update(
            {
                "residual_q10": (
                    float(np.quantile(residual, lower_level, method="lower"))
                    if n_origins >= min_samples
                    else np.nan
                ),
                "residual_q90": (
                    finite_sample_quantile(residual, 1 - alpha / 2)
                    if n_origins >= min_samples
                    else np.nan
                ),
                "calibration_n": n_origins,
                "calibration_rows": int(n_rows),
                "calibration_valid": bool(n_origins >= min_samples),
            }
        )
        rows.append(row)
    corrections = pd.DataFrame(rows)
    if corrections.empty:
        result = aggregate
        result["calibration_n"] = np.zeros(len(result), dtype=np.int32)
        result["calibration_valid"] = np.zeros(len(result), dtype=bool)
        result["q10"] = np.full(len(result), np.nan, dtype=np.float64)
        result["q90"] = np.full(len(result), np.nan, dtype=np.float64)
        result["interval_calibrated"] = np.zeros(len(result), dtype=bool)
        return result, corrections
    # Avoid DataFrame.merge here.  A merge duplicates every block in the
    # multi-million-row aggregate and can require several additional GB.  The
    # correction key has only a few thousand possible values, so use a compact
    # dense lookup and attach only the columns needed downstream.  Full
    # correction diagnostics remain available in the separate corrections
    # artifact.
    result = aggregate
    row_codes, levels, row_valid = _dense_key_codes(result, keys)
    correction_codes, _, correction_valid = _dense_key_codes(
        corrections, keys, levels=levels
    )
    lookup_size = int(np.prod([max(len(level), 1) for level in levels]))

    lower_lookup = np.full(lookup_size, np.nan, dtype=np.float64)
    upper_lookup = np.full(lookup_size, np.nan, dtype=np.float64)
    count_lookup = np.zeros(lookup_size, dtype=np.int32)
    valid_lookup = np.zeros(lookup_size, dtype=bool)
    usable_codes = correction_codes[correction_valid]
    lower_lookup[usable_codes] = corrections.loc[
        correction_valid, "residual_q10"
    ].to_numpy(dtype=np.float64)
    upper_lookup[usable_codes] = corrections.loc[
        correction_valid, "residual_q90"
    ].to_numpy(dtype=np.float64)
    count_lookup[usable_codes] = corrections.loc[
        correction_valid, "calibration_n"
    ].to_numpy(dtype=np.int32)
    valid_lookup[usable_codes] = corrections.loc[
        correction_valid, "calibration_valid"
    ].to_numpy(dtype=bool)

    lower = lower_lookup[row_codes]
    upper = upper_lookup[row_codes]
    calibration_n = count_lookup[row_codes]
    calibration_valid = valid_lookup[row_codes]
    if not row_valid.all():
        lower[~row_valid] = np.nan
        upper[~row_valid] = np.nan
        calibration_n[~row_valid] = 0
        calibration_valid[~row_valid] = False

    q50 = result["q50"].to_numpy(dtype=np.float64, copy=False)
    np.add(lower, q50, out=lower)
    np.maximum(lower, 0.0, out=lower)
    np.add(upper, q50, out=upper)
    result["calibration_n"] = calibration_n
    result["calibration_valid"] = calibration_valid
    result["q10"] = lower
    result["q90"] = upper
    result["interval_calibrated"] = calibration_valid
    return result, corrections


def _shift_profile(
    actual: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    fraction: float,
    max_shift_steps: int,
    destination_capacity_factor: float = 1.0,
) -> tuple[np.ndarray, float, int, int]:
    """Plan from forecasts only, then score the frozen plan on realised demand.

    ``actual`` is not read until every source, destination, planned amount, and
    acceptance decision has been frozen. A realised-availability cap may reduce
    delivery at the preselected source, but it cannot change or reject a plan.
    Consequently, detrimental realised peak reductions remain negative.
    """
    planned = np.clip(signal.astype(float), 0.0, None)
    transfers: list[tuple[int, int, float]] = []
    rejected = 0
    trigger_order = np.argsort(signal)[::-1]
    used_sources: set[int] = set()
    used_destinations: set[int] = set()
    for source in trigger_order:
        if (
            signal[source] <= threshold
            or source in used_sources
            or source in used_destinations
        ):
            continue
        stop = min(len(signal), source + max_shift_steps + 1)
        candidates = np.arange(source + 1, stop)
        if not len(candidates):
            continue
        candidates = candidates[signal[candidates] < signal[source]]
        if not len(candidates):
            rejected += 1
            continue
        destination = int(candidates[np.argmin(signal[candidates])])
        capacity = float(threshold) * float(destination_capacity_factor)
        available_capacity = max(0.0, capacity - planned[destination])
        amount = min(max(0.0, float(fraction) * planned[source]), available_capacity)
        if amount <= 0:
            rejected += 1
            continue
        trial = planned.copy()
        trial[source] -= amount
        trial[destination] += amount
        if np.max(trial) > np.max(planned) + 1e-12:
            rejected += 1
            continue
        planned = trial
        transfers.append((int(source), destination, float(amount)))
        used_sources.add(int(source))
        used_destinations.add(destination)

    scheduled = actual.astype(float).copy()
    delivered_energy = 0.0
    for source, destination, planned_amount in transfers:
        # This is an execution-time physical cap, not a planning decision.
        delivered = min(planned_amount, max(0.0, float(scheduled[source])))
        scheduled[source] -= delivered
        scheduled[destination] += delivered
        delivered_energy += delivered
    return scheduled, delivered_energy, len(transfers), rejected


def run_peak_management(
    portfolios: pd.DataFrame,
    cfg: dict[str, Any],
    detail_path: Path | None = None,
    detail_chunk_rows: int = 25_000,
    collect_detail: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_columns = [
        "portfolio_id", "cohort", "portfolio_type", "portfolio_size", "model",
        "split", "origin_time", "strategy", "flexible_fraction", "shift_hours",
        "threshold",
        "peak_before", "peak_after", "peak_reduction", "peak_reduction_fraction",
        "load_factor_before", "load_factor_after", "shifted_energy", "interventions",
        "rejected_interventions", "energy_balance_error", "actual_peak_event",
        "predicted_peak_event", "missed_event", "false_intervention",
    ]
    summary_keys = [
        "cohort",
        "portfolio_type",
        "portfolio_size",
        "model",
        "split",
        "strategy",
        "flexible_fraction",
        "shift_hours",
    ]

    def write_empty_detail() -> pd.DataFrame:
        empty = pd.DataFrame(columns=detail_columns)
        if detail_path is not None:
            frame_write(empty, detail_path)
        return empty

    if portfolios.empty:
        return write_empty_detail(), pd.DataFrame()
    quantile = float(cfg["evaluation"]["peak_threshold_quantile"])
    required_columns = [
        "portfolio_id",
        "cohort",
        "portfolio_type",
        "portfolio_size",
        "model",
        "split",
        "origin_time",
        "horizon",
        "y_true",
        "q50",
        "q90",
    ]
    if "calibration_valid" in portfolios:
        portfolios = portfolios.loc[
            portfolios["calibration_valid"].fillna(False), required_columns
        ]
    else:
        portfolios = portfolios.loc[:, required_columns]
    if portfolios.empty:
        return write_empty_detail(), pd.DataFrame()
    calibration = portfolios[
        (portfolios["cohort"] == "seen") & (portfolios["split"] == "calibration")
    ]
    # The operational threshold is a property of the portfolio stratum, not of
    # the forecasting model. Remove model-duplicated realised rows before the
    # calibration quantile so every model is assessed against the same value.
    threshold_keys = ["portfolio_type", "portfolio_size"]
    calibration = calibration.drop_duplicates(
        [
            "portfolio_id", "portfolio_type", "portfolio_size", "origin_time",
            "horizon", "y_true",
        ]
    )
    thresholds = (
        calibration.groupby(threshold_keys, observed=True)["y_true"]
        .quantile(quantile)
        .rename("threshold")
        .reset_index()
    )
    test = portfolios.loc[
        portfolios["split"].isin(["seen_test", "unseen_test"])
    ].copy()
    test_codes, levels, test_valid = _dense_key_codes(test, threshold_keys)
    threshold_codes, _, threshold_valid = _dense_key_codes(
        thresholds, threshold_keys, levels=levels
    )
    lookup_size = int(np.prod([max(len(level), 1) for level in levels]))
    threshold_lookup = np.full(lookup_size, np.nan, dtype=np.float64)
    threshold_lookup[threshold_codes[threshold_valid]] = thresholds.loc[
        threshold_valid, "threshold"
    ].to_numpy(dtype=np.float64)
    threshold_values = threshold_lookup[test_codes]
    threshold_values[~test_valid] = np.nan
    test["threshold"] = threshold_values
    group_keys = [
        "portfolio_id",
        "cohort",
        "portfolio_type",
        "portfolio_size",
        "model",
        "split",
        "origin_time",
    ]
    # Do not retain all origin/strategy rows in Python dictionaries.  A normal
    # run produces roughly 1.5 million detail rows and the dictionary overhead
    # alone can exhaust RAM.  Stream bounded chunks to a temporary Parquet file
    # and maintain the much smaller summary online.
    rows: list[dict[str, Any]] = []
    summary_state: dict[tuple[Any, ...], dict[str, Any]] = {}
    parquet_writer: Any = None
    temp_detail_path: Path | None = None

    def update_summary(row: dict[str, Any]) -> None:
        key = tuple(row[name] for name in summary_keys)
        state = summary_state.get(key)
        if state is None:
            state = {
                "count": 0,
                "peak_sum": 0.0,
                "peak_values": [],
                "peak_fraction_sum": 0.0,
                "peak_min": float("inf"),
                "load_before_sum": 0.0,
                "load_after_sum": 0.0,
                "shifted_sum": 0.0,
                "missed_sum": 0,
                "false_sum": 0,
                "interventions_sum": 0,
                "rejected_sum": 0,
                "energy_error_max": 0.0,
            }
            summary_state[key] = state
        peak_reduction = float(row["peak_reduction"])
        state["count"] += 1
        state["peak_sum"] += peak_reduction
        state["peak_values"].append(peak_reduction)
        state["peak_fraction_sum"] += float(row["peak_reduction_fraction"])
        state["peak_min"] = min(state["peak_min"], peak_reduction)
        state["load_before_sum"] += float(row["load_factor_before"])
        state["load_after_sum"] += float(row["load_factor_after"])
        state["shifted_sum"] += float(row["shifted_energy"])
        state["missed_sum"] += int(bool(row["missed_event"]))
        state["false_sum"] += int(bool(row["false_intervention"]))
        state["interventions_sum"] += int(row["interventions"])
        state["rejected_sum"] += int(row["rejected_interventions"])
        state["energy_error_max"] = max(
            state["energy_error_max"], float(row["energy_balance_error"])
        )

    def flush_rows() -> None:
        nonlocal parquet_writer, temp_detail_path
        if not rows or detail_path is None:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Memory-safe peak detail output requires pyarrow."
            ) from exc
        chunk = pd.DataFrame.from_records(rows, columns=detail_columns)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if parquet_writer is None:
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            temp_detail_path = detail_path.with_name(detail_path.name + ".tmp")
            parquet_writer = pq.ParquetWriter(
                temp_detail_path, table.schema, compression="snappy"
            )
        parquet_writer.write_table(table)
        rows.clear()

    def emit(row: dict[str, Any]) -> None:
        update_summary(row)
        if collect_detail or detail_path is not None:
            rows.append(row)
        if detail_path is not None and len(rows) >= int(detail_chunk_rows):
            flush_rows()

    for keys, group in test.groupby(
        group_keys, dropna=False, observed=True, sort=False
    ):
        group = group.sort_values("horizon")
        if len(group) < int(cfg["features"]["forecast_steps"]):
            continue
        actual = group["y_true"].to_numpy(dtype=float)
        threshold = float(group["threshold"].iloc[0])
        if not np.isfinite(threshold):
            continue
        base = dict(zip(group_keys, keys))
        actual_event = bool(np.max(actual) > threshold)
        emit(
            {
                **base,
                "strategy": "reference_no_forecast",
                "flexible_fraction": 0.0,
                "shift_hours": 0,
                "threshold": threshold,
                "peak_before": float(np.max(actual)),
                "peak_after": float(np.max(actual)),
                "peak_reduction": 0.0,
                "load_factor_before": float(np.mean(actual) / max(np.max(actual), 1e-12)),
                "load_factor_after": float(np.mean(actual) / max(np.max(actual), 1e-12)),
                "shifted_energy": 0.0,
                "interventions": 0,
                "rejected_interventions": 0,
                "energy_balance_error": 0.0,
                "peak_reduction_fraction": 0.0,
                "actual_peak_event": actual_event,
                "predicted_peak_event": False,
                "missed_event": actual_event,
                "false_intervention": False,
            }
        )
        for strategy, signal_col in [
            ("median", "q50"),
            ("conformal_upper_bound", "q90"),
        ]:
            signal = group[signal_col].to_numpy(dtype=float)
            predicted_event = bool(np.max(signal) > threshold)
            for fraction in cfg["portfolio"]["flexible_fractions"]:
                for shift_hours in cfg["portfolio"]["shift_hours"]:
                    scheduled, shifted, count, rejected = _shift_profile(
                        actual,
                        signal,
                        threshold,
                        float(fraction),
                        int(shift_hours * 4),
                        float(cfg["portfolio"].get("destination_capacity_factor", 1.0)),
                    )
                    peak_before = float(np.max(actual))
                    peak_after = float(np.max(scheduled))
                    emit(
                        {
                            **base,
                            "strategy": strategy,
                            "flexible_fraction": float(fraction),
                            "shift_hours": int(shift_hours),
                            "threshold": threshold,
                            "peak_before": peak_before,
                            "peak_after": peak_after,
                            "peak_reduction": peak_before - peak_after,
                            "peak_reduction_fraction": (peak_before - peak_after) / max(peak_before, 1e-12),
                            "load_factor_before": float(np.mean(actual) / max(np.max(actual), 1e-12)),
                            "load_factor_after": float(np.mean(scheduled) / max(np.max(scheduled), 1e-12)),
                            "shifted_energy": shifted,
                            "interventions": count,
                            "rejected_interventions": rejected,
                            "energy_balance_error": float(abs(np.sum(scheduled) - np.sum(actual))),
                            "actual_peak_event": actual_event,
                            "predicted_peak_event": predicted_event,
                            "missed_event": actual_event and not predicted_event,
                            "false_intervention": predicted_event and not actual_event,
                        }
                    )

    # Non-deployable diagnostic benchmark: the realised future trajectory is
    # supplied to the exact same dispatch rule.  Realised rows are de-duplicated
    # across forecasting models so the oracle is counted once per origin.
    oracle_group_keys = [
        "portfolio_id",
        "cohort",
        "portfolio_type",
        "portfolio_size",
        "split",
        "origin_time",
    ]
    oracle_columns = oracle_group_keys + ["horizon", "y_true", "threshold"]
    oracle_consistency = test.groupby(
        oracle_group_keys + ["horizon"], dropna=False, observed=True
    )[["y_true", "threshold"]].nunique(dropna=False)
    if (oracle_consistency > 1).any().any():
        raise RuntimeError(
            "Model-duplicated realised portfolio trajectories are inconsistent"
        )
    oracle_test = test[oracle_columns].drop_duplicates(
        oracle_group_keys + ["horizon"]
    )
    for keys, group in oracle_test.groupby(
        oracle_group_keys, dropna=False, observed=True, sort=False
    ):
        group = group.sort_values("horizon")
        if len(group) < int(cfg["features"]["forecast_steps"]):
            continue
        actual = group["y_true"].to_numpy(dtype=float)
        threshold = float(group["threshold"].iloc[0])
        if not np.isfinite(threshold):
            continue
        base = dict(zip(oracle_group_keys, keys))
        base["model"] = "perfect_foresight"
        actual_event = bool(np.max(actual) > threshold)
        for fraction in cfg["portfolio"]["flexible_fractions"]:
            for shift_hours in cfg["portfolio"]["shift_hours"]:
                scheduled, shifted, count, rejected = _shift_profile(
                    actual,
                    actual,
                    threshold,
                    float(fraction),
                    int(shift_hours * 4),
                    float(cfg["portfolio"].get("destination_capacity_factor", 1.0)),
                )
                peak_before = float(np.max(actual))
                peak_after = float(np.max(scheduled))
                energy_error = float(abs(np.sum(scheduled) - np.sum(actual)))
                if peak_after > peak_before + 1e-9:
                    raise RuntimeError(
                        "Perfect-foresight dispatch increased the realised peak"
                    )
                if energy_error > 1e-8:
                    raise RuntimeError(
                        "Perfect-foresight dispatch violated energy conservation"
                    )
                emit(
                    {
                        **base,
                        "strategy": "perfect_foresight",
                        "flexible_fraction": float(fraction),
                        "shift_hours": int(shift_hours),
                        "threshold": threshold,
                        "peak_before": peak_before,
                        "peak_after": peak_after,
                        "peak_reduction": peak_before - peak_after,
                        "peak_reduction_fraction": (peak_before - peak_after)
                        / max(peak_before, 1e-12),
                        "load_factor_before": float(
                            np.mean(actual) / max(peak_before, 1e-12)
                        ),
                        "load_factor_after": float(
                            np.mean(scheduled) / max(peak_after, 1e-12)
                        ),
                        "shifted_energy": shifted,
                        "interventions": count,
                        "rejected_interventions": rejected,
                        "energy_balance_error": energy_error,
                        "actual_peak_event": actual_event,
                        "predicted_peak_event": actual_event,
                        "missed_event": False,
                        "false_intervention": False,
                    }
                )
    if not summary_state:
        return write_empty_detail(), pd.DataFrame()

    if detail_path is not None:
        flush_rows()
        if parquet_writer is not None:
            parquet_writer.close()
            parquet_writer = None
        if temp_detail_path is None:
            return write_empty_detail(), pd.DataFrame()
        temp_detail_path.replace(detail_path)
        # The complete detail is on disk.  Return only a small integrity frame;
        # callers should use the streamed artifact rather than materialising it.
        detail = pd.DataFrame(
            {
                "peak_reduction": [
                    min(state["peak_min"] for state in summary_state.values())
                ],
                "energy_balance_error": [
                    max(state["energy_error_max"] for state in summary_state.values())
                ],
            }
        )
    elif collect_detail:
        detail = pd.DataFrame.from_records(rows, columns=detail_columns)
    else:
        detail = pd.DataFrame(columns=detail_columns)

    summary_rows: list[dict[str, Any]] = []
    for key, state in summary_state.items():
        count = int(state["count"])
        row = dict(zip(summary_keys, key))
        row.update(
            {
                "forecast_origins": count,
                "mean_peak_reduction": state["peak_sum"] / count,
                "median_peak_reduction": float(np.median(state["peak_values"])),
                "mean_peak_reduction_fraction": state["peak_fraction_sum"] / count,
                "minimum_peak_reduction": state["peak_min"],
                "mean_load_factor_change": (
                    state["load_after_sum"] - state["load_before_sum"]
                ) / count,
                "total_shifted_energy": state["shifted_sum"],
                "missed_events": state["missed_sum"],
                "false_interventions": state["false_sum"],
                "interventions": state["interventions_sum"],
                "rejected_interventions": state["rejected_sum"],
                "maximum_energy_balance_error": state["energy_error_max"],
                "mean_load_factor_before": state["load_before_sum"] / count,
            }
        )
        summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def run_portfolio_analysis(
    predictions: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    memberships = sample_portfolios(predictions, split_manifest, cfg)
    aggregate = aggregate_predictions(
        predictions, memberships, cfg["portfolio"]["min_household_coverage"]
    )
    calibrated, corrections = calibrate_portfolio_intervals(
        aggregate,
        cfg["models"]["conformal"]["alpha"],
        int(cfg["portfolio"].get("min_calibration_samples", 100)),
    )
    detail_path = output_dir / "peak_management_detail.parquet"
    _, summary = run_peak_management(calibrated, cfg, detail_path=detail_path)
    primary_threshold = float(cfg["evaluation"]["peak_threshold_quantile"])
    sensitivity_parts: list[pd.DataFrame] = [
        summary.assign(peak_threshold_quantile=primary_threshold)
    ]
    for threshold_quantile in cfg["portfolio"].get(
        "peak_threshold_quantiles", [primary_threshold]
    ):
        threshold_quantile = float(threshold_quantile)
        if np.isclose(threshold_quantile, primary_threshold):
            continue
        sensitivity_cfg = copy.deepcopy(cfg)
        sensitivity_cfg["evaluation"]["peak_threshold_quantile"] = threshold_quantile
        _, sensitivity_summary = run_peak_management(
            calibrated,
            sensitivity_cfg,
            collect_detail=False,
        )
        sensitivity_parts.append(
            sensitivity_summary.assign(
                peak_threshold_quantile=threshold_quantile
            )
        )
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True, sort=False)
    frame_write(memberships, output_dir / "portfolio_memberships.parquet")
    frame_write(calibrated, output_dir / "portfolio_predictions.parquet")
    frame_write(corrections, output_dir / "portfolio_interval_corrections.parquet")
    frame_write(summary, output_dir / "peak_management_summary.parquet")
    frame_write(
        sensitivity,
        output_dir / "peak_management_sensitivity.parquet",
    )
    sensitivity.to_csv(
        output_dir / "peak_management_sensitivity.csv", index=False
    )
    atomic_json(
        {
            "schema_version": 1,
            "threshold_partition": "seen-household temporal calibration",
            "threshold_grouping": ["portfolio_type", "portfolio_size"],
            "threshold_quantile": float(cfg["evaluation"]["peak_threshold_quantile"]),
            "sensitivity_threshold_quantiles": [
                float(value)
                for value in cfg["portfolio"].get(
                    "peak_threshold_quantiles",
                    [cfg["evaluation"]["peak_threshold_quantile"]],
                )
            ],
            "sensitivity_flexible_fractions": [
                float(value) for value in cfg["portfolio"]["flexible_fractions"]
            ],
            "sensitivity_shift_hours": [
                int(value) for value in cfg["portfolio"]["shift_hours"]
            ],
            "decision_signals": [
                "median",
                "conformal_upper_bound",
                "perfect_foresight_non_deployable_benchmark",
            ],
            "source_selection": "forecast values above threshold, descending",
            "destination_selection": "lowest forecast value in the forward window",
            "destination_capacity": "threshold * destination_capacity_factor minus planned destination load",
            "acceptance": "forecast-feasible and forecast running maximum non-increasing",
            "realised_demand_role": "ex-post scoring and source-availability cap only",
            "perfect_foresight_role": (
                "diagnostic upper benchmark using realised future load as the "
                "decision signal under the identical dispatch constraints"
            ),
            "negative_realised_peak_reductions_retained": True,
            "energy_direction": "forward only",
            "relative_peak_reduction_denominator": "realised pre-intervention peak",
        },
        output_dir / "portfolio_protocol.json",
    )
    return calibrated, summary
