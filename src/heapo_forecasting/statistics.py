from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_KEYS = ["Household_ID", "origin_time", "target_time", "horizon", "split", "HP_Type"]


def _local_day(frame: pd.DataFrame, timezone: str) -> pd.Series:
    origin = pd.to_datetime(frame["origin_time"], utc=True)
    return origin.dt.tz_convert(timezone).dt.floor("D")


def _hierarchical_ratio_draws(
    day_table: pd.DataFrame,
    numerator_columns: list[str],
    repetitions: int,
    rng: np.random.Generator,
    nested_days: bool,
) -> dict[str, np.ndarray]:
    """Bootstrap row-weighted ratios with households as primary clusters.

    ``day_table`` contains one row per household/day, one or more additive
    numerators and an additive denominator named ``_n``.  With
    ``nested_days=False`` every sampled household contributes all of its days.
    With ``nested_days=True`` local calendar days are independently resampled
    within every sampled household occurrence.
    """
    if day_table.empty:
        return {column: np.asarray([], dtype=float) for column in numerator_columns}
    grouped = list(day_table.groupby("Household_ID", sort=True, observed=True))
    household_count = len(grouped)
    household_draws = rng.integers(
        0, household_count, size=(repetitions, household_count)
    )
    flat_households = household_draws.ravel()
    denominator = np.zeros(flat_households.size, dtype=np.float64)
    numerators = {
        column: np.zeros(flat_households.size, dtype=np.float64)
        for column in numerator_columns
    }
    for household_index, (_, household_days) in enumerate(grouped):
        positions = np.flatnonzero(flat_households == household_index)
        if not len(positions):
            continue
        day_n = household_days["_n"].to_numpy(dtype=np.float64)
        day_values = {
            column: household_days[column].to_numpy(dtype=np.float64)
            for column in numerator_columns
        }
        if nested_days:
            day_count = len(household_days)
            day_draws = rng.integers(
                0, day_count, size=(len(positions), day_count)
            )
            denominator[positions] = day_n[day_draws].sum(axis=1)
            for column in numerator_columns:
                numerators[column][positions] = day_values[column][day_draws].sum(axis=1)
        else:
            denominator[positions] = float(day_n.sum())
            for column in numerator_columns:
                numerators[column][positions] = float(day_values[column].sum())
    denominator = denominator.reshape(repetitions, household_count).sum(axis=1)
    result: dict[str, np.ndarray] = {}
    for column in numerator_columns:
        numerator = numerators[column].reshape(repetitions, household_count).sum(axis=1)
        result[column] = np.divide(
            numerator,
            denominator,
            out=np.full(repetitions, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
    return result


def _holm_adjust(values: pd.Series) -> pd.Series:
    """Holm step-down family-wise error correction with monotonic adjustment."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    clean = values.dropna().astype(float)
    if clean.empty:
        return result
    ordered = clean.sort_values()
    count = len(ordered)
    adjusted = np.maximum.accumulate(
        np.asarray([(count - rank) * value for rank, value in enumerate(ordered)], dtype=float)
    )
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def add_holm_correction(
    tests: pd.DataFrame,
    p_column: str = "bootstrap_p_value_two_sided",
    family_columns: Iterable[str] = ("candidate", "comparator"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add Holm-adjusted p-values within each pre-specified model comparison."""
    result = tests.copy()
    if result.empty:
        result["holm_p_value"] = pd.Series(dtype=float)
        result["significant_holm"] = pd.Series(dtype=bool)
        return result
    columns = list(family_columns)
    result["holm_p_value"] = np.nan
    for _, index in result.groupby(columns, dropna=False, observed=True).groups.items():
        result.loc[index, "holm_p_value"] = _holm_adjust(result.loc[index, p_column])
    result["significant_holm"] = result["holm_p_value"] < float(alpha)
    if "cluster_inference_reliable" in result:
        result["confirmatory_holm"] = (
            result["significant_holm"]
            & result["cluster_inference_reliable"].fillna(False)
        )
    else:
        result["confirmatory_holm"] = result["significant_holm"]
    result["effect_direction"] = np.select(
        [result["mae_delta"] < 0, result["mae_delta"] > 0],
        ["candidate_better", "candidate_worse"],
        default="tie",
    )
    return result


def hierarchical_metric_intervals(
    predictions: pd.DataFrame,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
    timezone: str = "UTC",
    nested_days: bool = False,
    minimum_households_for_confirmatory: int = 20,
    group_columns: Iterable[str] = ("model", "split", "HP_Type", "horizon"),
) -> pd.DataFrame:
    """Household-cluster or nested household/day intervals for MAE and PICP."""
    columns = list(group_columns)
    required = [*columns, "Household_ID", "origin_time", "y_true", "y_pred", "q10", "q90"]
    working = predictions.loc[:, list(dict.fromkeys(required))].copy()
    working["_day"] = _local_day(working, timezone)
    working["_absolute_error"] = (working["y_true"] - working["y_pred"]).abs()
    working["_covered"] = (
        (working["y_true"] >= working["q10"])
        & (working["y_true"] <= working["q90"])
    ).astype(float)
    working["_n"] = 1
    interval_alpha = 1.0 - float(confidence)
    rows: list[dict[str, Any]] = []
    for group_index, (keys, group) in enumerate(
        working.groupby(columns, dropna=False, observed=True, sort=True)
    ):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        day = (
            group.groupby(["Household_ID", "_day"], observed=True, sort=True)
            .agg(
                _absolute_error=("_absolute_error", "sum"),
                _covered=("_covered", "sum"),
                _n=("_n", "sum"),
            )
            .reset_index()
        )
        rng = np.random.default_rng(seed + group_index * 1009 + int(nested_days) * 1_000_003)
        draws = _hierarchical_ratio_draws(
            day,
            ["_absolute_error", "_covered"],
            repetitions,
            rng,
            nested_days=nested_days,
        )
        mae_draws = draws["_absolute_error"]
        coverage_draws = draws["_covered"]
        row = dict(zip(columns, key_values))
        row.update(
            {
                "resampling_scheme": (
                    "household_then_local_day" if nested_days else "household_cluster"
                ),
                "n_rows": int(len(group)),
                "n_households": int(group["Household_ID"].nunique()),
                "n_household_days": int(len(day)),
                "cluster_inference_reliable": bool(
                    group["Household_ID"].nunique()
                    >= int(minimum_households_for_confirmatory)
                ),
                "mae": float(group["_absolute_error"].mean()),
                "mae_ci_low": float(np.nanquantile(mae_draws, interval_alpha / 2)),
                "mae_ci_high": float(np.nanquantile(mae_draws, 1 - interval_alpha / 2)),
                "picp": float(group["_covered"].mean()),
                "picp_ci_low": float(np.nanquantile(coverage_draws, interval_alpha / 2)),
                "picp_ci_high": float(np.nanquantile(coverage_draws, 1 - interval_alpha / 2)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchical_paired_bootstrap(
    predictions: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]] | None = None,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
    timezone: str = "UTC",
    nested_days: bool = False,
    multiple_testing_alpha: float = 0.05,
    minimum_households_for_confirmatory: int = 20,
) -> pd.DataFrame:
    """Paired model tests with households as the primary sampling unit."""
    required = [*DEFAULT_KEYS, "model", "y_true", "y_pred"]
    working = predictions.loc[:, list(dict.fromkeys(required))].copy()
    working["absolute_error"] = (working["y_true"] - working["y_pred"]).abs()
    models = sorted(working["model"].dropna().astype(str).unique().tolist())
    pairs = list(comparisons) if comparisons is not None else list(combinations(models, 2))
    interval_alpha = 1.0 - float(confidence)
    rows: list[dict[str, Any]] = []
    strata = ["split", "HP_Type", "horizon"]
    for stratum_index, (stratum, group) in enumerate(
        working.groupby(strata, dropna=False, observed=True, sort=True)
    ):
        pivot = group.pivot_table(
            index=DEFAULT_KEYS,
            columns="model",
            values="absolute_error",
            aggfunc="mean",
        )
        for pair_index, (candidate, comparator) in enumerate(pairs):
            if candidate not in pivot or comparator not in pivot:
                continue
            matched = pd.DataFrame(
                {
                    "Household_ID": pivot.index.get_level_values("Household_ID").to_numpy(),
                    "origin_time": pivot.index.get_level_values("origin_time").to_numpy(),
                    "_delta": (pivot[candidate] - pivot[comparator]).to_numpy(),
                }
            ).dropna()
            if matched.empty:
                continue
            matched["_day"] = _local_day(matched, timezone)
            matched["_n"] = 1
            day = (
                matched.groupby(["Household_ID", "_day"], observed=True, sort=True)
                .agg(_delta=("_delta", "sum"), _n=("_n", "sum"))
                .reset_index()
            )
            rng = np.random.default_rng(
                seed
                + stratum_index * 1009
                + pair_index * 9176
                + int(nested_days) * 1_000_003
            )
            draws = _hierarchical_ratio_draws(
                day,
                ["_delta"],
                repetitions,
                rng,
                nested_days=nested_days,
            )["_delta"]
            probability_better = float(np.nanmean(draws < 0))
            finite_repetitions = int(np.isfinite(draws).sum())
            lower_tail = (int(np.nansum(draws <= 0)) + 1) / (finite_repetitions + 1)
            upper_tail = (int(np.nansum(draws >= 0)) + 1) / (finite_repetitions + 1)
            p_value = float(min(1.0, 2 * min(lower_tail, upper_tail)))
            household_count = int(matched["Household_ID"].nunique())
            rows.append(
                {
                    "split": stratum[0],
                    "HP_Type": stratum[1],
                    "horizon": stratum[2],
                    "candidate": candidate,
                    "comparator": comparator,
                    "resampling_scheme": (
                        "household_then_local_day" if nested_days else "household_cluster"
                    ),
                    "n_pairs": int(len(matched)),
                    "n_households": household_count,
                    "n_household_days": int(len(day)),
                    "cluster_inference_reliable": bool(
                        household_count >= int(minimum_households_for_confirmatory)
                    ),
                    "mae_delta": float(matched["_delta"].mean()),
                    "ci_low": float(np.nanquantile(draws, interval_alpha / 2)),
                    "ci_high": float(np.nanquantile(draws, 1 - interval_alpha / 2)),
                    "probability_candidate_better": probability_better,
                    "bootstrap_p_value_two_sided": p_value,
                }
            )
    return add_holm_correction(
        pd.DataFrame(rows),
        alpha=multiple_testing_alpha,
    )


def _block_column(frame: pd.DataFrame) -> pd.Series:
    origin = pd.to_datetime(frame["origin_time"], utc=True)
    return frame["Household_ID"].astype(str) + "__" + origin.dt.strftime("%Y-%m-%d")


def _bootstrap_mean(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return np.asarray([], dtype=float)
    draw = rng.integers(0, clean.size, size=(repetitions, clean.size))
    return clean[draw].mean(axis=1)


def bootstrap_metric_intervals(
    predictions: pd.DataFrame,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
    group_columns: Iterable[str] = ("model", "split", "HP_Type", "horizon"),
) -> pd.DataFrame:
    """Household-day block bootstrap intervals for MAE and interval coverage."""
    working = predictions.copy()
    working["_block"] = _block_column(working)
    working["_absolute_error"] = (working["y_true"] - working["y_pred"]).abs()
    working["_covered"] = (
        (working["y_true"] >= working["q10"]) & (working["y_true"] <= working["q90"])
    ).astype(float)
    alpha = 1.0 - confidence
    rows: list[dict[str, Any]] = []
    columns = list(group_columns)
    for group_index, (keys, group) in enumerate(working.groupby(columns, dropna=False, observed=True)):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        block = group.groupby("_block")[["_absolute_error", "_covered"]].mean()
        rng = np.random.default_rng(seed + group_index)
        mae_draws = _bootstrap_mean(block["_absolute_error"].to_numpy(), repetitions, rng)
        coverage_draws = _bootstrap_mean(block["_covered"].to_numpy(), repetitions, rng)
        row = dict(zip(columns, key_values))
        row.update(
            {
                "n_rows": int(len(group)),
                "n_blocks": int(len(block)),
                "mae": float(group["_absolute_error"].mean()),
                "mae_ci_low": float(np.quantile(mae_draws, alpha / 2)),
                "mae_ci_high": float(np.quantile(mae_draws, 1 - alpha / 2)),
                "picp": float(group["_covered"].mean()),
                "picp_ci_low": float(np.quantile(coverage_draws, alpha / 2)),
                "picp_ci_high": float(np.quantile(coverage_draws, 1 - alpha / 2)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]] | None = None,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> pd.DataFrame:
    """Paired MAE differences on matching household/day forecast blocks.

    A negative delta means the candidate model has lower MAE than comparator.
    """
    working = predictions.copy()
    working["absolute_error"] = (working["y_true"] - working["y_pred"]).abs()
    models = sorted(working["model"].dropna().astype(str).unique().tolist())
    pairs = list(comparisons) if comparisons is not None else list(combinations(models, 2))
    alpha = 1.0 - confidence
    rows: list[dict[str, Any]] = []
    strata = ["split", "HP_Type", "horizon"]
    for stratum_index, (stratum, group) in enumerate(working.groupby(strata, dropna=False, observed=True)):
        pivot = group.pivot_table(index=DEFAULT_KEYS, columns="model", values="absolute_error", aggfunc="mean")
        origin_dates = pd.to_datetime(
            pivot.index.get_level_values("origin_time"), utc=True
        ).strftime("%Y-%m-%d")
        households = pivot.index.get_level_values("Household_ID").astype(str)
        block_id = pd.Index(households + "__" + origin_dates)
        for pair_index, (candidate, comparator) in enumerate(pairs):
            if candidate not in pivot or comparator not in pivot:
                continue
            matched = pd.DataFrame(
                {
                    "delta": pivot[candidate] - pivot[comparator],
                    "block": block_id,
                }
            ).dropna()
            if matched.empty:
                continue
            block_delta = matched.groupby("block")["delta"].mean().to_numpy(dtype=float)
            rng = np.random.default_rng(seed + stratum_index * 1009 + pair_index)
            draws = _bootstrap_mean(block_delta, repetitions, rng)
            probability_better = float(np.mean(draws < 0))
            p_value = float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
            rows.append(
                {
                    "split": stratum[0],
                    "HP_Type": stratum[1],
                    "horizon": stratum[2],
                    "candidate": candidate,
                    "comparator": comparator,
                    "n_pairs": int(len(matched)),
                    "n_blocks": int(len(block_delta)),
                    "mae_delta": float(matched["delta"].mean()),
                    "ci_low": float(np.quantile(draws, alpha / 2)),
                    "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
                    "probability_candidate_better": probability_better,
                    "bootstrap_p_value_two_sided": p_value,
                }
            )
    return pd.DataFrame(rows)
