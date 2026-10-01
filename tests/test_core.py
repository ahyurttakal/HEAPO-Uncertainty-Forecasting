from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from heapo_forecasting.conformal import ConditionalConformalizer, HorizonConformalizer
from heapo_forecasting.features import (
    history_features,
    make_supervised_rows,
    restrict_positions_to_target_end,
    stratified_evaluation_positions,
    valid_origin_positions,
)
from heapo_forecasting.metrics import (
    _seasonal_scale,
    point_probabilistic_metrics,
    three_quantile_crps_approximation,
)
from heapo_forecasting.models import TabularEncoder, select_feature_columns
from heapo_forecasting.portfolio import (
    _shift_profile,
    aggregate_predictions,
    calibrate_portfolio_intervals,
    run_peak_management,
)
from heapo_forecasting.splits import assign_split
from heapo_forecasting.tft_model import (
    _bounded_sequence,
    _evaluation_sequences,
    _training_and_early_stop_blocks,
)
from heapo_forecasting.statistics import paired_block_bootstrap
from heapo_forecasting.statistics import (
    add_holm_correction,
    hierarchical_metric_intervals,
    hierarchical_paired_bootstrap,
)
from heapo_forecasting.finalize import portfolio_calibration_sensitivity
from heapo_forecasting.leakage import LeakageProtocolError, validate_supervised_protocol


def config() -> dict:
    return {
        "project": {"seed": 7},
        "data": {
            "target": "kWh_received_HeatPump",
            "timestamp": "Timestamp",
            "timezone": "Europe/Zurich",
            "weather_scenario": "operational",
        },
        "features": {
            "lag_steps": [1, 4, 96, 672],
            "rolling_steps": [4, 96],
            "history_steps": 672,
            "forecast_steps": 96,
            "on_threshold_kwh": 0.05,
        },
        "sampling": {
            "eval_clock_hours": [0, 6, 12, 18],
            "max_eval_origins_per_household": 8,
        },
        "models": {"conformal": {"alpha": 0.2, "min_samples_per_horizon": 2}},
    }


def panel(rows: int = 1000) -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", periods=rows, freq="15min", tz="UTC")
    values = 1.0 + 0.5 * np.sin(np.arange(rows) * 2 * np.pi / 96)
    frame = pd.DataFrame(
        {
            "Timestamp": timestamp,
            "kWh_received_HeatPump": values,
            "Household_ID": 1,
            "HP_Type": "ASHP",
            "Segment_ID": 0,
            "cal_hour_sin": np.sin(np.arange(rows) * 2 * np.pi / 96),
            "cal_hour_cos": np.cos(np.arange(rows) * 2 * np.pi / 96),
            "Temperature_mean_hourly": 5 + np.sin(np.arange(rows) * 2 * np.pi / 96),
            "weather_operational__Temperature_mean_hourly": np.nan,
            "weather_causal__Temperature_mean_hourly": np.nan,
            "static__Survey_Building_LivingArea": 150.0,
        }
    )
    frame["weather_operational__Temperature_mean_hourly"] = frame["Temperature_mean_hourly"].shift(96)
    frame["weather_causal__Temperature_mean_hourly"] = frame["Temperature_mean_hourly"].where(
        frame["Timestamp"].dt.minute.eq(0)
    ).ffill()
    return frame


class FeatureTests(unittest.TestCase):
    def test_tft_sequence_is_bounded_and_respects_cutoff(self) -> None:
        timestamps = pd.date_range("2020-01-01", periods=1000, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Sequence_ID": "1_0_0",
                "value": np.arange(1000),
            }
        )
        cutoff = timestamps[899]
        bounded = _bounded_sequence(frame, "Timestamp", cutoff, 800, 768, "train")
        self.assertEqual(len(bounded), 800)
        self.assertEqual(bounded["Timestamp"].max(), cutoff)
        self.assertEqual(bounded["Sequence_ID"].unique().tolist(), ["1_0_0__train"])

    def test_tft_evaluation_sequence_uses_exact_requested_origin(self) -> None:
        timestamps = pd.date_range("2020-01-01", periods=900, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Sequence_ID": "1_0_0",
                "Household_ID": 1,
                "HP_Type": "ASHP",
                "time_idx": np.arange(900),
            }
        )
        origins = pd.DataFrame(
            {"origin_time": [timestamps[700]], "split": ["seen_test"]}
        )
        sequence = _evaluation_sequences(frame, origins, "Timestamp", 672, 96)
        self.assertEqual(len(sequence), 768)
        self.assertEqual(sequence.iloc[671]["Timestamp"], timestamps[700])
        self.assertTrue(sequence["Sequence_ID"].iloc[0].startswith("eval__1__seen_test"))

    def test_tft_early_stop_targets_are_after_fitting_targets(self) -> None:
        timestamps = pd.date_range("2020-01-01", periods=3000, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Sequence_ID": "1_0_0",
                "Household_ID": 1,
                "HP_Type": "ASHP",
            }
        )
        fitting, early_stop, metadata = _training_and_early_stop_blocks(
            frame,
            "Timestamp",
            timestamps[-1],
            encoder_length=672,
            prediction_length=96,
            maximum_train_rows=2000,
            early_stop_fraction=0.10,
        )
        self.assertFalse(fitting.empty)
        self.assertFalse(early_stop.empty)
        self.assertLess(metadata["fit_target_end"], metadata["early_stop_target_start"])
        self.assertEqual(
            early_stop.iloc[672]["Timestamp"], metadata["early_stop_target_start"]
        )

    def test_encoder_imputes_feature_missing_from_prediction_household(self) -> None:
        train = pd.DataFrame(
            {
                "weather_pressure": [990.0, 1010.0],
                "building_type": ["A", "B"],
            }
        )
        encoder = TabularEncoder().fit(train)
        transformed = encoder.transform(pd.DataFrame({"building_type": ["A"]}))
        self.assertEqual(float(transformed.loc[0, "weather_pressure"]), 1000.0)
        self.assertEqual(int(transformed.loc[0, "building_type"]), 0)

    def test_future_target_does_not_change_origin_features(self) -> None:
        cfg = config()
        frame = panel()
        before = history_features(frame, cfg).iloc[800].copy()
        frame.loc[850:, "kWh_received_HeatPump"] = 999.0
        after = history_features(frame, cfg).iloc[800].copy()
        pd.testing.assert_series_equal(before, after)

    def test_supervised_rows_have_all_horizons(self) -> None:
        cfg = config()
        frame = panel()
        positions = valid_origin_positions(frame, cfg, stride=96)
        rows = make_supervised_rows(frame, positions[:1], [1, 4, 24, 96], cfg)
        self.assertEqual(rows["horizon"].tolist(), [1, 4, 24, 96])
        self.assertTrue((rows["target_time"] > rows["origin_time"]).all())
        origin_position = int(positions[0])
        self.assertAlmostEqual(
            rows.loc[0, "load_lag_1"], frame.loc[origin_position, "kWh_received_HeatPump"]
        )
        self.assertIn("evaluation_temperature_c", rows)
        self.assertIn("conformal_temperature_c", rows)
        self.assertNotIn("evaluation_temperature_c", select_feature_columns(rows))
        self.assertNotIn("conformal_temperature_c", select_feature_columns(rows))

    def test_operational_conformal_temperature_never_uses_realised_future_weather(self) -> None:
        cfg = config()
        frame = panel()
        positions = valid_origin_positions(frame, cfg, stride=96)
        origin = int(positions[0])
        target_positions = [origin + 1, origin + 96]
        operational_values = frame.loc[
            target_positions, "weather_operational__Temperature_mean_hourly"
        ].to_numpy(dtype=float)
        frame.loc[target_positions, "Temperature_mean_hourly"] = [91.0, 96.0]
        rows = make_supervised_rows(frame, np.asarray([origin]), [1, 96], cfg)
        np.testing.assert_allclose(
            rows["conformal_temperature_c"].to_numpy(dtype=float),
            operational_values,
        )
        np.testing.assert_allclose(
            rows["conformal_temperature_c"].to_numpy(dtype=float),
            rows["future_weather__Temperature_mean_hourly"].to_numpy(dtype=float),
        )
        np.testing.assert_allclose(
            rows["evaluation_temperature_c"].to_numpy(dtype=float),
            [91.0, 96.0],
        )

    def test_evaluation_origins_are_clock_stratified(self) -> None:
        cfg = config()
        frame = panel(rows=5000)
        positions = stratified_evaluation_positions(frame, cfg, max_origins=8)
        local_hours = (
            frame.iloc[positions]["Timestamp"]
            .dt.tz_convert("Europe/Zurich")
            .dt.hour
            .value_counts()
            .to_dict()
        )
        self.assertEqual(set(local_hours), {0, 6, 12, 18})
        self.assertEqual(set(local_hours.values()), {2})

    def test_complete_target_path_cannot_cross_split_cutoff(self) -> None:
        cfg = config()
        frame = panel(rows=1400)
        candidates = valid_origin_positions(frame, cfg, stride=1)
        cutoff = frame.loc[1100, "Timestamp"]
        selected = restrict_positions_to_target_end(
            frame, candidates, cfg, cutoff
        )
        self.assertTrue(len(selected) > 0)
        target_end = frame.iloc[selected + cfg["features"]["forecast_steps"]][
            "Timestamp"
        ]
        self.assertTrue((target_end <= cutoff).all())
        self.assertLessEqual(
            int(selected.max()) + cfg["features"]["forecast_steps"], 1100
        )

    def test_leakage_audit_fails_when_target_path_crosses_cutoff(self) -> None:
        cfg = config()
        cfg["pipeline"] = {"schema_version": 1}
        origin = pd.Timestamp("2020-01-01", tz="UTC")
        rows = pd.DataFrame(
            {
                "Household_ID": 1,
                "origin_time": [origin] * 96,
                "target_time": [
                    origin + pd.Timedelta(minutes=15 * horizon)
                    for horizon in range(1, 97)
                ],
                "horizon": range(1, 97),
                "split": "train",
            }
        )
        split = {
            "training_households": [1],
            "unseen_validation_households": [],
            "unseen_test_households": [],
            "train_end": (origin + pd.Timedelta(hours=23)).isoformat(),
            "calibration_end": (origin + pd.Timedelta(days=2)).isoformat(),
        }
        with TemporaryDirectory() as directory:
            shard = Path(directory) / "crossing.csv"
            rows.to_csv(shard, index=False)
            manifest = pd.DataFrame(
                {
                    "household_id": [1],
                    "path": [str(shard)],
                    "conformal_temperature_feature": [""],
                }
            )
            cfg["data"]["weather_scenario"] = "oracle"
            with self.assertRaises(LeakageProtocolError):
                validate_supervised_protocol(
                    manifest,
                    split,
                    cfg,
                    Path(directory) / "leakage_audit.json",
                )

    def test_imputed_origin_or_target_is_never_scored(self) -> None:
        cfg = config()
        frame = panel(rows=1200)
        frame["Target_OriginallyMissing"] = 0
        candidates = valid_origin_positions(frame, cfg, stride=1)
        origin = int(candidates[0])
        frame.loc[origin, "Target_OriginallyMissing"] = 1
        frame.loc[origin + 10, "Target_OriginallyMissing"] = 1
        selected = set(valid_origin_positions(frame, cfg, stride=1).tolist())
        self.assertNotIn(origin, selected)
        self.assertNotIn(origin + 9, selected)


class CalibrationTests(unittest.TestCase):
    def test_crps_proxy_uses_trapezoidal_quantile_integration(self) -> None:
        self.assertAlmostEqual(three_quantile_crps_approximation([1.0, 1.0, 1.0]), 1.6)

    def test_conformal_widens_interval(self) -> None:
        calibration = pd.DataFrame(
            {
                "horizon": [1, 1, 4, 4],
                "y_true": [2.0, 3.0, 4.0, 5.0],
                "q10": [1.5, 2.5, 3.5, 4.5],
                "q90": [1.7, 2.7, 3.7, 4.7],
            }
        )
        model = HorizonConformalizer(alpha=0.2, min_samples=2).fit(calibration)
        transformed = model.transform(calibration)
        self.assertTrue((transformed["q10"] <= calibration["q10"]).all())
        self.assertTrue((transformed["q90"] >= calibration["q90"]).all())

    def test_metric_coverage(self) -> None:
        frame = pd.DataFrame(
            {"y_true": [1.0, 2.0], "y_pred": [1.0, 2.0], "q10": [0.5, 1.5], "q50": [1.0, 2.0], "q90": [1.5, 2.5]}
        )
        metrics = point_probabilistic_metrics(frame)
        self.assertEqual(metrics["picp"], 1.0)
        self.assertEqual(metrics["mae"], 0.0)

    def test_conditional_conformal_calibrates_every_model(self) -> None:
        rows = []
        for model in ["lightgbm", "seasonal_daily"]:
            for hp_type in ["ASHP", "GSHP"]:
                for value in range(8):
                    rows.append(
                        {
                            "model": model,
                            "HP_Type": hp_type,
                            "horizon": 1,
                            "temperature_c": -5.0 if value < 4 else 5.0,
                            "y_true": 2.0,
                            "q10": 1.0,
                            "q90": 1.0,
                        }
                    )
        calibration = pd.DataFrame(rows)
        model = ConditionalConformalizer(0.2, 2, 0.5).fit(calibration)
        transformed = model.transform(calibration)
        self.assertEqual(set(transformed["model"]), {"lightgbm", "seasonal_daily"})
        self.assertTrue((transformed["q90"] >= transformed["y_true"]).all())
        self.assertTrue(transformed["interval_calibrated"].all())

    def test_mase_uses_row_specific_household_scale(self) -> None:
        frame = pd.DataFrame(
            {
                "y_true": [2.0, 2.0],
                "y_pred": [1.0, 1.0],
                "q10": [1.0, 1.0],
                "q50": [1.0, 1.0],
                "q90": [1.0, 1.0],
                "mase_scale": [1.0, 2.0],
            }
        )
        self.assertAlmostEqual(point_probabilistic_metrics(frame)["mase"], 0.75)

    def test_daily_scale_does_not_cross_segment(self) -> None:
        timestamps = pd.date_range("2020-01-01", periods=200, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "target": np.arange(200, dtype=float),
                "Segment_ID": [0] * 100 + [1] * 100,
            }
        )
        scale, count = _seasonal_scale(frame, "target", "Timestamp", 96)
        self.assertEqual(count, 8)
        self.assertEqual(scale, 96.0)


class SplitAndPortfolioTests(unittest.TestCase):
    def test_unseen_house_overrides_time(self) -> None:
        rows = pd.DataFrame(
            {"Household_ID": [1, 2], "origin_time": pd.to_datetime(["2020-01-01", "2020-01-01"], utc=True)}
        )
        split = {
            "train_end": "2021-01-01T00:00:00+00:00",
            "calibration_end": "2021-06-01T00:00:00+00:00",
            "unseen_validation_households": [],
            "unseen_test_households": [2],
        }
        self.assertEqual(assign_split(rows, split).tolist(), ["train", "unseen_test"])

    def test_portfolio_interval_uses_aggregate_residuals(self) -> None:
        predictions = pd.DataFrame(
            {
                "Household_ID": [1, 2, 1, 2],
                "model": "m",
                "split": "calibration",
                "origin_time": pd.to_datetime(["2020-01-01"] * 4, utc=True),
                "target_time": pd.to_datetime(["2020-01-01 00:15", "2020-01-01 00:15", "2020-01-01 00:30", "2020-01-01 00:30"], utc=True),
                "horizon": [1, 1, 2, 2],
                "y_true": [1.0, 2.0, 1.5, 2.5],
                "y_pred": [1.0, 1.0, 1.0, 2.0],
                "q50": [1.0, 1.0, 1.0, 2.0],
            }
        )
        memberships = pd.DataFrame(
            {
                "portfolio_id": ["p", "p"], "cohort": ["seen", "seen"],
                "portfolio_type": ["mixed", "mixed"], "portfolio_size": [2, 2],
                "repetition": [0, 0], "Household_ID": [1, 2],
            }
        )
        aggregate = aggregate_predictions(predictions, memberships, 1.0)
        calibrated, corrections = calibrate_portfolio_intervals(aggregate, min_samples=1)
        self.assertFalse(corrections.empty)
        self.assertTrue((calibrated["q90"] >= calibrated["q50"]).all())

    def test_portfolios_are_aggregated_independently(self) -> None:
        predictions = pd.DataFrame(
            {
                "Household_ID": [1, 2],
                "model": ["m", "m"],
                "split": ["calibration", "calibration"],
                "origin_time": pd.to_datetime(["2020-01-01"] * 2, utc=True),
                "target_time": pd.to_datetime(["2020-01-01 00:15"] * 2, utc=True),
                "horizon": [1, 1],
                "y_true": [1.0, 2.0],
                "y_pred": [1.0, 2.0],
                "q50": [1.0, 2.0],
            }
        )
        memberships = pd.DataFrame(
            {
                "portfolio_id": ["both", "both", "one"],
                "cohort": ["seen", "seen", "seen"],
                "portfolio_type": ["mixed", "mixed", "ASHP"],
                "portfolio_size": [2, 2, 1],
                "repetition": [0, 0, 0],
                "Household_ID": [1, 2, 1],
            }
        )
        aggregate = aggregate_predictions(predictions, memberships, 1.0)
        totals = aggregate.set_index("portfolio_id")["y_true"].to_dict()
        self.assertEqual(totals, {"both": 3.0, "one": 1.0})

    def test_peak_shift_conserves_energy(self) -> None:
        actual = np.asarray([5.0, 1.0, 1.0, 1.0])
        signal = np.asarray([6.0, 1.0, 2.0, 3.0])
        scheduled, shifted, interventions, _ = _shift_profile(
            actual, signal, threshold=4.0, fraction=0.2, max_shift_steps=3
        )
        self.assertAlmostEqual(float(scheduled.sum()), float(actual.sum()))
        self.assertGreater(shifted, 0)
        self.assertEqual(interventions, 1)

    def test_peak_shift_retains_negative_realised_reduction(self) -> None:
        actual = np.asarray([1.0, 5.0])
        signal = np.asarray([6.0, 1.0])
        scheduled, shifted, interventions, _ = _shift_profile(
            actual, signal, threshold=4.0, fraction=0.2, max_shift_steps=1
        )
        self.assertAlmostEqual(float(scheduled.sum()), float(actual.sum()))
        self.assertEqual(interventions, 1)
        self.assertGreater(shifted, 0)
        self.assertLess(float(actual.max() - scheduled.max()), 0.0)

    def test_perfect_foresight_is_unique_and_error_free(self) -> None:
        rows = []
        origin_cal = pd.Timestamp("2020-01-01", tz="UTC")
        origin_test = pd.Timestamp("2020-01-02", tz="UTC")
        actual = [1.0, 5.0, 1.0, 1.0]
        for model in ["lightgbm", "persistence"]:
            for split, origin, values in [
                ("calibration", origin_cal, [1.0, 2.0, 3.0, 4.0]),
                ("seen_test", origin_test, actual),
            ]:
                for horizon, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "portfolio_id": "p1",
                            "cohort": "seen",
                            "portfolio_type": "ASHP",
                            "portfolio_size": 4,
                            "model": model,
                            "split": split,
                            "origin_time": origin,
                            "horizon": horizon,
                            "y_true": value,
                            "q50": value,
                            "q90": value,
                            "calibration_valid": True,
                        }
                    )
        cfg = config()
        cfg["features"]["forecast_steps"] = 4
        cfg["evaluation"] = {"peak_threshold_quantile": 0.5}
        cfg["portfolio"] = {
            "flexible_fractions": [0.2],
            "shift_hours": [1],
            "destination_capacity_factor": 1.0,
        }
        detail, summary = run_peak_management(pd.DataFrame(rows), cfg)
        oracle = detail[
            detail["model"].eq("perfect_foresight")
            & detail["strategy"].eq("perfect_foresight")
        ]
        self.assertEqual(len(oracle), 1)
        self.assertFalse(bool(oracle["missed_event"].any()))
        self.assertFalse(bool(oracle["false_intervention"].any()))
        self.assertGreaterEqual(float(oracle.iloc[0]["peak_reduction"]), 0.0)
        self.assertLessEqual(float(oracle.iloc[0]["energy_balance_error"]), 1e-8)
        self.assertEqual(
            int(summary[summary.model.eq("perfect_foresight")].forecast_origins.sum()),
            1,
        )


class StatisticalTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        rows = []
        for household in [1, 2]:
            for day in range(4):
                origin = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=day)
                for model, prediction in [("better", 1.0), ("worse", 0.0)]:
                    rows.append(
                        {
                            "Household_ID": household,
                            "origin_time": origin,
                            "target_time": origin + pd.Timedelta(minutes=15),
                            "horizon": 1,
                            "split": "seen_test",
                            "HP_Type": "ASHP",
                            "model": model,
                            "y_true": 1.0,
                            "y_pred": prediction,
                        }
                    )
        frame = pd.DataFrame(rows)
        first = paired_block_bootstrap(frame, [("better", "worse")], repetitions=100, seed=3)
        second = paired_block_bootstrap(frame, [("better", "worse")], repetitions=100, seed=3)
        pd.testing.assert_frame_equal(first, second)
        self.assertLess(float(first.loc[0, "mae_delta"]), 0)

    def test_household_cluster_and_nested_bootstrap_are_deterministic(self) -> None:
        rows = []
        for household, error in [(1, 0.1), (2, 1.0), (3, 0.4)]:
            for day in range(3):
                origin = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=day)
                rows.append(
                    {
                        "Household_ID": household,
                        "origin_time": origin,
                        "target_time": origin + pd.Timedelta(minutes=15),
                        "horizon": 1,
                        "split": "unseen_test",
                        "HP_Type": "ASHP",
                        "model": "model",
                        "y_true": 1.0,
                        "y_pred": 1.0 - error,
                        "q10": 0.0,
                        "q90": 2.0,
                    }
                )
        frame = pd.DataFrame(rows)
        cluster = hierarchical_metric_intervals(
            frame, repetitions=100, seed=11, timezone="Europe/Zurich"
        )
        nested_first = hierarchical_metric_intervals(
            frame,
            repetitions=100,
            seed=11,
            timezone="Europe/Zurich",
            nested_days=True,
        )
        nested_second = hierarchical_metric_intervals(
            frame,
            repetitions=100,
            seed=11,
            timezone="Europe/Zurich",
            nested_days=True,
        )
        self.assertEqual(int(cluster.loc[0, "n_households"]), 3)
        self.assertEqual(int(cluster.loc[0, "n_household_days"]), 9)
        self.assertEqual(cluster.loc[0, "resampling_scheme"], "household_cluster")
        pd.testing.assert_frame_equal(nested_first, nested_second)

    def test_hierarchical_paired_bootstrap_and_holm(self) -> None:
        rows = []
        for household in [1, 2, 3]:
            for day in range(3):
                origin = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=day)
                for model, prediction in [("better", 1.0), ("worse", 0.0)]:
                    rows.append(
                        {
                            "Household_ID": household,
                            "origin_time": origin,
                            "target_time": origin + pd.Timedelta(minutes=15),
                            "horizon": 1,
                            "split": "seen_test",
                            "HP_Type": "ASHP",
                            "model": model,
                            "y_true": 1.0,
                            "y_pred": prediction,
                        }
                    )
        result = hierarchical_paired_bootstrap(
            pd.DataFrame(rows),
            [("better", "worse")],
            repetitions=100,
            nested_days=True,
            seed=5,
        )
        self.assertEqual(int(result.loc[0, "n_households"]), 3)
        self.assertLess(float(result.loc[0, "mae_delta"]), 0)
        self.assertTrue(bool(result.loc[0, "significant_holm"]))
        self.assertFalse(bool(result.loc[0, "confirmatory_holm"]))

        tests = pd.DataFrame(
            {
                "candidate": ["a", "a", "a"],
                "comparator": ["b", "b", "b"],
                "mae_delta": [-1.0, -1.0, 1.0],
                "bootstrap_p_value_two_sided": [0.01, 0.03, 0.20],
            }
        )
        adjusted = add_holm_correction(tests)
        np.testing.assert_allclose(adjusted["holm_p_value"], [0.03, 0.06, 0.20])

    def test_portfolio_calibration_sensitivity_flags_small_cells(self) -> None:
        corrections = pd.DataFrame(
            {
                "portfolio_type": ["GSHP", "GSHP", "ASHP", "ASHP"],
                "portfolio_size": [10, 10, 4, 4],
                "model": ["m1", "m2", "m1", "m2"],
                "horizon": [1, 1, 1, 1],
                "calibration_n": [28, 28, 62, 62],
                "calibration_rows": [100, 100, 200, 200],
            }
        )
        sensitivity = portfolio_calibration_sensitivity(
            corrections, thresholds=[20, 30, 50]
        )
        gshp_30 = sensitivity[
            (sensitivity["portfolio_type"] == "GSHP")
            & (sensitivity["min_calibration_samples"] == 30)
        ].iloc[0]
        ashp_50 = sensitivity[
            (sensitivity["portfolio_type"] == "ASHP")
            & (sensitivity["min_calibration_samples"] == 50)
        ].iloc[0]
        self.assertFalse(bool(gshp_30["all_cells_reportable"]))
        self.assertTrue(bool(ashp_50["all_cells_reportable"]))


if __name__ == "__main__":
    unittest.main()
