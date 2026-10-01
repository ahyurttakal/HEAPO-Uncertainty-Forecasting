from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_manuscript_outputs.py"
SPEC = importlib.util.spec_from_file_location("heapo_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIGURES
SPEC.loader.exec_module(FIGURES)
FIGURES.configure_style()


def interval_frame() -> pd.DataFrame:
    rows = []
    for split, hp in FIGURES.STRATUM_ORDER:
        for model_index, model in enumerate(FIGURES.MODEL_ORDER):
            for horizon_index, horizon in enumerate(FIGURES.HORIZONS):
                mae = 0.08 + 0.025 * model_index + 0.02 * horizon_index
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "HP_Type": hp,
                        "horizon": horizon,
                        "mae": mae,
                        "mae_ci_low": mae - 0.01,
                        "mae_ci_high": mae + 0.01,
                        "picp": 0.76 + 0.01 * horizon_index,
                        "n_households": 4 if (split, hp) == ("unseen_test", "GSHP") else 12,
                        "n_rows": 48,
                    }
                )
    return pd.DataFrame(rows)


def paired_frame(candidate: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "confirmatory_holm_primary": [True, True],
            "confirmatory_holm_nested": [True, False],
            "horizon": [1, 24],
            "comparator": ["persistence", "seasonal_daily"],
            "ci_low_primary": [-0.08, -0.04],
            "ci_high_primary": [-0.01, 0.01],
            "mae_delta_primary": [-0.04, -0.02],
            "candidate": candidate,
        }
    )


class FigureQualityTests(unittest.TestCase):
    def assert_panel_header_above_axis(self, axis, prefix: str) -> None:
        matches = [
            text
            for text in axis.texts
            if text.get_text().startswith(prefix)
        ]
        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0].get_position()[1], 1.0)
        self.assertFalse(matches[0].get_clip_on())

    def test_no_axes_titles_and_fixed_multihorizon_scale(self) -> None:
        intervals = interval_frame()
        figures = [
            FIGURES.figure_2(intervals),
            FIGURES.figure_3(intervals),
            FIGURES.figure_4(paired_frame("lightgbm"), paired_frame("tft")),
        ]
        try:
            for figure in figures:
                self.assertTrue(all(axis.get_title() == "" for axis in figure.axes))
            for axis in figures[0].axes:
                self.assertEqual(tuple(np.round(axis.get_ylim(), 6)), (0.0, 0.7))
            for axis in figures[2].axes[:2]:
                left, right = axis.get_xlim()
                self.assertLess(left, 0.0)
                self.assertGreater(right, 0.0)
            for index, axis in enumerate(figures[1].axes[:4]):
                self.assert_panel_header_above_axis(axis, f"{chr(65 + index)}.")
            for index, axis in enumerate(figures[2].axes[:2]):
                self.assert_panel_header_above_axis(axis, f"{chr(65 + index)}.")
            self.assertLessEqual(float(figures[1].get_size_inches()[1]), 5.5)
            self.assertLessEqual(float(figures[2].get_size_inches()[1]), 4.7)
        finally:
            for figure in figures:
                FIGURES.plt.close(figure)

    def test_portfolio_figure_has_uncertainty_and_no_titles(self) -> None:
        summary_rows = []
        detail_rows = []
        for portfolio_type in ["ASHP", "GSHP", "mixed"]:
            for strategy_index, strategy in enumerate(
                ["median", "conformal_upper_bound"]
            ):
                summary_rows.append(
                    {
                        "portfolio_type": portfolio_type,
                        "portfolio_size": 4,
                        "model": "lightgbm",
                        "strategy": strategy,
                        "flexible_fraction": 0.20,
                        "shift_hours": 3,
                        "forecast_origins": 20,
                        "mean_peak_reduction_fraction": 0.01 + 0.005 * strategy_index,
                        "mean_load_factor_change": 0.01,
                        "false_interventions": 2 + strategy_index,
                        "missed_events": 1,
                        "interventions": 5,
                    }
                )
                for portfolio_id in ["p0", "p1"]:
                    for origin in range(4):
                        detail_rows.append(
                            {
                                "portfolio_id": f"{portfolio_type}_{portfolio_id}",
                                "portfolio_type": portfolio_type,
                                "portfolio_size": 4,
                                "model": "lightgbm",
                                "strategy": strategy,
                                "flexible_fraction": 0.20,
                                "shift_hours": 3,
                                "peak_reduction_fraction": 0.008 + 0.002 * origin,
                                "false_intervention": bool(origin == strategy_index),
                            }
                        )
        figure = FIGURES.figure_6(
            pd.DataFrame(summary_rows),
            pd.DataFrame(detail_rows),
            {("ASHP", 4), ("GSHP", 4), ("mixed", 4)},
        )
        try:
            self.assertTrue(all(axis.get_title() == "" for axis in figure.axes))
            self.assertEqual(len(figure.axes), 3)
            for index, axis in enumerate(figure.axes):
                self.assert_panel_header_above_axis(axis, f"{chr(65 + index)}.")
            self.assertLessEqual(float(figure.get_size_inches()[1]), 3.7)
        finally:
            FIGURES.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
