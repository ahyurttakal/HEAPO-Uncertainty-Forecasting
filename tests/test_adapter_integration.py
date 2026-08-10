from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from heapo_forecasting.audit import audit_household
from heapo_forecasting.features import make_supervised_rows, prepare_household_panel, valid_origin_positions
from heapo_forecasting.io import HEAPOAdapter


class AdapterIntegrationTest(unittest.TestCase):
    def test_audit_panel_and_supervised_rows(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for directory in [
                "meta_data", "reports", "smart_meter_data/15min",
                "smart_meter_data/overview", "weather_data/hourly",
            ]:
                (root / directory).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"Household_ID": [1], "Weather_ID": ["W1"], "Group": ["control"]}
            ).to_csv(root / "meta_data/households.csv", sep=";", index=False)
            pd.DataFrame(
                {
                    "Household_ID": [1],
                    "Survey_HeatPump_Installation_Type": ["air-source"],
                    "Survey_Building_LivingArea": [140.0],
                }
            ).to_csv(root / "meta_data/meta_data.csv", sep=";", index=False)
            pd.DataFrame({"Household_ID": [1]}).to_csv(
                root / "smart_meter_data/overview/smart_meter_data_15min_overview.csv",
                sep=";", index=False,
            )
            pd.DataFrame(
                {"Report_ID": pd.Series(dtype=str), "Household_ID": pd.Series(dtype=int), "Visit_Date": pd.Series(dtype=str)}
            ).to_csv(root / "reports/protocols.csv", sep=";", index=False)
            times = pd.date_range("2020-01-01", periods=900, freq="15min", tz="UTC")
            pd.DataFrame(
                {
                    "Timestamp": times,
                    "Household_ID": 1,
                    "kWh_received_HeatPump": 1 + 0.2 * np.sin(np.arange(len(times)) / 96 * 2 * np.pi),
                    "kWh_received_Total": 2.0,
                }
            ).to_csv(root / "smart_meter_data/15min/1.csv", sep=";", index=False)
            hourly = times[::4]
            pd.DataFrame(
                {"Timestamp": hourly, "Weather_ID": "W1", "Temperature_mean_hourly": 5.0}
            ).to_csv(root / "weather_data/hourly/W1.csv", sep=";", index=False)

            cfg = {
                "data": {
                    "root": str(root), "target": "kWh_received_HeatPump",
                    "timestamp": "Timestamp", "household_id": "Household_ID",
                    "timezone": "Europe/Zurich", "short_gap_steps": 1,
                    "hp_type_column": "Survey_HeatPump_Installation_Type",
                    "allowed_hp_types": {"air-source": "ASHP", "ground-source": "GSHP"},
                    "weather_scenario": "operational",
                },
                "inclusion": {
                    "min_days": 1, "max_target_missing_fraction": 0.01,
                    "min_heating_seasons": 0, "heating_months": [1, 2, 3],
                    "min_weather_match_fraction": 0.95,
                },
                "features": {
                    "lag_steps": [1, 4, 96, 672], "rolling_steps": [4, 96],
                    "history_steps": 672, "forecast_steps": 96,
                    "on_threshold_kwh": 0.05,
                },
            }
            adapter = HEAPOAdapter(root, cfg)
            audit = audit_household(adapter, 1)
            self.assertTrue(audit.eligible, audit.exclusion_reasons)
            panel, metadata = prepare_household_panel(adapter, 1)
            self.assertEqual(metadata["hp_type"], "ASHP")
            positions = valid_origin_positions(panel, cfg, stride=96)
            rows = make_supervised_rows(panel, positions, [1, 96], cfg)
            self.assertEqual(set(rows["horizon"]), {1, 96})
            self.assertIn("future_weather__Temperature_mean_hourly", rows)


if __name__ == "__main__":
    unittest.main()
