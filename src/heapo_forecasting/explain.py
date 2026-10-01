from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import temperature_feature_name
from .models import LightGBMBundle
from .utils import frame_write


def readable_feature_name(name: str) -> str:
    exact = {
        "load_on_fraction_24h": "On fraction (24 h)",
        "load_roll_mean_96": "Rolling load mean (24 h)",
        "load_roll_mean_48": "Rolling load mean (12 h)",
        "load_roll_mean_24": "Rolling load mean (6 h)",
        "load_roll_mean_12": "Rolling load mean (3 h)",
        "load_roll_mean_4": "Rolling load mean (1 h)",
        "load_day_base": "Daily base load",
        "load_day_max": "Daily maximum load",
        "load_day_min": "Daily minimum load",
        "load_lag_1": "Load lag (15 min)",
        "load_lag_4": "Load lag (1 h)",
        "load_lag_24": "Load lag (6 h)",
        "future_weather__Temperature_avg_hourly": "Outdoor temperature",
        "future_weather__Temperature_mean_hourly": "Outdoor temperature",
        "future_weather__Sunshine_duration_hourly": "Sunshine duration",
        "future__cal_hour_sin": "Hour-of-day sine",
        "future__cal_hour_cos": "Hour-of-day cosine",
        "future__cal_doy_sin": "Day-of-year sine",
        "future__cal_doy_cos": "Day-of-year cosine",
    }
    if name in exact:
        return exact[name]
    cleaned = str(name)
    for prefix in (
        "future_weather__", "future__cal_", "static__Survey_", "static__", "load_"
    ):
        cleaned = cleaned.replace(prefix, "")
    return cleaned.replace("_", " ").strip().title()


def run_shap(
    bundle: LightGBMBundle,
    rows: pd.DataFrame,
    output_dir: Path,
    cfg: dict[str, Any],
    max_rows_per_type: int = 5000,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to run SHAP") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    importance_parts: list[pd.DataFrame] = []
    explainer = shap.TreeExplainer(bundle.models["mean"])
    for hp_type, group in rows.groupby("HP_Type", dropna=False):
        if len(group) > max_rows_per_type:
            group = group.iloc[rng.choice(len(group), max_rows_per_type, replace=False)]
        x = bundle.encoder.transform(group)
        values = explainer.shap_values(x)
        values = np.asarray(values)
        feature_labels = [readable_feature_name(column) for column in x.columns]
        plot_x = x.copy()
        plot_x.columns = feature_labels
        importance = pd.DataFrame(
            {
                "feature": x.columns,
                "feature_label": feature_labels,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "HP_Type": hp_type,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance_parts.append(importance)
        summary_kwargs: dict[str, Any] = {"show": False, "max_display": 25}
        if "rng" in inspect.signature(shap.summary_plot).parameters:
            summary_kwargs["rng"] = rng
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The NumPy global RNG was seeded by calling `np.random.seed`.*",
                category=FutureWarning,
            )
            shap.summary_plot(values, plot_x, **summary_kwargs)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{hp_type}.png", dpi=600, bbox_inches="tight")
        plt.close()
        temp = temperature_feature_name(x.columns)
        if temp is not None:
            # SHAP's automatic interaction search can fit a StandardScaler on
            # a DataFrame and repeatedly transform unnamed arrays, producing
            # thousands of harmless sklearn feature-name warnings. Use the
            # pre-specified operationally meaningful interaction directly.
            interaction = (
                "load_on_fraction_24h"
                if "load_on_fraction_24h" in x.columns
                else None
            )
            plot_temperature = readable_feature_name(temp)
            plot_interaction = (
                readable_feature_name(interaction) if interaction is not None else None
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "X does not have valid feature names, but StandardScaler "
                        "was fitted with feature names"
                    ),
                    category=UserWarning,
                    module=r"sklearn\.utils\.validation",
                )
                shap.dependence_plot(
                    plot_temperature,
                    values,
                    plot_x,
                    interaction_index=plot_interaction,
                    show=False,
                )
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_temperature_{hp_type}.png", dpi=600, bbox_inches="tight")
            plt.close()
    result = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    frame_write(result, output_dir / "shap_importance.parquet")
    return result
