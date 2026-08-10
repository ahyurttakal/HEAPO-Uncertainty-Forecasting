from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import temperature_feature_name
from .models import LightGBMBundle
from .utils import frame_write


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
        importance = pd.DataFrame(
            {
                "feature": x.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "HP_Type": hp_type,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance_parts.append(importance)
        shap.summary_plot(values, x, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{hp_type}.png", dpi=180, bbox_inches="tight")
        plt.close()
        temp = temperature_feature_name(x.columns)
        if temp is not None:
            shap.dependence_plot(temp, values, x, interaction_index="auto", show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_temperature_{hp_type}.png", dpi=180, bbox_inches="tight")
            plt.close()
    result = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    frame_write(result, output_dir / "shap_importance.parquet")
    return result
