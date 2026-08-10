from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(root: Path, names: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name in names:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        result[name] = path
    return result


def atomic_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
    temp.replace(path)


def frame_write(frame: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write parquet when available and fall back to gzipped CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=index)
        return path
    except (ImportError, ModuleNotFoundError):
        csv_path = path.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=index, compression="gzip")
        return csv_path


def frame_read(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.exists():
        if path.name.endswith(".csv.gz") or path.suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        return pd.read_parquet(path, **kwargs)
    csv_path = path.with_suffix(".csv.gz")
    if csv_path.exists():
        return pd.read_csv(csv_path, **kwargs)
    raise FileNotFoundError(path)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def input_fingerprint(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            result[str(path)] = {
                "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
    return result


def environment_manifest() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ["numpy", "pandas", "sklearn", "lightgbm", "shap", "torch"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def finite_sample_quantile(values: np.ndarray, probability: float) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    level = min(1.0, np.ceil((clean.size + 1) * probability) / clean.size)
    return float(np.quantile(clean, level, method="higher"))
