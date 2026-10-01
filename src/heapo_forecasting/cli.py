from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, validate_config
from .experiment import initialize_run, prepare_stage, run_full, run_pilot, run_scenario_suite
from .audit import run_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HEAPO forecasting study pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["audit", "prepare", "pilot", "full", "tft", "suite", "finalize"]:
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    return parser


def _print_result(result: dict[str, Any]) -> None:
    safe = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
        if key != "metrics"
    }
    print(json.dumps(safe, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    validate_config(cfg, require_data=True)
    if args.command == "audit":
        root, dirs = initialize_run(cfg)
        audit = run_audit(cfg, dirs["audit"])
        _print_result(
            {
                "run_dir": root,
                "households": len(audit),
                "eligible": int(audit["eligible"].sum()),
                "eligible_ASHP": int(((audit["eligible"]) & (audit["hp_type"] == "ASHP")).sum()),
                "eligible_GSHP": int(((audit["eligible"]) & (audit["hp_type"] == "GSHP")).sum()),
            }
        )
    elif args.command == "prepare":
        root, _, audit, panels, splits, supervised = prepare_stage(cfg)
        _print_result(
            {
                "run_dir": root,
                "eligible_households": int(audit["eligible"].sum()),
                "panels": len(panels),
                "supervised_shards": len(supervised),
                "training_households": len(splits["training_households"]),
                "unseen_test_households": len(splits["unseen_test_households"]),
            }
        )
    elif args.command == "pilot":
        _print_result(run_pilot(cfg))
    elif args.command == "full":
        _print_result(run_full(cfg))
    elif args.command == "suite":
        _print_result(run_scenario_suite(cfg))
    elif args.command == "finalize":
        from .finalize import run_statistical_finalize

        _print_result(run_statistical_finalize(cfg))
    elif args.command == "tft":
        from .tft_model import build_common_evaluation_index, train_tft
        root, dirs, _, panels, splits, _ = prepare_stage(cfg)
        prediction_path = dirs["predictions"] / "household_predictions.parquet"
        evaluation_index = None
        if prediction_path.exists() or prediction_path.with_suffix(".csv.gz").exists():
            from .utils import frame_read, frame_write

            predictions = frame_read(prediction_path)
            evaluation_index = build_common_evaluation_index(predictions, cfg)
            frame_write(evaluation_index, dirs["tft"] / "tft_common_evaluation_index.parquet")
        checkpoint = train_tft(
            panels, splits, cfg, dirs["tft"], evaluation_index=evaluation_index
        )
        _print_result({"run_dir": root, "checkpoint": checkpoint})


if __name__ == "__main__":
    main()
