"""Select one completed ablation by validation MAE and write an audit summary."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate_result",
        action="append",
        default=[],
        help="Additional baseline results.json to include in Val-only selection",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    queue_root = Path(args.queue_root)
    completed = []
    result_paths = list(queue_root.glob("exp*/fold0/results.json"))
    result_paths.extend(Path(path) for path in args.candidate_result)
    for result_path in sorted(set(path.resolve() for path in result_paths)):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        best_val_mae = result.get("best_val_mae")
        checkpoint = result_path.parent / "best_checkpoint.pth"
        if (
            not isinstance(best_val_mae, (int, float))
            or not math.isfinite(best_val_mae)
            or not checkpoint.is_file()
        ):
            continue
        completed.append(
            {
                "experiment": result_path.parents[1].name,
                "best_val_mae": best_val_mae,
                "best_epoch": result.get("best_epoch"),
                "completed_epochs": result.get("completed_epochs"),
                "checkpoint": str(checkpoint.resolve()),
                "results": str(result_path.resolve()),
            }
        )

    completed.sort(key=lambda item: (item["best_val_mae"], item["experiment"]))
    if not completed:
        raise RuntimeError(f"No completed ablations found under {queue_root}")

    summary = {
        "selection_metric": "best EMA validation MAE with original 1x view",
        "selection_rule": "minimum validation MAE; Test is not used for model selection",
        "completed_experiments": completed,
        "selected": completed[0],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
