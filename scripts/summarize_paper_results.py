import argparse
import csv
import math
from pathlib import Path


METRIC_FIELDS = (
    "selected_test_mae",
    "mae_raw",
    "mae_flip",
    "mae_multi",
)


def _candidate_key(row):
    source = row.get("source", "").strip()
    backbone = row.get("backbone", "").strip()
    return f"{source}/{backbone}" if source else backbone


def _row_key(row):
    candidate = _candidate_key(row)
    ablation_id = row.get("ablation_id", "").strip()
    return f"{ablation_id}:{candidate}" if ablation_id else candidate


def _format_metric(values):
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def _format_std(values):
    if len(values) < 2:
        return ""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return f"{math.sqrt(variance):.4f}"


def _read_ready_rows(audit_paths):
    ready = {}
    for audit_path in audit_paths:
        with Path(audit_path).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "paper-ready":
                    continue
                try:
                    seed = int(row.get("seed", ""))
                except ValueError:
                    continue
                ready[(_row_key(row), seed)] = row
    return ready


def build_summary(audit_paths, candidates, seeds):
    ready_rows = _read_ready_rows(audit_paths)
    summary = []
    for candidate in candidates:
        ablation_id = ""
        candidate_name = candidate
        if ":" in candidate:
            ablation_id, candidate_name = candidate.split(":", 1)
        row_metrics = {field: [] for field in METRIC_FIELDS}
        ready_seeds = []
        missing_seeds = []
        for seed in seeds:
            row = ready_rows.get((candidate, seed))
            if row is None:
                missing_seeds.append(seed)
                continue
            ready_seeds.append(seed)
            for field in METRIC_FIELDS:
                try:
                    row_metrics[field].append(float(row.get(field, "")))
                except ValueError:
                    pass

        if len(ready_seeds) == len(seeds):
            status = "complete"
        elif ready_seeds:
            status = "partial"
        else:
            status = "missing"

        selected_values = row_metrics["selected_test_mae"]
        summary.append({
            "ablation_id": ablation_id,
            "candidate": candidate,
            "candidate_name": candidate_name,
            "status": status,
            "ready_seeds": ",".join(str(seed) for seed in ready_seeds),
            "missing_seeds": ",".join(str(seed) for seed in missing_seeds),
            "mean_selected_test_mae": _format_metric(selected_values),
            "std_selected_test_mae": _format_std(selected_values),
            "mean_mae_raw": _format_metric(row_metrics["mae_raw"]),
            "mean_mae_flip": _format_metric(row_metrics["mae_flip"]),
            "mean_mae_multi": _format_metric(row_metrics["mae_multi"]),
        })
    return summary


def write_markdown(rows, output_path, seeds):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    planned = ",".join(str(seed) for seed in seeds)
    lines = [
        "# FADE-Net Paper Result Summary",
        "",
        f"Planned seeds: `{planned}`",
        "",
        "Rows with `partial` or `missing` status are not final paper mean/std results.",
        "",
        "| Ablation | Candidate | Status | Ready seeds | Missing seeds | Mean Selected_Test_MAE | Std | Mean raw | Mean flip | Mean multi |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {ablation_id} | {candidate_name} | {status} | {ready_seeds} | {missing_seeds} | "
            "{mean_selected_test_mae} | {std_selected_test_mae} | "
            "{mean_mae_raw} | {mean_mae_flip} | {mean_mae_multi} |".format(**row)
        )
    lines.append("")
    lines.append("Only `complete` rows with all planned seeds should be used as final paper mean/std evidence.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value):
    return [int(item) for item in parse_list(value)]


def main():
    parser = argparse.ArgumentParser(description="Summarize paper-ready FADE-Net audit rows")
    parser.add_argument("--audit", action="append", required=True, help="Audit CSV path; may be repeated")
    parser.add_argument("--candidates", required=True, help="Comma-separated source/backbone names")
    parser.add_argument("--seeds", default="42,3407,2026", help="Comma-separated planned seeds")
    parser.add_argument("--output", default="docs/paper_result_summary.md")
    args = parser.parse_args()

    rows = build_summary(
        [Path(path) for path in args.audit],
        candidates=parse_list(args.candidates),
        seeds=parse_seeds(args.seeds),
    )
    write_markdown(rows, Path(args.output), seeds=parse_seeds(args.seeds))
    print(f"Summary written: {args.output}")


if __name__ == "__main__":
    main()
