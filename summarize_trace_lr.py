#!/usr/bin/env python3
"""Summarize positive linear-regression slopes from trace CSV files.

Given a trace root directory, this script reads per-taskset CSV files and computes,
for each task in each taskset, the slope of linear regression using the first N points
(sorted by profiling_count). It then reports how many tasks have positive slope.

Default behavior focuses on unschedulable tasksets and R metric.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _extract_taskset_idx(path: Path) -> int | None:
    m = re.search(r"_idx(\d+)_", path.name)
    if not m:
        return None
    return int(m.group(1))


def _linear_regression_slope(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None

    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        dx = x - x_mean
        num += dx * (y - y_mean)
        den += dx * dx

    if den == 0.0:
        return None
    return num / den


def _read_rows(path: Path, metric: str, phase: str) -> Dict[int, List[Tuple[float, float]]]:
    by_task: Dict[int, List[Tuple[float, float]]] = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if phase != "both" and row.get("row_phase") != phase:
                continue

            task_raw = row.get("task_index")
            x_raw = row.get("profiling_count", row.get("split_count"))
            y_raw = row.get(metric)
            if task_raw in (None, "") or x_raw in (None, "") or y_raw in (None, "", "inf", "-inf"):
                continue

            try:
                task_idx = int(task_raw)
                x = float(x_raw)
                y = float(y_raw)
            except ValueError:
                continue

            by_task.setdefault(task_idx, []).append((x, y))

    for task_idx in by_task:
        by_task[task_idx].sort(key=lambda p: p[0])

    return by_task


def _iter_target_csvs(root: Path, subset: str):
    subsets = []
    if subset == "all":
        subsets = ["unschedulable", "schedulable"]
    elif subset == "unsched":
        subsets = ["unschedulable"]
    else:
        subsets = ["schedulable"]

    for sub in subsets:
        folder = root / sub
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*_splitting_trace.csv")):
            yield sub, p


def main():
    parser = argparse.ArgumentParser(description="Summarize positive LR slopes per taskset from trace CSVs")
    parser.add_argument("--trace-root", required=True, help="Trace root directory containing schedulable/unschedulable")
    parser.add_argument("--n", type=int, required=True, help="Use first n points per task for regression")
    parser.add_argument("--metric", default="R", choices=["R", "B_low", "B_high", "tolerance_i"], help="Metric column")
    parser.add_argument("--subset", default="unsched", choices=["unsched", "sched", "all"], help="Target subset")
    parser.add_argument("--phase", default="both", choices=["before", "after", "both"], help="row_phase filter (optional; default: both)")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    if args.n < 2:
        raise ValueError("--n must be >= 2")

    trace_root = Path(args.trace_root).expanduser().resolve()
    if not trace_root.exists():
        raise FileNotFoundError(f"trace root not found: {trace_root}")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        phase_suffix = "" if args.phase == "both" else f"_{args.phase}"
        out_name = f"lr_positive_summary_{args.subset}_{args.metric}_n{args.n}{phase_suffix}.txt"
        out_path = trace_root / out_name

    lines: List[str] = []
    lines.append(f"trace_root={trace_root}")
    lines.append(f"subset={args.subset} metric={args.metric} phase={args.phase} n={args.n}")
    lines.append("")

    total_tasksets = 0
    total_tasks = 0
    total_positive = 0

    for sub, csv_path in _iter_target_csvs(trace_root, args.subset):
        taskset_idx = _extract_taskset_idx(csv_path)
        by_task = _read_rows(csv_path, args.metric, args.phase)

        slopes: Dict[int, float] = {}
        for task_idx, points in by_task.items():
            trimmed = points[: args.n]
            if len(trimmed) < 2:
                continue
            xs = [p[0] for p in trimmed]
            ys = [p[1] for p in trimmed]
            slope = _linear_regression_slope(xs, ys)
            if slope is None:
                continue
            slopes[task_idx] = slope

        positive_tasks = sorted([t for t, s in slopes.items() if s > 0.0])
        considered_tasks = sorted(slopes.keys())

        total_tasksets += 1
        total_tasks += len(considered_tasks)
        total_positive += len(positive_tasks)

        lines.append("------------------------------------------------------------")
        lines.append(
            f"task_set_index={taskset_idx} subset={sub} file={csv_path.name}"
        )
        lines.append(
            f"positive_slope_tasks={len(positive_tasks)}/{len(considered_tasks)} "
            f"ratio={(len(positive_tasks)/len(considered_tasks) if considered_tasks else 0.0):.4f}"
        )

        if positive_tasks:
            lines.append("positive_task_indices=" + ",".join(str(x) for x in positive_tasks))
        else:
            lines.append("positive_task_indices=(none)")

        slope_items = ", ".join(f"{t}:{slopes[t]:.6f}" for t in considered_tasks)
        lines.append("task_slopes=" + (slope_items if slope_items else "(none)"))
        lines.append("")

    lines.append("============================================================")
    lines.append(f"total_tasksets={total_tasksets}")
    lines.append(f"total_considered_tasks={total_tasks}")
    lines.append(f"total_positive_slope_tasks={total_positive}")
    lines.append(
        f"overall_positive_ratio={(total_positive/total_tasks if total_tasks else 0.0):.6f}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
