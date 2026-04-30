#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


METHOD_LABELS = {
    "RTA_SS_tol_fb_rbest": "RTA_SS_tol_fb_rbest",
    "RTA_UNI_tol_fb": "RTA_UNI_tol_fb",
    "RTA_UNI_heu": "RTA_UNI_heu",
}


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p")


def load_trace_rows(run_dir):
    csv_path = Path(run_dir) / "analysis_exports" / "min_tolerance_trace.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing trace csv: {csv_path}")

    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            util = _to_float(row.get("utilization"))
            taskset_idx = _to_int(row.get("task_set_index"))
            profiling_count = _to_int(row.get("profiling_count"))
            min_tolerance = _to_float(row.get("min_tolerance"))
            if (
                util is None
                or taskset_idx is None
                or profiling_count is None
                or min_tolerance is None
                or math.isinf(min_tolerance)
                or math.isnan(min_tolerance)
            ):
                continue
            row = dict(row)
            row["utilization"] = util
            row["task_set_index"] = taskset_idx
            row["profiling_count"] = profiling_count
            row["min_tolerance"] = min_tolerance
            rows.append(row)
    return rows


def load_schedulability_map(run_dir):
    rta_log_dir = Path(run_dir) / "rta_logs"
    sched_map = {}
    if not rta_log_dir.exists():
        return sched_map

    util_re = re.compile(r"_u(\d+(?:\.\d+)?)\.log$")
    taskset_re = re.compile(r"^\[task_set_index=(\d+)\]")
    method_re = re.compile(r"^\s{2}<([^>]+)>\s+schedulable=(True|False)")

    for log_path in sorted(rta_log_dir.glob("rta_*.log")):
        m_util = util_re.search(log_path.name)
        util = _to_float(m_util.group(1)) if m_util else None
        if util is None:
            continue

        current_taskset = None
        with log_path.open("r") as f:
            for line in f:
                m_taskset = taskset_re.match(line)
                if m_taskset:
                    current_taskset = _to_int(m_taskset.group(1))
                    continue

                m_method = method_re.match(line)
                if m_method and current_taskset is not None:
                    method_name = m_method.group(1)
                    sched = m_method.group(2) == "True"
                    sched_map[(util, current_taskset, method_name)] = sched

    return sched_map


def filter_rows(rows, utilization=None, task_set_index=None, method_name=None):
    out = []
    for row in rows:
        if utilization is not None and abs(row["utilization"] - utilization) > 1e-9:
            continue
        if task_set_index is not None and row["task_set_index"] != task_set_index:
            continue
        if method_name is not None and row["method_name"] != method_name:
            continue
        out.append(row)
    return out


def collapse_to_taskset_trajectory(rows):
    by_point = {}
    for row in rows:
        key = (
            row["utilization"],
            row["task_set_index"],
            row["method_name"],
            row["profiling_count"],
        )
        by_point[key] = min(by_point.get(key, row["min_tolerance"]), row["min_tolerance"])

    by_series = {}
    for (util, taskset_idx, method_name, profiling_count), min_tolerance in by_point.items():
        key = (util, taskset_idx, method_name)
        by_series.setdefault(key, []).append((profiling_count, min_tolerance))

    for key in by_series:
        by_series[key].sort(key=lambda item: item[0])
    return by_series


def _sched_text(sched_map, util, taskset_idx, method_name):
    sched = sched_map.get((util, taskset_idx, method_name))
    if sched is None:
        return "sched=?"
    return "sched=Y" if sched else "sched=N"


def plot_series(by_series, output_dir, sched_map=None, combined=False, show=False):
    import matplotlib.pyplot as plt

    sched_map = sched_map or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if combined:
        grouped = {}
        for (util, taskset_idx, method_name), points in by_series.items():
            grouped.setdefault((util, taskset_idx), []).append((method_name, points))

        saved = []
        for (util, taskset_idx), series_list in sorted(grouped.items()):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            title_status = []
            for method_name, points in sorted(series_list):
                status = _sched_text(sched_map, util, taskset_idx, method_name)
                title_status.append(f"{METHOD_LABELS.get(method_name, method_name)}:{status}")
                ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    linewidth=1.6,
                    label=f"{METHOD_LABELS.get(method_name, method_name)} ({status})",
                )
            ax.set_title(
                f"Min Tolerance Trace (U={util:g}, taskset={taskset_idx})\n"
                + ", ".join(title_status)
            )
            ax.set_xlabel("Profiling Count")
            ax.set_ylabel("Min Tolerance")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            out_path = output_dir / f"min_tolerance_u{_safe_name(util)}_idx{taskset_idx}_combined.png"
            fig.savefig(out_path, dpi=200)
            if show:
                plt.show()
            else:
                plt.close(fig)
            saved.append(out_path)
        return saved

    saved = []
    for (util, taskset_idx, method_name), points in sorted(by_series.items()):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        status = _sched_text(sched_map, util, taskset_idx, method_name)
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            linewidth=1.6,
        )
        ax.set_title(
            f"{METHOD_LABELS.get(method_name, method_name)} Min Tolerance "
            f"(U={util:g}, taskset={taskset_idx}, {status})"
        )
        ax.set_xlabel("Profiling Count")
        ax.set_ylabel("Min Tolerance")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        out_path = (
            output_dir
            / f"min_tolerance_u{_safe_name(util)}_idx{taskset_idx}_{_safe_name(METHOD_LABELS.get(method_name, method_name))}.png"
        )
        fig.savefig(out_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)
        saved.append(out_path)
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-taskset min tolerance traces from analysis_exports/min_tolerance_trace.csv."
    )
    parser.add_argument("--run-dir", required=True, help="Simulation result directory.")
    parser.add_argument("--utilization", type=float, default=None, help="Filter by utilization, e.g. 0.7.")
    parser.add_argument("--task-set-index", type=int, default=None, help="Filter by task set index.")
    parser.add_argument("--method-name", default=None, help="Filter by exact method name.")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Plot RBest and Aromolo heuristic on the same figure for each taskset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <run-dir>/analysis_exports/min_tolerance_plots.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "analysis_exports" / "min_tolerance_plots"
    )

    rows = load_trace_rows(run_dir)
    sched_map = load_schedulability_map(run_dir)
    rows = filter_rows(
        rows,
        utilization=args.utilization,
        task_set_index=args.task_set_index,
        method_name=args.method_name,
    )
    by_series = collapse_to_taskset_trajectory(rows)
    if not by_series:
        raise SystemExit("No matching min tolerance trace rows.")

    saved = plot_series(
        by_series,
        output_dir,
        sched_map=sched_map,
        combined=args.combined,
        show=args.show,
    )
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
