from pathlib import Path
import concurrent.futures
import csv
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime

from tqdm import tqdm

from utils import load_task_set_list, load_yaml
import analysis


CONFIG_PATH = "simulation.yaml"
GEN_CONFIG_PATH = "generate_task_set.yaml"

DEFAULT_METHOD_NAMES = (
    "RTA_SS_single",
    "RTA_SS_max",
    "RTA_SS_tol",
    "RTA_SS_tol_fb",
    "RTA_SS_tol_fb_rbest",
    "RTA_SS_tol_fb_early",
    "RTA_UNI_tol_fb",
    "RTA_UNI_opt",
    "RTA_UNI_heu",
    "RTA_SS_heu",
    "RTA_SS_opt",
)

METHOD_ENABLE_CONFIG_KEYS = {
    "RTA_SS_single": "enable_RTA_SS_single",
    "RTA_SS_max": "enable_RTA_SS_max",
    "RTA_SS_tol": "enable_RTA_SS_tol",
    "RTA_SS_tol_fb": "enable_RTA_SS_tol_fb",
    "RTA_SS_tol_fb_rbest": "enable_RTA_SS_tol_fb_rbest",
    "RTA_SS_tol_fb_early": "enable_RTA_SS_tol_fb_early",
    "RTA_UNI_tol_fb": "enable_RTA_UNI_tol_fb",
    "RTA_UNI_opt": "enable_RTA_UNI_opt",
    "RTA_UNI_heu": "enable_RTA_UNI_heu",
    "RTA_SS_heu": "enable_RTA_SS_heu",
    "RTA_SS_opt": "enable_RTA_SS_opt",
}

PROFILING_COUNT_EXCLUDED_METHODS = {
    "RTA_SS_single",
    "RTA_SS_max",
}


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_util_from_name(task_set_list_path):
    name = Path(task_set_list_path).name
    prefix = "task_set_list_u"
    suffix = ".pkl"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return _to_float_or_none(name[len(prefix):-len(suffix)])


def _util_to_key(util):
    if util is None:
        return None
    return int(round(float(util) * 1_000_000_000))


def _parse_skip_utilizations(value):
    if value in (None, ""):
        return set()
    if isinstance(value, (int, float, str)):
        value = [value]
    return {_util_to_key(float(item)) for item in value}


def _compute_actual_utilization_from_taskset(task_set):
    cpus = task_set.get("cpus", {}) if isinstance(task_set, dict) and "cpus" in task_set else task_set
    if not isinstance(cpus, dict) or not cpus:
        return None

    per_cpu_utils = []
    for _, cpu_tasks in cpus.items():
        cpu_util = 0.0
        for task in cpu_tasks:
            if getattr(task, "T", 0):
                cpu_util += (task.C + task.G) / task.T
        per_cpu_utils.append(cpu_util)

    if not per_cpu_utils:
        return None
    return sum(per_cpu_utils) / len(per_cpu_utils)


def _compute_actual_utilization_from_taskset_list(task_set_list):
    values = [
        util
        for util in (_compute_actual_utilization_from_taskset(task_set) for task_set in task_set_list)
        if util is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _get_taskset_cpu_count(task_set):
    cpus = task_set.get("cpus", {}) if isinstance(task_set, dict) and "cpus" in task_set else task_set
    if isinstance(cpus, dict):
        return len(cpus)
    return None


def _get_taskset_list_cpu_count(task_set_list):
    counts = {
        count
        for count in (_get_taskset_cpu_count(task_set) for task_set in task_set_list)
        if count is not None
    }
    if len(counts) == 1:
        return next(iter(counts))
    return None


def _build_result_run_name(result_dir_prefix=""):
    now = datetime.now()
    date_part = now.strftime("%y%m%d")
    time_part = now.strftime("%H%M")
    if result_dir_prefix:
        return f"{date_part}-{time_part}_{result_dir_prefix}"
    return f"{date_part}-{time_part}"


def _make_unique_run_log_dir(root_dir, run_name):
    root_dir = Path(root_dir)
    run_log_dir = root_dir / run_name
    if not run_log_dir.exists():
        return run_log_dir

    counter = 1
    while True:
        candidate = root_dir / f"{run_name}_{counter:02d}"
        if not candidate.exists():
            return candidate
        counter += 1


def _enabled_methods(config):
    method_names = []
    skipped = []

    for method_name in DEFAULT_METHOD_NAMES:
        config_key = METHOD_ENABLE_CONFIG_KEYS[method_name]
        if not _to_bool(config.get(config_key, False), default=False):
            continue

        fn = getattr(analysis, method_name, None)
        if fn is None or getattr(fn, "__code__", None) is None or fn.__code__.co_argcount < 1:
            skipped.append(method_name)
            continue

        method_names.append(method_name)

    return method_names, skipped


def _normalize_result(raw_result):
    if isinstance(raw_result, tuple):
        schedulable = bool(raw_result[0]) if len(raw_result) >= 1 else False
        profiling_count = int(raw_result[1]) if len(raw_result) >= 2 and raw_result[1] is not None else 0
        return schedulable, profiling_count
    return bool(raw_result), 0


def _run_method(method_name, task_set):
    fn = getattr(analysis, method_name)
    return _normalize_result(fn(task_set))


def _worker(args):
    """Module-level worker: run one method on one task_set. Top-level for pickling."""
    task_set, method_name = args
    task_copy = deepcopy(task_set)
    try:
        schedulable, profiling_count = _run_method(method_name, task_copy)
        return schedulable, profiling_count, False
    except analysis.NumeratorExplosionError:
        return False, 0, True


def analyze_task_set_list(task_set_list_path, method_names, num_workers=1):
    data, task_set_list = load_task_set_list(str(task_set_list_path))
    total = len(task_set_list)
    file_name = Path(task_set_list_path).name

    schedulable_counts = {m: 0 for m in method_names}
    profiling_count_samples = {
        m: {"all": [], "sched": [], "unsched": []} for m in method_names
    }

    chunksize = max(1, total // (num_workers * 4)) if num_workers > 1 else 1

    executor = (
        concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
        if num_workers > 1 else None
    )
    try:
        for method_name in method_names:
            jobs = [(ts, method_name) for ts in task_set_list]
            it = (
                executor.map(_worker, jobs, chunksize=chunksize)
                if executor else (_worker(j) for j in jobs)
            )
            with tqdm(total=total, desc=f"{file_name} | {method_name}", leave=True) as pbar:
                for schedulable, profiling_count, warn in it:
                    if warn:
                        tqdm.write(
                            f"[warn] NumeratorExplosionError in {method_name} "
                            f"for {file_name}: treated as unschedulable."
                        )
                    if schedulable:
                        schedulable_counts[method_name] += 1
                        profiling_count_samples[method_name]["sched"].append(profiling_count)
                    else:
                        profiling_count_samples[method_name]["unsched"].append(profiling_count)
                    profiling_count_samples[method_name]["all"].append(profiling_count)
                    pbar.update(1)
    finally:
        if executor:
            executor.shutdown(wait=True)

    ratios = {
        method_name: (schedulable_counts[method_name] / total if total else 0.0)
        for method_name in method_names
    }

    selected_util = _to_float_or_none(data.get("selected_utilization"))
    if selected_util is None:
        selected_util = _extract_util_from_name(task_set_list_path)

    selected_util_kind = str(data.get("selected_utilization_kind", "per_cpu")).strip().lower()
    number_of_cpu = _get_taskset_list_cpu_count(task_set_list)
    if selected_util_kind == "total":
        total_util = selected_util
    else:
        total_util = selected_util * number_of_cpu if selected_util is not None and number_of_cpu is not None else selected_util

    return {
        "util": total_util,
        "ratios": ratios,
        "total": total,
        "actual_util": _compute_actual_utilization_from_taskset_list(task_set_list),
        "number_of_cpu": number_of_cpu,
        "profiling_count_samples": profiling_count_samples,
    }


def _write_plot_input_logs(run_log_dir, method_to_utilization_ratio, method_to_util_profiling_counts, number_of_cpu):
    plot_input_dir = run_log_dir / "plot_inputs"
    plot_input_dir.mkdir(parents=True, exist_ok=True)

    sched_csv = plot_input_dir / "schedulability_ratio.csv"
    with sched_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method_name", "utilization", "schedulability_ratio"])
        writer.writeheader()
        for method_name, util_map in method_to_utilization_ratio.items():
            for util, ratio in sorted(util_map.items()):
                writer.writerow(
                    {
                        "method_name": method_name,
                        "utilization": util,
                        "schedulability_ratio": ratio,
                    }
                )

    profiling_csv = plot_input_dir / "profiling_count_samples.csv"
    with profiling_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method_name", "utilization", "category", "profiling_count"],
        )
        writer.writeheader()
        for method_name, util_map in method_to_util_profiling_counts.items():
            for util, category_map in sorted(util_map.items()):
                for category in ("all", "sched", "unsched"):
                    for count in category_map.get(category, []):
                        writer.writerow(
                            {
                                "method_name": method_name,
                                "utilization": util,
                                "category": category,
                                "profiling_count": count,
                            }
                        )

    metadata_csv = plot_input_dir / "plot_metadata.csv"
    with metadata_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "value"])
        writer.writeheader()
        writer.writerow({"key": "number_of_cpu", "value": "" if number_of_cpu is None else number_of_cpu})

    readme_path = plot_input_dir / "README.txt"
    with readme_path.open("w") as f:
        f.write("Generated by simulation.py\n")
        f.write("Contains schedulability and profiling-count CSV inputs.\n")


def _write_ratio_summary(run_log_dir, method_to_utilization_ratio, runtime_info):
    summary_path = run_log_dir / "schedulability_ratio_summary.txt"
    with summary_path.open("w") as f:
        f.write("Schedulability Ratio Summary\n")
        for key, value in runtime_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        for method_name, util_map in method_to_utilization_ratio.items():
            f.write(f"[{method_name}]\n")
            for util, ratio in sorted(util_map.items()):
                f.write(f"  utilization={util:g} ratio={ratio:.6f}\n")
            f.write("\n")


def plot_schedulability_ratio(
    method_to_utilization_ratio,
    output_path,
    show_plot=False,
    number_of_cpu=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for marker_index, method_name in enumerate(method_to_utilization_ratio.keys()):
        utilization_to_ratio = method_to_utilization_ratio.get(method_name, {})
        xs = sorted(utilization_to_ratio.keys())
        if not xs:
            continue
        ys = [utilization_to_ratio[x] for x in xs]
        marker = markers[marker_index % len(markers)]
        plt.plot(xs, ys, marker=marker, label=method_name)

    plt.xlabel("Total U across CPUs")
    plt.ylabel("Schedulability Ratio")
    cpu_text = f"CPU={number_of_cpu}" if number_of_cpu is not None else "CPU=unknown"
    plt.title(f"Schedulability Ratio by Total U ({cpu_text})")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()
    print(f"Saved plot: {output_path}")


def plot_profiling_count_boxplots(
    method_to_util_profiling_counts,
    output_path,
    show_plot=False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    method_names = [
        method_name
        for method_name in method_to_util_profiling_counts.keys()
        if method_name not in PROFILING_COUNT_EXCLUDED_METHODS
    ]
    if not method_names:
        print("No profiling-count data to plot.")
        return []

    groups = [("all", "All"), ("sched", "Sched"), ("unsched", "Unsched")]
    saved_paths = []

    for group_key, group_title in groups:
        fig, axes = plt.subplots(
            1,
            len(method_names),
            figsize=(3.0 * len(method_names), 5.8),
            squeeze=False,
        )
        axes = axes[0]

        for col_idx, method_name in enumerate(method_names):
            ax = axes[col_idx]
            util_to_counts = method_to_util_profiling_counts.get(method_name, {})
            utils = sorted(util_to_counts.keys())
            data = [
                util_to_counts.get(util, {}).get(group_key, [])
                for util in utils
            ]
            tick_labels = [f"{util:.2f}" for util in utils]

            y_max = 500
            has_data = bool(tick_labels) and any(len(values) > 0 for values in data)
            if has_data:
                ax.boxplot(data, tick_labels=tick_labels, showfliers=False)
                ax.set_ylim(0, y_max)

                for pos, values in enumerate(data, start=1):
                    if not values:
                        continue
                    values_np = np.asarray(values, dtype=float)
                    median_value = float(np.median(values_np))
                    max_value = float(np.max(values_np))
                    ax.text(
                        pos,
                        max(median_value - 17.5, 10),
                        f"{median_value:.2f}".rstrip("0").rstrip("."),
                        fontsize=12,
                        color="#1f77b4",
                        ha="center",
                        va="top",
                    )
                    ax.text(
                        pos,
                        y_max * 0.95,
                        f"{max_value:.2f}".rstrip("0").rstrip("."),
                        fontsize=12,
                        color="#d62728",
                        ha="center",
                        va="bottom",
                        clip_on=False,
                    )
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_ylim(0, y_max)

            ax.set_title(method_name, fontsize=12)
            ax.set_xlabel("Total U across CPUs", fontsize=10)
            ax.set_ylabel("Profiling Count" if col_idx == 0 else "", fontsize=10)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.tick_params(axis="x", rotation=45, labelsize=9)

        fig.suptitle(f"Profiling Count Boxplot ({group_title})")
        fig.subplots_adjust(left=0.05, right=0.995, bottom=0.18, top=0.86, wspace=0.20)

        out = output_path.with_name(f"{output_path.stem}_{group_key}{output_path.suffix}")
        fig.savefig(out, dpi=200)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        saved_paths.append(out)
        print(f"Saved profiling count boxplot: {out}")

    return saved_paths


def _write_runtime_log(run_log_dir, started_at, started_ts):
    finished_at = datetime.now()
    elapsed_seconds = max(time.time() - started_ts, 0.0)
    runtime_path = run_log_dir / "simulation_runtime.txt"
    with runtime_path.open("w") as f:
        f.write("Simulation Runtime\n")
        f.write("status: completed\n")
        f.write(f"started_at: {started_at.isoformat(timespec='seconds')}\n")
        f.write(f"finished_at: {finished_at.isoformat(timespec='seconds')}\n")
        f.write(f"elapsed_seconds: {elapsed_seconds:.3f}\n")
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_seconds": f"{elapsed_seconds:.3f}",
    }


def main():
    started_at = datetime.now()
    started_ts = time.time()

    config = load_yaml(CONFIG_PATH)
    task_set_list_dir_path = config.get("task_set_list_dir_path")
    plot_output_path = config.get(
        "plot_output_path",
        "schedulability_ratio_by_utilization.png",
    )
    profiling_count_boxplot_output_path = config.get(
        "profiling_count_boxplot_output_path",
        "profiling_count_boxplot.png",
    )
    show_plot = _to_bool(config.get("show_plot", False), default=False)
    if not task_set_list_dir_path:
        raise ValueError("Missing task_set_list_dir_path in simulation.yaml")

    task_set_source_dir = Path(task_set_list_dir_path)
    if not task_set_source_dir.exists():
        raise ValueError(f"task_set_list_dir_path does not exist: {task_set_list_dir_path}")

    method_names, skipped_methods = _enabled_methods(config)
    if not method_names:
        raise ValueError("No implemented RTA methods are enabled.")

    result_dir_prefix = str(config.get("result_dir_prefix", "") or "").strip()
    run_log_dir = _make_unique_run_log_dir("result", _build_result_run_name(result_dir_prefix))
    run_log_dir.mkdir(parents=True, exist_ok=False)

    if Path(CONFIG_PATH).exists():
        shutil.copy2(CONFIG_PATH, run_log_dir / Path(CONFIG_PATH).name)
    if Path(GEN_CONFIG_PATH).exists():
        shutil.copy2(GEN_CONFIG_PATH, run_log_dir / Path(GEN_CONFIG_PATH).name)

    task_set_snapshot_dir = run_log_dir / "task_sets"
    shutil.copytree(task_set_source_dir, task_set_snapshot_dir)

    skip_util_keys = _parse_skip_utilizations(config.get("skip_utilizations", []))
    task_set_list_paths = sorted(task_set_source_dir.glob("*.pkl"))
    if not task_set_list_paths:
        raise ValueError("No .pkl files found for analysis.")

    num_workers = int(config.get("num_workers", 0)) or os.cpu_count() or 1

    method_to_utilization_ratio = {method_name: {} for method_name in method_names}
    method_to_util_profiling_counts = {method_name: {} for method_name in method_names}
    number_of_cpu_for_plot = None

    print(f"result directory: {run_log_dir}")
    print("enabled methods:", ", ".join(method_names))
    if skipped_methods:
        print("skipped unimplemented methods:", ", ".join(skipped_methods))

    pending_paths = [
        p for p in task_set_list_paths
        if _util_to_key(_extract_util_from_name(p)) not in skip_util_keys
    ]

    print(f"num_workers={num_workers}, files={len(pending_paths)}, "
          f"methods={len(method_names)}")

    for task_set_list_path in pending_paths:
        result = analyze_task_set_list(task_set_list_path, method_names, num_workers=num_workers)
        util = result["util"]
        if util is None:
            continue

        if number_of_cpu_for_plot is None and result["number_of_cpu"] is not None:
            number_of_cpu_for_plot = result["number_of_cpu"]

        for method_name in method_names:
            method_to_utilization_ratio[method_name][util] = result["ratios"].get(method_name, 0.0)
            method_to_util_profiling_counts[method_name][util] = result["profiling_count_samples"].get(
                method_name,
                {"all": [], "sched": [], "unsched": []},
            )

        ratio_text = ", ".join(
            f"{method_name}={result['ratios'].get(method_name, 0.0):.3f}"
            for method_name in method_names
        )
        print(f"utilization={util:g} tasksets={result['total']} {ratio_text}")

    _write_plot_input_logs(
        run_log_dir,
        method_to_utilization_ratio,
        method_to_util_profiling_counts,
        number_of_cpu_for_plot,
    )
    runtime_info = _write_runtime_log(run_log_dir, started_at, started_ts)
    _write_ratio_summary(run_log_dir, method_to_utilization_ratio, runtime_info)
    plot_schedulability_ratio(
        method_to_utilization_ratio,
        run_log_dir / "plot" / Path(plot_output_path).name,
        show_plot=show_plot,
        number_of_cpu=number_of_cpu_for_plot,
    )
    plot_profiling_count_boxplots(
        method_to_util_profiling_counts,
        run_log_dir / "plot" / Path(profiling_count_boxplot_output_path).name,
        show_plot=show_plot,
    )


if __name__ == "__main__":
    main()
