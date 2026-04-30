#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOT_UTIL_MIN = 0.6
PLOT_UTIL_MAX = 1.0


def _fmt_num(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _method_sort_key(name: str):
    prefix = "RTA_SS_tol_fb_lr_early_n"
    if name.startswith(prefix):
        tail = name[len(prefix):]
        try:
            return (0, int(tail), name)
        except ValueError:
            return (0, 10**9, name)
    if name == "RTA_SS_tol":
        return (1, 0, name)
    if name == "RTA_SS_tol_fb":
        return (2, 0, name)
    if name in {
        "RTA_UNI_tol_fb",
        "UNI_tol_fb",
    }:
        return (3, 0, name)
    if name == "RTA_SS_max":
        return (4, 0, name)
    if name == "RTA_UNI_opt":
        return (5, 0, name)
    if name == "RTA_UNI_heu":
        return (6, 0, name)
    return (9, 0, name)


def _method_label(name: str) -> str:
    prefix = "RTA_SS_tol_fb_lr_early_n"
    if name.startswith(prefix):
        return "RTA_SS_tol_fb_lr_early_n" + name[len(prefix):]
    if name == "RTA_UNI_opt":
        return "RTA_UNI_opt"
    if name == "RTA_UNI_heu":
        return "RTA_UNI_heu"
    if name in {
        "RTA_UNI_tol_fb",
        "UNI_tol_fb",
    }:
        return "RTA_UNI_tol_fb"
    return name


def load_unsched_block_counts(csv_path: Path):
    method_to_util = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            method = str(row.get("method_name", "")).strip()
            group = str(row.get("group", "")).strip().lower()
            if group != "unsched":
                continue
            if method in {"base", "BASE", "RTA_SS_single"}:
                continue
            try:
                util = float(row["utilization"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if util < PLOT_UTIL_MIN - 1e-9 or util > PLOT_UTIL_MAX + 1e-9:
                continue
            method_to_util.setdefault(method, {}).setdefault(util, []).append(value)
    return method_to_util


def plot_unsched_block_count(run_dir: Path, output_name: str):
    csv_path = run_dir / "plot_inputs" / "profiling_count_samples.csv"
    if not csv_path.exists():
        csv_path = run_dir / "plot_inputs" / "splitting_count_samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing csv: {csv_path}")

    method_to_util = load_unsched_block_counts(csv_path)
    out_path = run_dir / output_name
    if not method_to_util:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.text(
            0.5,
            0.5,
            "No unsched profiling count data (BASE excluded)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    triads = [
        ("RTA_SS_max", "RTA_SS_max"),
        ("RTA_SS_tol_fb", "RTA_SS_tol_fb"),
        ("RTA_UNI_tol_fb", "RTA_UNI_tol_fb"),
        ("RTA_UNI_opt", "RTA_UNI_opt"),
        ("RTA_UNI_heu", "RTA_UNI_heu"),
        ("RTA_SS_tol_fb_lr_early_n2", "RTA_SS_tol_fb_lr_early_n2"),
        ("RTA_SS_max", "RTA_SS_max"),
        ("RTA_SS_tol_fb", "RTA_SS_tol_fb"),
        ("RTA_SS_tol_fb_lr_early_n4", "RTA_SS_tol_fb_lr_early_n4"),
        ("RTA_SS_max", "RTA_SS_max"),
        ("RTA_SS_tol_fb", "RTA_SS_tol_fb"),
        ("RTA_SS_tol_fb_lr_early_n6", "RTA_SS_tol_fb_lr_early_n6"),
        ("RTA_SS_max", "RTA_SS_max"),
        ("RTA_SS_tol_fb", "RTA_SS_tol_fb"),
        ("RTA_SS_tol_fb_lr_early_n8", "RTA_SS_tol_fb_lr_early_n8"),
        ("RTA_SS_max", "RTA_SS_max"),
        ("RTA_SS_tol_fb", "RTA_SS_tol_fb"),
        ("RTA_SS_tol_fb_lr_early_n10", "RTA_SS_tol_fb_lr_early_n10"),
    ]

    plot_specs = [(label, method_name) for (label, method_name) in triads if method_name in method_to_util]

    if not plot_specs:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.text(0.5, 0.5, "No requested method data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    n_cols = len(plot_specs)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 5.8), squeeze=False)
    axes = axes[0]

    for ax, (label, method_name) in zip(axes, plot_specs):
        util_to_values = method_to_util.get(method_name, {})
        utils = sorted(util_to_values.keys())
        data = [util_to_values[u] for u in utils]
        tick_labels = [f"{u:.2f}" for u in utils]

        if not any(len(v) > 0 for v in data):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_title(label, fontsize=12)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            continue

        ax.boxplot(data, tick_labels=tick_labels, showfliers=False)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("U", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        y_low, y_high = ax.get_ylim()
        span = max(y_high - y_low, 1.0)
        ax.set_ylim(y_low, y_high + 0.20 * span)
        outside_y = y_high + 0.125 * span

        for pos, vals in enumerate(data, start=1):
            if not vals:
                continue
            vals_np = np.asarray(vals, dtype=float)
            med_v = float(np.median(vals_np))
            max_v = float(np.max(vals_np))

            mean_y = med_v - 0.035 * span
            mean_y = max(mean_y, y_low + 0.02 * span)
            ax.text(pos, mean_y, _fmt_num(med_v), fontsize=12, color="#1f77b4", ha="center", va="top")
            ax.text(
                pos,
                outside_y,
                _fmt_num(max_v),
                fontsize=12,
                color="#d62728",
                ha="center",
                va="bottom",
                clip_on=False,
            )

    axes[0].set_ylabel("Profiling Count", fontsize=10)

    fig.suptitle("Unsched Profiling Count Boxplot (BASE Excluded)")
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.20, top=0.88, wspace=0.20)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Plot unsched profiling count boxplots from result/plot_inputs/profiling_count_samples.csv (BASE excluded)."
    )
    ap.add_argument("--run-dir", required=True, help="Path to simulation run dir (e.g., simulation/result/<run_id>)")
    ap.add_argument(
        "--output-name",
        default="unsched_block_count_boxplot_no_base.png",
        help="Output filename under run-dir",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out = plot_unsched_block_count(run_dir, args.output_name)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
