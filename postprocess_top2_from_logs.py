#!/usr/bin/env python3
"""
Post-process existing simulation result logs (no re-simulation) and report top-2 methods.

Rule requested by user:
- Compare each RTA_SS_tol_fb_lr_early_n method against RTA_SS_tol_fb schedulability ratio.
- If difference <= threshold (default 0.05), treat it as equivalent to one-higher N.
- Above highest N, treat as equivalent to RTA_SS_tol_fb.

Outputs under <run_dir>/postprocess_top2/:
- method_summary.csv
- top2_by_metric.csv
- report.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

RTA_SS_tol_fb_METHOD = "RTA_SS_tol_fb"
LR_PREFIX = "RTA_SS_tol_fb_lr_early_n"


def _parse_util_from_name(path: Path):
    m = re.search(r"_u(\d+(?:\.\d+)?)\.log$", path.name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _stats(values):
    if not values:
        return None
    arr = sorted(values)
    n = len(arr)
    return {
        "count": n,
        "mean": sum(arr) / n,
        "min": arr[0],
        "q1": _percentile(arr, 0.25),
        "median": _percentile(arr, 0.50),
        "q3": _percentile(arr, 0.75),
        "max": arr[-1],
    }


def _parse_rta_sched(rta_dir: Path):
    # method -> {"sched": int, "total": int, "by_util": {u: {"sched": int, "total": int}}}
    out = {}
    for path in sorted(rta_dir.glob("rta_task_set_list_u*.log")):
        util = _parse_util_from_name(path)
        if util is None:
            continue
        with path.open("r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                m = re.match(r"^\s{2}<([^>]+)>\s+schedulable=(True|False)$", line)
                if not m:
                    continue
                method = m.group(1)
                sched = (m.group(2) == "True")
                row = out.setdefault(method, {"sched": 0, "total": 0, "by_util": {}})
                row["total"] += 1
                if sched:
                    row["sched"] += 1
                by_u = row["by_util"].setdefault(util, {"sched": 0, "total": 0})
                by_u["total"] += 1
                if sched:
                    by_u["sched"] += 1
    return out


def _parse_tolerance_profiling_counts(rta_dir: Path):
    # method -> [profiling_count, ...].
    counts = {}
    for path in sorted(rta_dir.glob("tolerance_task_set_list_u*.log")):
        current_method = None
        with path.open("r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                m_method = re.match(r"^\s{2}<([^>]+)>$", line)
                if m_method:
                    current_method = m_method.group(1)
                    continue
                m_count = re.match(r"^\s{4}(?:profiling_count|splitting_count)=(\d+)$", line)
                if m_count and current_method is not None:
                    counts.setdefault(current_method, []).append(int(m_count.group(1)))
    return counts


def _lr_n(method_name: str):
    if not method_name.startswith(LR_PREFIX):
        return None
    tail = method_name[len(LR_PREFIX):]
    try:
        return int(tail)
    except ValueError:
        return None


def _format_float(v):
    if v is None:
        return ""
    return f"{float(v):.6f}"


def main():
    ap = argparse.ArgumentParser(description="Top-2 postprocess from existing result logs.")
    ap.add_argument("--run-dir", required=True, help="result/<run_id> directory")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="sched-ratio closeness threshold vs RTA_SS_tol_fb (default: 0.05)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    rta_dir = run_dir / "rta_logs"
    if not rta_dir.exists():
        raise FileNotFoundError(f"rta_logs not found: {rta_dir}")

    sched = _parse_rta_sched(rta_dir)
    profiling_counts = _parse_tolerance_profiling_counts(rta_dir)

    if RTA_SS_tol_fb_METHOD not in sched:
        raise ValueError(f"RTA_SS_tol_fb method not found in rta logs: {RTA_SS_tol_fb_METHOD}")

    # Target methods: RTA_SS_tol_fb + RTA_SS_tol_fb_lr_early_n methods that have both sched and profiling data.
    lr_methods = []
    for m in sorted(sched.keys()):
        n = _lr_n(m)
        if n is None:
            continue
        if m in profiling_counts:
            lr_methods.append(m)
    lr_methods = sorted(lr_methods, key=lambda m: _lr_n(m))
    methods = [RTA_SS_tol_fb_METHOD] + lr_methods

    tol_ratio = (
        sched[RTA_SS_tol_fb_METHOD]["sched"] / sched[RTA_SS_tol_fb_METHOD]["total"]
        if sched[RTA_SS_tol_fb_METHOD]["total"] > 0
        else 0.0
    )

    # Build N ladder for promotion
    n_values = sorted({_lr_n(m) for m in lr_methods if _lr_n(m) is not None})
    n_index = {n: i for i, n in enumerate(n_values)}  # 0-based
    tol_tier = len(n_values) + 1

    rows = []
    for method in methods:
        sr = sched.get(method, {"sched": 0, "total": 0})
        sched_ratio = (sr["sched"] / sr["total"]) if sr["total"] > 0 else 0.0
        diff = abs(tol_ratio - sched_ratio)
        st = _stats(profiling_counts.get(method, []))

        promoted_to = method
        effective_tier = tol_tier if method == RTA_SS_tol_fb_METHOD else 0
        n = _lr_n(method)
        if n is not None:
            base_tier = n_index.get(n, -1) + 1
            effective_tier = base_tier
            if diff <= args.threshold:
                next_idx = n_index[n] + 1
                if next_idx < len(n_values):
                    promoted_to = f"{LR_PREFIX}{n_values[next_idx]}"
                    effective_tier = next_idx + 1
                else:
                    promoted_to = RTA_SS_tol_fb_METHOD
                    effective_tier = tol_tier

        rows.append(
            {
                "method_name": method,
                "sched_ratio": sched_ratio,
                "diff_vs_tol_max": diff,
                "promoted_to": promoted_to,
                "effective_tier": effective_tier,
                "count": st["count"] if st else 0,
                "mean": st["mean"] if st else None,
                "min": st["min"] if st else None,
                "q1": st["q1"] if st else None,
                "median": st["median"] if st else None,
                "q3": st["q3"] if st else None,
                "max": st["max"] if st else None,
            }
        )

    metrics = ["min", "q1", "median", "q3", "max", "mean"]
    top2 = {}
    for metric in metrics:
        rankable = [r for r in rows if r[metric] is not None]
        rankable.sort(
            key=lambda r: (
                -r["effective_tier"],   # higher promoted tier first
                float(r[metric]),       # lower profiling count metric better
                -r["sched_ratio"],      # higher sched ratio better
                r["method_name"],
            )
        )
        top2[metric] = rankable[:2]

    out_dir = run_dir / "postprocess_top2"
    out_dir.mkdir(parents=True, exist_ok=True)

    method_csv = out_dir / "method_summary.csv"
    with method_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method_name",
                "sched_ratio",
                "diff_vs_tol_max",
                "promoted_to",
                "effective_tier",
                "count",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "mean",
            ],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: (-x["effective_tier"], x["method_name"])):
            w.writerow(
                {
                    "method_name": r["method_name"],
                    "sched_ratio": _format_float(r["sched_ratio"]),
                    "diff_vs_tol_max": _format_float(r["diff_vs_tol_max"]),
                    "promoted_to": r["promoted_to"],
                    "effective_tier": r["effective_tier"],
                    "count": r["count"],
                    "min": _format_float(r["min"]),
                    "q1": _format_float(r["q1"]),
                    "median": _format_float(r["median"]),
                    "q3": _format_float(r["q3"]),
                    "max": _format_float(r["max"]),
                    "mean": _format_float(r["mean"]),
                }
            )

    top2_csv = out_dir / "top2_by_metric.csv"
    with top2_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "rank",
                "method_name",
                "metric_value",
                "sched_ratio",
                "diff_vs_tol_max",
                "promoted_to",
                "effective_tier",
            ],
        )
        w.writeheader()
        for metric in metrics:
            picks = top2.get(metric, [])
            for i, r in enumerate(picks, start=1):
                w.writerow(
                    {
                        "metric": metric,
                        "rank": i,
                        "method_name": r["method_name"],
                        "metric_value": _format_float(r[metric]),
                        "sched_ratio": _format_float(r["sched_ratio"]),
                        "diff_vs_tol_max": _format_float(r["diff_vs_tol_max"]),
                        "promoted_to": r["promoted_to"],
                        "effective_tier": r["effective_tier"],
                    }
                )

    report_path = out_dir / "report.txt"
    with report_path.open("w") as f:
        f.write("Top-2 Postprocess Report\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"threshold: {args.threshold}\n")
        f.write(f"RTA_SS_tol_fb method: {RTA_SS_tol_fb_METHOD}\n")
        f.write(f"RTA_SS_tol_fb sched_ratio: {tol_ratio:.6f}\n\n")

        f.write("[Method Summary]\n")
        for r in sorted(rows, key=lambda x: (-x["effective_tier"], x["method_name"])):
            f.write(
                f"- {r['method_name']}: sched_ratio={r['sched_ratio']:.6f}, "
                f"diff_vs_tol_max={r['diff_vs_tol_max']:.6f}, "
                f"promoted_to={r['promoted_to']}, effective_tier={r['effective_tier']}, "
                f"median={r['median']}, mean={r['mean']}\n"
            )

        f.write("\n[Top2 By Metric]\n")
        for metric in metrics:
            f.write(f"- {metric}\n")
            picks = top2.get(metric, [])
            if not picks:
                f.write("  (no data)\n")
                continue
            for i, r in enumerate(picks, start=1):
                f.write(
                    f"  {i}) {r['method_name']} "
                    f"(value={r[metric]}, sched_ratio={r['sched_ratio']:.6f}, "
                    f"promoted_to={r['promoted_to']}, tier={r['effective_tier']})\n"
                )

    print(f"Saved: {method_csv}")
    print(f"Saved: {top2_csv}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
