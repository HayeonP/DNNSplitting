#!/usr/bin/env python3
"""Unified trace runner for TOL_MAX and LR-by-n in one file.

Modes:
- tol_max  : generate TOL_MAX slice trace CSV/PNG
- r_best : generate Optimistic vs Actual trace CSV/PNG
- lr      : generate LR-by-n plots from existing trace CSVs
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# --------------------------
# Shared helpers
# --------------------------


def _util_tag(u: float) -> str:
    return f"{u:.3f}".rstrip("0").rstrip(".")


def _to_num(token: str):
    if token in {"None", "nan", "NaN", "inf", "-inf"}:
        if token == "inf":
            return float("inf")
        if token == "-inf":
            return float("-inf")
        return None
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return None


# --------------------------
# TOL_MAX trace implementation
# --------------------------


def _tol_find_log_for_util(log_dir: Path, prefix: str, utilization: float, required: bool = True) -> Path | None:
    candidates = sorted(log_dir.glob(f"{prefix}_task_set_list_u*.log"))
    if not candidates:
        if required:
            raise FileNotFoundError(f"No {prefix}_task_set_list_u*.log under: {log_dir}")
        return None

    best = None
    best_diff = None
    for path in candidates:
        m = re.search(r"_u(\d+(?:\.\d+)?)\.log$", path.name)
        if not m:
            continue
        u = float(m.group(1))
        diff = abs(u - utilization)
        if best is None or diff < best_diff:
            best = path
            best_diff = diff

    if best is None and required:
        raise FileNotFoundError(f"No utilization-tagged log found for prefix={prefix} in {log_dir}")
    return best


def _tol_parse_schedulability(rta_log_path: Path, method_name: str):
    out = {}
    current_idx = None
    current_method = None
    with rta_log_path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m_idx = re.match(r"^\[task_set_index=(\d+)\]$", line)
            if m_idx:
                current_idx = int(m_idx.group(1))
                out.setdefault(current_idx, None)
                current_method = None
                continue

            m_method = re.match(r"^\s{2}<([^>]+)>\s+schedulable=(True|False)$", line)
            if m_method:
                current_method = m_method.group(1)
                if current_idx is not None and current_method == method_name:
                    out[current_idx] = (m_method.group(2) == "True")
    return out


def _tol_parse_task_indices(rta_log_path: Path):
    out = {}
    current_idx = None
    in_task_info = False
    with rta_log_path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m_idx = re.match(r"^\[task_set_index=(\d+)\]$", line)
            if m_idx:
                current_idx = int(m_idx.group(1))
                out.setdefault(current_idx, [])
                in_task_info = False
                continue

            if current_idx is None:
                continue

            if line.strip() == "<task_info>":
                in_task_info = True
                continue

            if in_task_info and not line.startswith("    task("):
                in_task_info = False

            if not in_task_info:
                continue

            m_task = re.search(r"order=(\d+)", line)
            if not m_task:
                continue
            out[current_idx].append(int(m_task.group(1)))

    for idx in list(out.keys()):
        out[idx] = sorted(set(out[idx]))
    return out


def _tol_collect_rta_methods(rta_log_path: Path):
    methods = set()
    with rta_log_path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m_method = re.match(r"^\s{2}<([^>]+)>\s+schedulable=(True|False)$", line)
            if m_method:
                methods.add(m_method.group(1))
    return methods


def _tol_collect_tolerance_methods(tol_log_path: Path):
    methods = set()
    with tol_log_path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m_method = re.match(r"^\s{2}<([^>]+)>$", line)
            if m_method:
                methods.add(m_method.group(1))
    return methods


def _tol_select_method_name(rta_log_path: Path, tol_log_path: Path, preferred: str | None = None):
    common = _tol_collect_rta_methods(rta_log_path) & _tol_collect_tolerance_methods(tol_log_path)
    if not common:
        raise ValueError("No common tolerance method found between rta/tolerance logs.")

    if preferred:
        if preferred not in common:
            raise ValueError(f"Requested tol method '{preferred}' not found. common={sorted(common)}")
        return preferred

    preferred_exact = [
        "RTA_SS_tol_fb",
        "RTA_SS_tol_fb_early",
    ]
    for m in preferred_exact:
        if m in common:
            return m

    fallback_candidates = sorted(
        m for m in common
        if "RTA_SS_tol_fb" in m
        and "lr_early_n" not in m
        and "sentinel" not in m
    )
    if fallback_candidates:
        return fallback_candidates[0]

    lr_candidates = sorted(
        m for m in common
        if "RTA_SS_tol_fb_lr_early_n" in m
    )
    if lr_candidates:
        return lr_candidates[0]
    return sorted(common)[0]


def _tol_parse_log(tol_log_path: Path, method_name: str):
    result = {}
    current_idx = None
    current_method = None
    in_trace = False
    in_split_events = False
    in_max_fallback_changes = False

    with tol_log_path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m_idx = re.match(r"^\[task_set_index=(\d+)\]$", line)
            if m_idx:
                current_idx = int(m_idx.group(1))
                result.setdefault(current_idx, {"trace": [], "split_events": []})
                current_method = None
                in_trace = False
                in_split_events = False
                in_max_fallback_changes = False
                continue

            m_method = re.match(r"^\s{2}<([^>]+)>$", line)
            if m_method:
                current_method = m_method.group(1)
                in_trace = False
                in_split_events = False
                in_max_fallback_changes = False
                continue

            if current_idx is None or current_method != method_name:
                continue

            if line.strip().startswith("trace:"):
                in_trace = True
                in_split_events = False
                continue

            if line.strip().startswith("split_events:"):
                in_trace = False
                in_split_events = True
                in_max_fallback_changes = False
                continue

            if line.strip().startswith("max_fallback_changes:"):
                in_trace = False
                in_split_events = False
                in_max_fallback_changes = True
                continue

            if in_trace:
                if not line.startswith("      "):
                    in_trace = False
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                # New format variants:
                #   split_count idx R D B B_low I tolerance_i tolerance_target
                #   split_count idx R D B B_low B_high I C G max_block_size tolerance_i tolerance_target
                # Legacy format:
                #   idx R D B B_low I tolerance_i tolerance_target before after
                if len(parts) == 10:
                    split_count = None
                    idx_pos = 0
                    b_high = None
                    max_block_size = None
                    i_val = _to_num(parts[idx_pos + 5])
                elif len(parts) >= 13:
                    split_count = _to_num(parts[0])
                    idx_pos = 1
                    b_high = _to_num(parts[idx_pos + 5])
                    max_block_size = _to_num(parts[idx_pos + 9])
                    i_val = _to_num(parts[idx_pos + 6])
                    tolerance_i = _to_num(parts[idx_pos + 10])
                else:
                    split_count = _to_num(parts[0])
                    idx_pos = 1
                    b_high = None
                    max_block_size = None
                    i_val = _to_num(parts[idx_pos + 5])
                    tolerance_i = _to_num(parts[idx_pos + 6])
                if len(parts) == 10:
                    tolerance_i = _to_num(parts[idx_pos + 6])
                result[current_idx]["trace"].append(
                    {
                        "step": len(result[current_idx]["trace"]),
                        "split_count": split_count,
                        "task_index": _to_num(parts[idx_pos + 0]),
                        "R": _to_num(parts[idx_pos + 1]),
                        "D": _to_num(parts[idx_pos + 2]),
                        "B": _to_num(parts[idx_pos + 3]),
                        "B_low": _to_num(parts[idx_pos + 4]),
                        "B_high": b_high,
                        "I": i_val,
                        "max_block_size": max_block_size,
                        "tolerance_i": tolerance_i,
                    }
                )
                continue

            if in_split_events:
                if not line.startswith("      "):
                    in_split_events = False
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                result[current_idx]["split_events"].append(
                    {
                        "step": _to_num(parts[0]),
                        "trigger_task_index": _to_num(parts[1]),
                        "target_task_index": _to_num(parts[2]),
                        "mode": parts[3],
                        "split_count": _to_num(parts[4]) if len(parts) >= 5 else None,
                    }
                )
                continue

            if in_max_fallback_changes:
                if not line.startswith("      "):
                    in_max_fallback_changes = False
                    continue
                # format:
                # step trigger target split_count ... decision mode
                parts = line.split()
                if len(parts) < 6:
                    continue
                result[current_idx]["split_events"].append(
                    {
                        "step": _to_num(parts[0]),
                        "trigger_task_index": _to_num(parts[1]),
                        "target_task_index": _to_num(parts[2]),
                        "split_count": _to_num(parts[3]),
                        "mode": parts[-1],
                        "decision": parts[-2],
                        "source": "max_fallback_changes",
                    }
                )

    return result


def _tol_build_event_index_for_split_counts(split_events, by_slice):
    indexed = {}
    if not by_slice:
        return indexed

    slice_steps = {}
    for sc, by_task in by_slice.items():
        steps = [r.get("step") for r in by_task.values() if r.get("step") is not None]
        if steps:
            slice_steps[sc] = (min(steps), max(steps))

    ordered_scs = sorted(by_slice.keys())
    for sc in ordered_scs:
        indexed.setdefault(sc, [])

    for ev in split_events or []:
        assigned = None
        explicit_sc = ev.get("split_count")
        if explicit_sc in by_slice:
            assigned = explicit_sc
        else:
            ev_step = ev.get("step")
            if ev_step is not None and slice_steps:
                for sc in ordered_scs:
                    bounds = slice_steps.get(sc)
                    if not bounds:
                        continue
                    lo, hi = bounds
                    if lo <= ev_step <= hi:
                        assigned = sc
                        break
                if assigned is None:
                    past = [(sc, slice_steps[sc][1]) for sc in ordered_scs if sc in slice_steps and slice_steps[sc][1] <= ev_step]
                    if past:
                        assigned = max(past, key=lambda x: x[1])[0]
                    else:
                        future = [(sc, slice_steps[sc][0]) for sc in ordered_scs if sc in slice_steps and slice_steps[sc][0] >= ev_step]
                        if future:
                            assigned = min(future, key=lambda x: x[1])[0]

        if assigned is None:
            continue
        indexed.setdefault(assigned, []).append(ev)

    return indexed


def _tol_build_snapshots_by_slice(trace_rows, split_events, all_task_indices=None):
    if not trace_rows:
        return [], []

    # Preferred path (new logging): rows already tagged with split_count,
    # and represent post-slice snapshots only.
    by_slice = {}
    for r in trace_rows:
        sc = r.get("split_count")
        tidx = r.get("task_index")
        if sc is None or tidx is None:
            continue
        by_slice.setdefault(sc, {})
        by_slice[sc][tidx] = r

    if by_slice:
        out = []
        indexed_events = []
        if all_task_indices:
            task_indices = sorted(set(all_task_indices))
        else:
            task_indices = sorted({tidx for by_task in by_slice.values() for tidx in by_task.keys()})

        events_by_sc = _tol_build_event_index_for_split_counts(split_events, by_slice)

        for sc in sorted(by_slice.keys()):
            evs = events_by_sc.get(sc, [])
            indexed_events.append({"split_count": sc, "events": evs})

            commit_events = [
                ev
                for ev in evs
                if ev.get("target_task_index") is not None
                and ev.get("mode") is not None
                and "cancel" not in str(ev.get("mode"))
            ]
            target_mode_pairs = []
            for ev in commit_events:
                target_mode_pairs.append((ev.get("target_task_index"), ev.get("mode")))
            if not target_mode_pairs:
                # Fallback: include rollback/cancel target events as unknown-mode targets.
                for ev in evs:
                    t = ev.get("target_task_index")
                    if t is None:
                        continue
                    target_mode_pairs.append((t, ev.get("mode") or "unknown"))
            unique_pairs = []
            seen_pairs = set()
            for t, m in target_mode_pairs:
                key = (t, m)
                if t is None or key in seen_pairs:
                    continue
                seen_pairs.add(key)
                unique_pairs.append((t, m))
            selected_targets = sorted({t for t, _ in unique_pairs})
            trigger_set = sorted({ev.get("trigger_task_index") for ev in evs if ev.get("trigger_task_index") is not None})
            mode_set = sorted({ev.get("mode") for ev in evs if ev.get("mode")})
            event_steps = sorted({ev.get("step") for ev in evs if ev.get("step") is not None})

            for tidx in task_indices:
                src = by_slice[sc].get(tidx)
                out.append(
                    {
                        "split_count": sc,
                        "step": None if src is None else src.get("step"),
                        "event_step": event_steps[-1] if event_steps else None,
                        "row_phase": "after",
                        "task_index": tidx,
                        "R": None if src is None else src.get("R"),
                        "B_low": None if src is None else src.get("B_low"),
                        "B_high": (
                            src.get("B_high")
                            if src is not None and src.get("B_high") is not None
                            else (
                                src.get("B") - src.get("B_low")
                                if src is not None and src.get("B") is not None and src.get("B_low") is not None
                                else None
                            )
                        ),
                        "max_block_size": (
                            None if src is None else src.get("max_block_size")
                        ),
                        "tolerance_i": None if src is None else src.get("tolerance_i"),
                        "event_target_task_indices": "|".join(str(v) for v in selected_targets),
                        "event_target_task_modes": "|".join(f"{t}:{m}" for t, m in unique_pairs),
                        "event_trigger_task_indices": "|".join(str(v) for v in trigger_set),
                        "event_modes": "|".join(mode_set),
                    }
                )
        return out, indexed_events

    # Legacy fallback path: reconstruct before/after from old trace/event logs.
    if not split_events:
        return [], []

    rows_sorted = sorted(
        [r for r in trace_rows if r.get("step") is not None and r.get("task_index") is not None],
        key=lambda r: (r["step"], r["task_index"]),
    )
    if not rows_sorted:
        return [], []

    events_sorted = sorted(
        [ev for ev in split_events if ev.get("step") is not None],
        key=lambda ev: (ev["step"], ev.get("target_task_index", -1)),
    )
    if not events_sorted:
        return [], []

    out = []
    indexed_events = []

    latest_after = {}
    row_idx_after = 0

    if all_task_indices:
        task_indices = sorted(set(all_task_indices))
    else:
        task_indices = sorted({r["task_index"] for r in rows_sorted if r.get("task_index") is not None})

    for s_count, ev in enumerate(events_sorted, start=1):
        ev_step = ev["step"]

        while row_idx_after < len(rows_sorted) and rows_sorted[row_idx_after]["step"] <= ev_step:
            row = rows_sorted[row_idx_after]
            latest_after[row["task_index"]] = row
            row_idx_after += 1

        indexed_events.append(
            {
                "split_count": s_count,
                "step": ev_step,
                "trigger_task_index": ev.get("trigger_task_index"),
                "target_task_index": ev.get("target_task_index"),
                "mode": ev.get("mode"),
            }
        )

        for tidx in task_indices:
            src = latest_after.get(tidx)
            out.append(
                {
                    "split_count": s_count,
                    "step": None if src is None else src.get("step"),
                    "event_step": ev_step,
                    "row_phase": "after",
                    "task_index": tidx,
                    "R": None if src is None else src.get("R"),
                    "B_low": None if src is None else src.get("B_low"),
                    "B_high": (
                        src.get("B_high")
                        if src is not None and src.get("B_high") is not None
                        else (
                            src.get("B") - src.get("B_low")
                            if src is not None and src.get("B") is not None and src.get("B_low") is not None
                            else None
                        )
                    ),
                    "max_block_size": None if src is None else src.get("max_block_size"),
                    "tolerance_i": None if src is None else src.get("tolerance_i"),
                    "event_target_task_indices": (
                        str(ev.get("target_task_index"))
                        if ev.get("target_task_index") is not None
                        and ev.get("mode") is not None
                        and "cancel" not in str(ev.get("mode"))
                        else ""
                    ),
                    "event_target_task_modes": (
                        f"{ev.get('target_task_index')}:{ev.get('mode')}"
                        if ev.get("target_task_index") is not None and ev.get("mode")
                        else ""
                    ),
                    "event_trigger_task_indices": str(ev.get("trigger_task_index"))
                    if ev.get("trigger_task_index") is not None
                    else "",
                    "event_modes": ev.get("mode") or "",
                }
            )

    return out, indexed_events


def _tol_write_snapshot_csv(rows, out_csv: Path):
    if not rows:
        return
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split_count",
                "step",
                "event_step",
                "row_phase",
                "task_index",
                "R",
                "B_low",
                "B_high",
                "max_block_size",
                "tolerance_i",
                "event_target_task_indices",
                "event_target_task_modes",
                "event_trigger_task_indices",
                "event_modes",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def _tol_plot(rows, out_png: Path, title: str):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if not rows:
        return

    rows_plot = [r for r in rows if r.get("row_phase") in {"before", "after"}]
    if not rows_plot:
        return

    all_task_indices = sorted({r["task_index"] for r in rows_plot if r.get("task_index") is not None})
    task_indices = all_task_indices
    if not task_indices:
        return

    palette = list(mcolors.TABLEAU_COLORS.values())
    if not palette:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i) for i in range(20)]
    colors = {t: palette[i % len(palette)] for i, t in enumerate(task_indices)}

    grouped = {t: [] for t in task_indices}
    for row in rows_plot:
        tidx = row.get("task_index")
        if tidx in grouped:
            grouped[tidx].append(row)
    for tidx in task_indices:
        grouped[tidx].sort(
            key=lambda r: (
                r.get("split_count", -1),
                0 if r.get("row_phase") == "before" else 1,
                r.get("step", -1),
            )
        )

    metrics = [
        ("R", "R"),
        ("B_low", "B_low"),
        ("B_high", "B_high"),
        ("max_block_size", "Max block size"),
        ("tolerance_i", "tolerance_i"),
    ]
    fig, axes = plt.subplots(6, 1, figsize=(13, 22), sharex=True)
    metric_axes = axes[:5]
    event_ax = axes[5]

    # Prepare per-split task info (shown in a dedicated bottom panel to avoid clutter).
    event_targets_by_sc = {}
    event_triggers_by_sc = {}
    for row in rows_plot:
        sc = row.get("split_count")
        raw_targets = row.get("event_target_task_indices")
        raw_target_modes = row.get("event_target_task_modes")
        raw_triggers = row.get("event_trigger_task_indices")
        if sc is None:
            continue
        sc = int(sc)
        event_targets_by_sc.setdefault(sc, {"tol": set(), "max": set(), "other": set()})
        if raw_target_modes:
            for tok in str(raw_target_modes).split("|"):
                tok = tok.strip()
                if not tok or ":" not in tok:
                    continue
                t_str, mode = tok.split(":", 1)
                try:
                    tidx = int(t_str.strip())
                except ValueError:
                    continue
                if tidx not in task_indices:
                    continue
                mode_l = str(mode).lower()
                if "max_fallback" in mode_l:
                    event_targets_by_sc[sc]["max"].add(tidx)
                elif mode_l.startswith("tol"):
                    event_targets_by_sc[sc]["tol"].add(tidx)
                else:
                    event_targets_by_sc[sc]["other"].add(tidx)
        elif raw_targets:
            targets = []
            for tok in str(raw_targets).split("|"):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    targets.append(int(tok))
                except ValueError:
                    continue
            if targets:
                filtered = sorted({t for t in targets if t in task_indices})
                if filtered:
                    # Legacy rows don't carry per-target mode; classify with event_modes when possible.
                    mode_l = str(row.get("event_modes") or "").lower()
                    if "max_fallback" in mode_l:
                        event_targets_by_sc[sc]["max"].update(filtered)
                    elif "tol" in mode_l:
                        event_targets_by_sc[sc]["tol"].update(filtered)
                    else:
                        event_targets_by_sc[sc]["other"].update(filtered)
        if sc is None or not raw_triggers:
            continue
        triggers = []
        for tok in str(raw_triggers).split("|"):
            tok = tok.strip()
            if not tok:
                continue
            try:
                triggers.append(int(tok))
            except ValueError:
                continue
        if triggers:
            filtered = sorted({t for t in triggers if t in task_indices})
            if filtered:
                event_triggers_by_sc[sc] = filtered

    for ax, (key, label) in zip(metric_axes, metrics):
        for tidx in task_indices:
            entries = [e for e in grouped[tidx] if e.get(key) is not None and e.get("split_count") is not None]
            if not entries:
                continue

            xs = [e["split_count"] for e in entries]
            ys = [e[key] for e in entries]
            color = colors[tidx]

            ax.plot(xs, ys, linewidth=1.1, color=color, label=f"task[{tidx}]")

            prev = None
            for x, y in zip(xs, ys):
                if prev is None:
                    marker = "o"
                elif y > prev:
                    marker = "^"
                elif y < prev:
                    marker = "v"
                else:
                    marker = "o"
                ax.scatter([x], [y], marker=marker, s=34, color=color, edgecolors="black", linewidths=0.7, zorder=3)
                prev = y

        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8, ncol=4)
        for sc in sorted(event_targets_by_sc.keys()):
            ax.axvline(sc, color="gray", linestyle=":", linewidth=0.8, alpha=0.45, zorder=1)

    # Bottom panel: which task(s) got profiled at each profiling_count.
    for sc in sorted(event_targets_by_sc.keys()):
        mode_to_style = {
            "tol": ("tab:blue", "x"),
            "max": ("tab:red", "x"),
            "other": ("black", "x"),
        }
        for mode_name, (color, marker) in mode_to_style.items():
            targets = sorted(event_targets_by_sc[sc][mode_name])
            for t in targets:
                event_ax.scatter(
                    [sc],
                    [t],
                    marker=marker,
                    s=42,
                    color=color,
                    linewidths=1.2,
                    zorder=4,
                )
                event_ax.annotate(
                    f"t{t}",
                    (sc, t),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color=color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
                )
    for sc in sorted(event_triggers_by_sc.keys()):
        triggers = event_triggers_by_sc[sc]
        for t in triggers:
            event_ax.scatter(
                [sc],
                [t],
                marker="+",
                s=30,
                color="dimgray",
                linewidths=1.1,
                zorder=4,
            )
            event_ax.annotate(
                f"tr{t}",
                (sc, t),
                textcoords="offset points",
                xytext=(0, -10),
                ha="center",
                fontsize=7,
                color="dimgray",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
            )
    event_ax.set_ylabel("Split task")
    event_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    event_ax.set_yticks(task_indices)

    axes[0].set_title(title)
    all_split_counts = sorted({r.get("split_count") for r in rows_plot if r.get("split_count") is not None})
    if all_split_counts:
        event_ax.set_xticks(all_split_counts)
        event_ax.set_xticklabels(
            [str(sc) if int(sc) % 5 == 0 else "" for sc in all_split_counts]
        )
    event_ax.set_xlabel("Profiling Count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _tol_process_index(idx, info_tol, sched_map, util_tag, sched_dir, unsched_dir, all_task_indices, method_tag):
    sched = sched_map.get(idx)
    sched = bool(sched) if sched is not None else False

    out_dir = Path(sched_dir) if sched else Path(unsched_dir)
    sched_tag = "sched" if sched else "unsched"
    stem = f"trace_{method_tag}_u{util_tag}_idx{idx}_{sched_tag}"

    rows = info_tol.get("trace", [])
    events = info_tol.get("split_events", [])
    snap_rows, indexed_events = _tol_build_snapshots_by_slice(
        rows,
        events,
        all_task_indices=all_task_indices,
    )
    if not snap_rows:
        return idx, sched, 0, "skip_no_split_events"

    _tol_write_snapshot_csv(snap_rows, out_dir / f"{stem}_splitting_trace.csv")
    _tol_plot(snap_rows, out_dir / f"{stem}_splitting_trace.png", f"{method_tag.upper()} Slice Trace")
    return idx, sched, len(indexed_events), "ok"


def run_tol_trace(
    run_dir: Path,
    utilization: float,
    output_dir: Path,
    task_set_idx: int | None,
    max_workers: int | None,
    method_name: str | None = None,
):
    rta_log_dir = run_dir / "rta_logs"
    if not rta_log_dir.exists():
        raise FileNotFoundError(f"rta_logs not found: {rta_log_dir}")

    rta_log = _tol_find_log_for_util(rta_log_dir, "rta", utilization, required=True)
    tol_log = _tol_find_log_for_util(rta_log_dir, "tolerance", utilization, required=True)
    selected_method = _tol_select_method_name(rta_log, tol_log, preferred=method_name)
    method_tag = "sentinel" if "sentinel" in selected_method.lower() else "tol_max"

    sched_map = _tol_parse_schedulability(rta_log, selected_method)
    task_indices_map = _tol_parse_task_indices(rta_log)
    tol_map = _tol_parse_log(tol_log, selected_method)

    all_indices = sorted(set(sched_map.keys()) | set(tol_map.keys()))
    if task_set_idx is None:
        target_indices = all_indices
    else:
        if task_set_idx not in all_indices:
            raise IndexError(f"task_set_idx not found: {task_set_idx}")
        target_indices = [task_set_idx]

    util_tag = _util_tag(utilization)
    sched_dir = output_dir / "schedulable"
    unsched_dir = output_dir / "unschedulable"
    sched_dir.mkdir(parents=True, exist_ok=True)
    unsched_dir.mkdir(parents=True, exist_ok=True)

    workers = max_workers
    if workers is None:
        workers = min(os.cpu_count() or 1, max(1, len(target_indices)))
    workers = max(1, min(workers, max(1, len(target_indices))))

    print(f"[trace:tol_max] run_dir={run_dir}")
    print(f"[trace:tol_max] rta_log={rta_log}")
    print(f"[trace:tol_max] tol_log={tol_log}")
    print(f"[trace:tol_max] method={selected_method}")
    print(f"[trace:tol_max] target_indices={len(target_indices)} workers={workers}")

    if workers == 1 or len(target_indices) <= 1:
        for idx in target_indices:
            i, sched, n_events, status = _tol_process_index(
                idx,
                tol_map.get(idx, {}),
                sched_map,
                util_tag,
                str(sched_dir),
                str(unsched_dir),
                task_indices_map.get(idx),
                method_tag,
            )
            print(f"[trace:tol_max] idx={i} schedulable={sched} split_events={n_events} status={status}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _tol_process_index,
                    idx,
                    tol_map.get(idx, {}),
                    sched_map,
                    util_tag,
                    str(sched_dir),
                    str(unsched_dir),
                    task_indices_map.get(idx),
                    method_tag,
                )
                for idx in target_indices
            ]
            for fut in as_completed(futures):
                i, sched, n_events, status = fut.result()
                print(f"[trace:tol_max] idx={i} schedulable={sched} split_events={n_events} status={status}")

    print(f"[trace:tol_max] saved_root={output_dir}")


# --------------------------
# R_best trace implementation
# --------------------------


def _rbest_parse_log(tol_log_path: Path):
    result = {}
    current_idx = None
    current_method = None
    in_rbest_trace = False

    with tol_log_path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m_idx = re.match(r"^\[task_set_index=(\d+)\]$", line)
            if m_idx:
                current_idx = int(m_idx.group(1))
                result.setdefault(current_idx, {})
                current_method = None
                in_rbest_trace = False
                continue

            m_method = re.match(r"^\s{2}<([^>]+)>$", line)
            if m_method:
                current_method = m_method.group(1)
                result[current_idx].setdefault(current_method, [])
                in_rbest_trace = False
                continue

            if current_idx is None or current_method is None:
                continue

            if line.strip().startswith("r_best_trace:"):
                in_rbest_trace = True
                continue

            if in_rbest_trace:
                if not line.startswith("      "):
                    in_rbest_trace = False
                    continue
                parts = line.split()
                if len(parts) < 12:
                    continue
                result[current_idx][current_method].append(
                    {
                        "step": _to_num(parts[0]),
                        "split_count": _to_num(parts[1]),
                        "trigger_task_index": _to_num(parts[2]),
                        "target_task_index": _to_num(parts[3]),
                        "task_index": _to_num(parts[4]),
                        "task_id": _to_num(parts[5]),
                        "R_best": _to_num(parts[6]),
                        "R_actual": _to_num(parts[7]),
                        "D": _to_num(parts[8]),
                        "best_sched": parts[9],
                        "actual_sched": parts[10],
                        "B_low_best": _to_num(parts[11]),
                    }
                )

    return result


def _rbest_select_method_name(rta_log_path: Path, rbest_map, preferred: str | None = None):
    methods_in_rta = _tol_collect_rta_methods(rta_log_path)
    methods_with_rows = {}
    for idx_map in rbest_map.values():
        for method, rows in idx_map.items():
            if rows:
                methods_with_rows[method] = methods_with_rows.get(method, 0) + len(rows)
    common = sorted(m for m in methods_with_rows.keys() if m in methods_in_rta)
    if not common:
        raise ValueError("No method with r_best_trace rows found in tolerance log.")

    if preferred:
        if preferred not in common:
            raise ValueError(f"Requested r_best method '{preferred}' not found. common={common}")
        return preferred

    exact = "RTA_SS_tol_fb_rbest"
    if exact in common:
        return exact
    return max(common, key=lambda m: methods_with_rows.get(m, 0))


def _rbest_build_rows_for_idx(rbest_map, idx: int, method_name: str):
    rows = (rbest_map.get(idx) or {}).get(method_name, [])
    if not rows:
        return []

    grouped = {}
    for r in rows:
        sc = r.get("split_count")
        tidx = r.get("task_index")
        if sc is None or tidx is None:
            continue
        grouped.setdefault((int(sc), int(tidx)), r)

    out = []
    for (_, _), r in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append(
            {
                "split_count": int(r["split_count"]),
                "task_index": int(r["task_index"]),
                "task_id": r.get("task_id"),
                "R_best": r.get("R_best"),
                "R_actual": r.get("R_actual"),
                "D": r.get("D"),
                "step": r.get("step"),
                "trigger_task_index": r.get("trigger_task_index"),
                "target_task_index": r.get("target_task_index"),
                "best_sched": r.get("best_sched"),
                "actual_sched": r.get("actual_sched"),
                "B_low_best": r.get("B_low_best"),
            }
        )
    return out


def _rbest_write_csv(rows, out_csv: Path):
    if not rows:
        return
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split_count",
                "task_index",
                "task_id",
                "R_best",
                "R_actual",
                "D",
                "step",
                "trigger_task_index",
                "target_task_index",
                "best_sched",
                "actual_sched",
                "B_low_best",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def _rbest_plot(rows, out_png: Path, title: str):
    import matplotlib.pyplot as plt

    if not rows:
        return

    by_task = {}
    for r in rows:
        tidx = r.get("task_index")
        sc = r.get("split_count")
        if tidx is None or sc is None:
            continue
        by_task.setdefault(int(tidx), []).append(r)

    if not by_task:
        return

    cmap = plt.get_cmap("tab20")
    task_ids = sorted(by_task.keys())
    color_map = {t: cmap(i % 20) for i, t in enumerate(task_ids)}
    n_tasks = len(task_ids)
    ncols = 2 if n_tasks > 1 else 1
    nrows = math.ceil(n_tasks / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.3 * nrows), sharex=True)
    if not hasattr(axes, "ravel"):
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    for i, t in enumerate(task_ids):
        ax = axes_list[i]
        rows_t = sorted(by_task[t], key=lambda x: x["split_count"])
        xs = [r["split_count"] for r in rows_t]
        y_actual = [r.get("R_actual") for r in rows_t]
        y_best = [r.get("R_best") for r in rows_t]
        d_vals = [r.get("D") for r in rows_t if r.get("D") is not None]
        c = color_map[t]

        if any(v is not None for v in y_actual):
            ax.plot(xs, y_actual, color=c, linewidth=1.8, marker="o", markersize=3, label="Actual")
        if any(v is not None for v in y_best):
            ax.plot(xs, y_best, color=c, linewidth=1.6, linestyle="--", marker="x", markersize=3, label="Optimistic")
        if d_vals:
            ax.axhline(
                d_vals[0],
                color="black",
                linewidth=1.6,
                linestyle="-",
                alpha=0.95,
                label=f"deadline={d_vals[0]}",
            )

        ax.set_title(f"task[{t}]")
        ax.set_ylabel("WCRT")
        ax.set_xlabel("Splitting decisions")
        ax.set_ylim(0, 11000)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=12, ncol=3)

    for j in range(n_tasks, len(axes_list)):
        axes_list[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def run_r_best_trace(
    run_dir: Path,
    utilization: float,
    output_dir: Path,
    task_set_idx: int | None,
    method_name: str | None = None,
):
    rta_log_dir = run_dir / "rta_logs"
    if not rta_log_dir.exists():
        raise FileNotFoundError(f"rta_logs not found: {rta_log_dir}")

    rta_log = _tol_find_log_for_util(rta_log_dir, "rta", utilization, required=True)
    tol_log = _tol_find_log_for_util(rta_log_dir, "tolerance", utilization, required=True)
    rbest_map = _rbest_parse_log(tol_log)
    selected_method = _rbest_select_method_name(rta_log, rbest_map, preferred=method_name)
    sched_map = _tol_parse_schedulability(rta_log, selected_method)

    all_indices = sorted(set(sched_map.keys()) | set(rbest_map.keys()))
    if task_set_idx is None:
        target_indices = all_indices
    else:
        if task_set_idx not in all_indices:
            raise IndexError(f"task_set_idx not found: {task_set_idx}")
        target_indices = [task_set_idx]

    util_tag = _util_tag(utilization)
    sched_dir = output_dir / "schedulable"
    unsched_dir = output_dir / "unschedulable"
    sched_dir.mkdir(parents=True, exist_ok=True)
    unsched_dir.mkdir(parents=True, exist_ok=True)

    method_tag = "rbest"
    saved = 0
    print(f"[trace:r_best] run_dir={run_dir}")
    print(f"[trace:r_best] rta_log={rta_log}")
    print(f"[trace:r_best] tol_log={tol_log}")
    print(f"[trace:r_best] method={selected_method}")
    print(f"[trace:r_best] target_indices={len(target_indices)}")

    for idx in target_indices:
        rows = _rbest_build_rows_for_idx(rbest_map, idx, selected_method)
        if not rows:
            print(f"[trace:r_best] idx={idx} status=skip_no_rbest_rows")
            continue
        sched = bool(sched_map.get(idx, False))
        sched_tag = "sched" if sched else "unsched"
        out_dir = sched_dir if sched else unsched_dir
        stem = f"trace_{method_tag}_u{util_tag}_idx{idx}_{sched_tag}"
        out_csv = out_dir / f"{stem}_trace.csv"
        out_png = out_dir / f"{stem}_trace.png"
        _rbest_write_csv(rows, out_csv)
        _rbest_plot(rows, out_png, f"Optimistic vs Actual | idx={idx} ({sched_tag})")
        saved += 1
        print(f"[trace:r_best] idx={idx} schedulable={sched} status=ok rows={len(rows)}")

    print(f"[trace:r_best] saved_plots={saved}")
    print(f"[trace:r_best] saved_root={output_dir}")


# --------------------------
# LR-by-n implementation
# --------------------------


def _lr_extract_taskset_idx(path: Path):
    m = re.search(r"_idx(\d+)_", path.name)
    return int(m.group(1)) if m else None


def _lr_to_float(token: str):
    if token in {None, "", "inf", "-inf", "nan", "NaN", "None"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _lr_linear_regression_slope(xs, ys):
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


def _lr_iter_target_csvs(trace_root: Path, subset: str):
    if subset == "all":
        subsets = ["unschedulable", "schedulable"]
    elif subset == "unsched":
        subsets = ["unschedulable"]
    else:
        subsets = ["schedulable"]

    for sub in subsets:
        folder = trace_root / sub
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*_splitting_trace.csv")):
            yield sub, p


def _lr_read_rows(path: Path, metric: str):
    by_task = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_phase") != "after":
                continue

            t_raw = row.get("task_index")
            x_raw = row.get("profiling_count", row.get("split_count"))
            y_raw = row.get(metric)
            if t_raw in (None, "") or x_raw in (None, ""):
                continue

            y = _lr_to_float(y_raw)
            if y is None:
                continue

            try:
                task_idx = int(t_raw)
                x = int(float(x_raw))
            except ValueError:
                continue

            by_task.setdefault(task_idx, []).append((x, y))

    normalized = {}
    for task_idx, points in by_task.items():
        points.sort(key=lambda p: p[0])
        tmp = {}
        for x, y in points:
            tmp[x] = y
        normalized[task_idx] = sorted(tmp.items(), key=lambda p: p[0])
    return normalized


def _lr_build_series(points, n):
    if len(points) < n:
        return [], []

    xs_out = []
    ys_out = []
    for i in range(0, len(points) - n + 1):
        window = points[i : i + n]
        xs = [p[0] for p in window]
        ys = [p[1] for p in window]
        slope = _lr_linear_regression_slope(xs, ys)
        if slope is None or math.isnan(slope) or math.isinf(slope):
            continue
        xs_out.append(window[0][0])
        ys_out.append(slope)
    return xs_out, ys_out


def _lr_plot_single_csv(csv_path: Path, subset_name: str, out_dir: Path, metric: str, n_list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task = _lr_read_rows(csv_path, metric=metric)
    if not by_task:
        return None

    task_ids = sorted(by_task.keys())
    cmap = plt.get_cmap("tab20")
    color_map = {tid: cmap(i % 20) for i, tid in enumerate(task_ids)}

    fig, axes = plt.subplots(len(n_list), 1, figsize=(14, 3.6 * len(n_list)), squeeze=False)
    axes = [row[0] for row in axes]

    has_any = False
    for row_idx, n in enumerate(n_list):
        ax = axes[row_idx]
        for tid in task_ids:
            xs, ys = _lr_build_series(by_task[tid], n)
            if not xs:
                continue
            has_any = True
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.5, color=color_map[tid], label=f"task[{tid}]")

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(f"n={n}")
        ax.set_xlabel("window start profiling_count (x -> x..x+n-1)")
        ax.set_ylabel(f"LR slope ({metric})")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if not has_any:
        plt.close(fig)
        return None

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper right", ncol=1)

    taskset_idx = _lr_extract_taskset_idx(csv_path)
    fig.suptitle(f"LR Slope By n | idx={taskset_idx} | subset={subset_name} | phase=after | metric={metric}")
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tag = "-".join(str(n) for n in n_list)
    out_name = csv_path.stem.replace("_splitting_trace", f"_lr_{metric}_after_n{n_tag}") + ".png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _lr_plot_single_rows(name_stem: str, subset_name: str, rows, out_dir: Path, metric: str, n_list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task = {}
    for row in rows:
        if row.get("row_phase") != "after":
            continue
        t_raw = row.get("task_index")
        x_raw = row.get("profiling_count", row.get("split_count"))
        y_raw = row.get(metric)
        if t_raw is None or x_raw is None:
            continue
        y = _lr_to_float(str(y_raw))
        if y is None:
            continue
        try:
            task_idx = int(t_raw)
            x = int(float(x_raw))
        except (TypeError, ValueError):
            continue
        by_task.setdefault(task_idx, []).append((x, y))

    if not by_task:
        return None

    normalized = {}
    for task_idx, points in by_task.items():
        points.sort(key=lambda p: p[0])
        tmp = {}
        for x, y in points:
            tmp[x] = y
        normalized[task_idx] = sorted(tmp.items(), key=lambda p: p[0])
    by_task = normalized

    task_ids = sorted(by_task.keys())
    cmap = plt.get_cmap("tab20")
    color_map = {tid: cmap(i % 20) for i, tid in enumerate(task_ids)}

    fig, axes = plt.subplots(len(n_list), 1, figsize=(14, 3.6 * len(n_list)), squeeze=False)
    axes = [row[0] for row in axes]

    has_any = False
    for row_idx, n in enumerate(n_list):
        ax = axes[row_idx]
        for tid in task_ids:
            xs, ys = _lr_build_series(by_task[tid], n)
            if not xs:
                continue
            has_any = True
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.5, color=color_map[tid], label=f"task[{tid}]")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(f"n={n}")
        ax.set_xlabel("window start profiling_count (x -> x..x+n-1)")
        ax.set_ylabel(f"LR slope ({metric})")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if not has_any:
        plt.close(fig)
        return None

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper right", ncol=1)

    fig.suptitle(f"LR Slope By n | {name_stem} | phase=after | metric={metric}")
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tag = "-".join(str(n) for n in n_list)
    out_name = f"{name_stem}_lr_{metric}_after_n{n_tag}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _lr_parse_n_list(raw: str):
    values = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n < 2:
            raise ValueError("all n must be >= 2")
        values.append(n)
    if not values:
        raise ValueError("n list is empty")
    return sorted(set(values))


def _bool_from_text(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _plot_lr_taskset_from_logged_rows(rows, out_png: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task = {}
    for row in rows:
        tidx = row.get("task_index")
        x = row.get("profiling_count", row.get("split_count"))
        y = row.get("lr_slope")
        if tidx is None or x is None or y is None:
            continue
        by_task.setdefault(int(tidx), []).append((int(x), float(y)))

    if not by_task:
        return False

    task_ids = sorted(by_task.keys())
    cmap = plt.get_cmap("tab20")
    color_map = {tid: cmap(i % 20) for i, tid in enumerate(task_ids)}

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for tid in task_ids:
        points = sorted(by_task[tid], key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, color=color_map[tid], label=f"task[{tid}]")

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("last slice count")
    ax.set_ylabel("LR slope")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return True


def run_lr_trace_from_logged_csv(
    run_dir: Path,
    utilization: float,
    subset: str,
    n_list_raw: str,
    out_dir: Path,
    task_set_idx: int | None,
):
    util_tag = _util_tag(utilization)
    n_list = _lr_parse_n_list(n_list_raw)
    lr_log_dir = run_dir / "lr_logs"
    if not lr_log_dir.exists():
        print(f"[trace:lr] missing lr_logs dir: {lr_log_dir}")
        return 0
    include_sched = subset in {"all", "sched"}
    include_unsched = subset in {"all", "unsched"}
    saved_count = 0

    for n_val in n_list:
        csv_path = lr_log_dir / f"u{util_tag}_lr_n{n_val}.csv"
        if not csv_path.exists():
            print(f"[trace:lr] skip missing csv: {csv_path}")
            continue

        grouped = {}
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row.get("task_set_index"))
                except (TypeError, ValueError):
                    continue
                if task_set_idx is not None and idx != task_set_idx:
                    continue
                sched = _bool_from_text(row.get("schedulable"))
                if sched and not include_sched:
                    continue
                if (not sched) and not include_unsched:
                    continue
                try:
                    split_count = int(row.get("profiling_count", row.get("split_count")))
                    task_index = int(row.get("task_index"))
                    lr_slope = float(row.get("lr_slope"))
                except (TypeError, ValueError):
                    continue
                grouped.setdefault(idx, {"schedulable": sched, "rows": []})
                grouped[idx]["rows"].append(
                    {
                        "split_count": split_count,
                        "task_index": task_index,
                        "lr_slope": lr_slope,
                    }
                )

        for idx, info in sorted(grouped.items(), key=lambda kv: kv[0]):
            sched = info["schedulable"]
            sched_tag = "sched" if sched else "unsched"
            subset_dir = out_dir / ("schedulable" if sched else "unschedulable")
            out_png = subset_dir / f"u{util_tag}_lr_n{n_val}_idx{idx}_{sched_tag}.png"
            ok = _plot_lr_taskset_from_logged_rows(
                info["rows"],
                out_png,
                    f"LR by last profiling count | u={util_tag} n={n_val} idx={idx} ({sched_tag})",
            )
            if ok:
                saved_count += 1

    print(f"[trace:lr] saved_plots={saved_count}")
    if saved_count > 0:
        print(f"[trace:lr] output_root={out_dir}")
    return saved_count


def run_lr_trace(trace_root: Path, subset: str, metric: str, n_list_raw: str, out_dir: Path, task_set_idx: int | None, max_workers: int | None):
    if not trace_root.exists():
        raise FileNotFoundError(f"trace root not found: {trace_root}")

    n_list = _lr_parse_n_list(n_list_raw)

    idx_filter = None
    if task_set_idx is not None:
        idx_filter = {task_set_idx}

    targets = []
    for sub, csv_path in _lr_iter_target_csvs(trace_root, subset):
        idx = _lr_extract_taskset_idx(csv_path)
        if idx_filter is not None and idx not in idx_filter:
            continue
        targets.append((sub, csv_path))

    if not targets:
        print("No target CSV files found.")
        return

    workers = max_workers if max_workers and max_workers > 0 else None
    saved = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for sub, csv_path in targets:
            save_subdir = out_dir / sub
            futures.append(
                ex.submit(
                    _lr_plot_single_csv,
                    csv_path,
                    sub,
                    save_subdir,
                    metric,
                    n_list,
                )
            )

        for fut in as_completed(futures):
            out_path = fut.result()
            if out_path is not None:
                saved.append(out_path)

    print(f"[trace:lr] saved_plots={len(saved)}")
    if saved:
        print(f"[trace:lr] output_root={out_dir}")


def _find_trace_source_root(run_dir: Path, utilization: float) -> Path | None:
    candidates = sorted(run_dir.glob("trace_source_u*"))
    if not candidates:
        return None
    util_tag = _util_tag(utilization)
    exact = run_dir / f"trace_source_u{util_tag}"
    if exact.exists():
        return exact
    best = None
    best_diff = None
    for p in candidates:
        m = re.search(r"trace_source_u(\d+(?:\.\d+)?)$", p.name)
        if not m:
            continue
        u = float(m.group(1))
        d = abs(u - utilization)
        if best is None or d < best_diff:
            best = p
            best_diff = d
    return best


# --------------------------
# Unified entrypoint
# --------------------------


def main():
    parser = argparse.ArgumentParser(description="Unified trace runner for TOL_MAX and LR-by-n")
    parser.add_argument("--mode", default="tol_max", choices=["tol_max", "r_best", "lr"])
    parser.add_argument("--run-dir", default=None, help="required for mode=tol_max, optional source for mode=lr")
    parser.add_argument("--utilization", type=float, default=0.2)
    parser.add_argument("--trace-root", default=None, help="mode=lr source trace root (optional if --run-dir is given)")
    parser.add_argument("--output-root", default="trace")
    parser.add_argument("--tol-output-dir", default=None)
    parser.add_argument("--rbest-output-dir", default=None)
    parser.add_argument("--lr-output-dir", default=None)
    parser.add_argument("--tol-method", default=None, help="mode=tol_max method override")
    parser.add_argument("--task-set-idx", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--subset", default="unsched", choices=["unsched", "sched", "all"])
    parser.add_argument("--metric", default="R", choices=["R", "B", "B_low", "B_high", "tolerance_i"])
    parser.add_argument("--n-list", default="10")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    util_text = _util_tag(float(args.utilization))
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "tol_max":
        if not args.run_dir:
            raise ValueError("--run-dir is required for --mode tol_max")
        tol_output = (
            Path(args.tol_output_dir).expanduser().resolve()
            if args.tol_output_dir
            else output_root / f"trace_tol_max_{ts}_u{util_text}"
        )
        run_tol_trace(
            run_dir=Path(args.run_dir).expanduser().resolve(),
            utilization=float(args.utilization),
            output_dir=tol_output,
            task_set_idx=args.task_set_idx,
            max_workers=args.max_workers,
            method_name=args.tol_method,
        )

    if args.mode == "r_best":
        if not args.run_dir:
            raise ValueError("--run-dir is required for --mode r_best")
        rbest_output = (
            Path(args.rbest_output_dir).expanduser().resolve()
            if args.rbest_output_dir
            else output_root / f"trace_r_best_{ts}_u{util_text}"
        )
        run_r_best_trace(
            run_dir=Path(args.run_dir).expanduser().resolve(),
            utilization=float(args.utilization),
            output_dir=rbest_output,
            task_set_idx=args.task_set_idx,
            method_name=args.tol_method,
        )

    if args.mode == "lr":
        lr_output = (
            Path(args.lr_output_dir).expanduser().resolve()
            if args.lr_output_dir
            else output_root / f"trace_lr_by_n_{ts}_u{util_text}"
        )
        # mode=lr:
        # 1) prefer simulation-logged LR CSVs in run-dir (uX_lr_nX.csv)
        # 2) fallback to legacy trace-root/slice-trace based flow
        if args.run_dir:
            run_dir_path = Path(args.run_dir).expanduser().resolve()
            saved_count = run_lr_trace_from_logged_csv(
                run_dir=run_dir_path,
                utilization=float(args.utilization),
                subset=args.subset,
                n_list_raw=args.n_list,
                out_dir=lr_output,
                task_set_idx=args.task_set_idx,
            )
            if saved_count == 0 and args.trace_root:
                run_lr_trace(
                    trace_root=Path(args.trace_root).expanduser().resolve(),
                    subset=args.subset,
                    metric=args.metric,
                    n_list_raw=args.n_list,
                    out_dir=lr_output,
                    task_set_idx=args.task_set_idx,
                    max_workers=args.max_workers,
                )
        elif args.trace_root:
            run_lr_trace(
                trace_root=Path(args.trace_root).expanduser().resolve(),
                subset=args.subset,
                metric=args.metric,
                n_list_raw=args.n_list,
                out_dir=lr_output,
                task_set_idx=args.task_set_idx,
                max_workers=args.max_workers,
            )
        else:
            raise ValueError("--mode lr requires --run-dir (preferred) or --trace-root")


if __name__ == "__main__":
    main()
