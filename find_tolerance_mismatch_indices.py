#!/usr/bin/env python3
"""Find task_set indices where RTA_SS_max is schedulable but
RTA_SS_tol is not schedulable, and tolerance failure happened
before the final task in the task set.

Usage:
  python3 simulation/find_tolerance_mismatch_indices.py <run_dir_or_rta_logs_dir>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TASKSET_RE = re.compile(r"^\[task_set_index=(\d+)\]")
METHOD_RE = re.compile(r"^<([^>]+)>\s+schedulable=(True|False)")
TASK_ORDER_RE = re.compile(r"task\(order=(\d+)\s")


def _resolve_rta_logs_dirs(path: Path):
    dirs = []
    if path.name == "rta_logs" and path.is_dir():
        dirs.append(path)
        return dirs

    direct = path / "rta_logs"
    if direct.is_dir():
        dirs.append(direct)

    # Also support selecting a parent folder (e.g., result/) containing many runs.
    for child in sorted(path.glob("*/rta_logs")):
        if child.is_dir() and child not in dirs:
            dirs.append(child)

    return dirs


def _parse_one_log(log_path: Path):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    results = []

    current_idx = None
    max_sched = None
    tol_sched = None
    current_method = None
    in_tol_rdb = False
    tol_rdb_last_idx = None
    tol_rdb_last_sched = None
    max_task_order = None

    def flush_block():
        if current_idx is None:
            return
        if (
            max_sched is True
            and tol_sched is False
            and tol_rdb_last_idx is not None
            and tol_rdb_last_sched == "N"
            and max_task_order is not None
            and tol_rdb_last_idx < max_task_order
        ):
            results.append(current_idx)

    for raw in lines:
        line = raw.strip()

        m_taskset = TASKSET_RE.match(line)
        if m_taskset:
            flush_block()
            current_idx = int(m_taskset.group(1))
            max_sched = None
            tol_sched = None
            current_method = None
            in_tol_rdb = False
            tol_rdb_last_idx = None
            tol_rdb_last_sched = None
            max_task_order = None
            continue

        m_method = METHOD_RE.match(line)
        if m_method:
            current_method = m_method.group(1)
            sched = (m_method.group(2) == "True")
            in_tol_rdb = False
            if current_method == "RTA_SS_max":
                max_sched = sched
            elif current_method == "RTA_SS_tol":
                tol_sched = sched
            continue

        if line.startswith("RDB:"):
            in_tol_rdb = (current_method == "RTA_SS_tol")
            continue

        m_order = TASK_ORDER_RE.search(line)
        if m_order:
            order = int(m_order.group(1))
            if max_task_order is None or order > max_task_order:
                max_task_order = order

        if in_tol_rdb:
            if not line:
                in_tol_rdb = False
                continue
            if line.startswith("<") or line.startswith("[task_set_index="):
                in_tol_rdb = False
                continue
            parts = line.split()
            # RDB row format: idx id R D B ... sched
            if len(parts) >= 2 and parts[0].isdigit() and parts[-1] in {"Y", "N"}:
                tol_rdb_last_idx = int(parts[0])
                tol_rdb_last_sched = parts[-1]

    flush_block()
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find task_set_index where RTA_SS_max=True, "
            "RTA_SS_tol=False, and tolerance fails before final task"
        )
    )
    parser.add_argument("path", help="run directory or rta_logs directory")
    args = parser.parse_args()

    input_path = Path(args.path).expanduser().resolve()
    rta_logs_dirs = _resolve_rta_logs_dirs(input_path)
    if not rta_logs_dirs:
        raise FileNotFoundError(f"No rta_logs directory found under: {input_path}")

    any_found = False
    any_logs = False
    for rta_logs_dir in rta_logs_dirs:
        log_paths = sorted(rta_logs_dir.glob("rta_*.log"))
        if not log_paths:
            continue
        any_logs = True
        print(f"# {rta_logs_dir}")
        for log_path in log_paths:
            indices = _parse_one_log(log_path)
            if not indices:
                continue
            any_found = True
            print(f"{log_path.name}: {', '.join(str(i) for i in indices)}")

    if not any_logs:
        print(f"No rta_*.log files found under: {input_path}")
    elif not any_found:
        print("No matching task_set_index found.")


if __name__ == "__main__":
    main()
