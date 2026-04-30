#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/overnight"
GEN_YAML="$SCRIPT_DIR/generate_task_set.yaml"
SIM_YAML="$SCRIPT_DIR/simulation.yaml"
GEN_PY="$SCRIPT_DIR/generate_task_set.py"
SIM_PY="$SCRIPT_DIR/simulation.py"
TRACE_PY="$SCRIPT_DIR/trace.py"
RESULT_DIR="$SCRIPT_DIR/result"
TRACE_DIR="$SCRIPT_DIR/trace"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$SCRIPT_DIR/overnight_runs/${RUN_TS}_from_configs"
mkdir -p "$RUN_ROOT" "$TRACE_DIR"

GEN_BAK="$RUN_ROOT/generate_task_set.yaml.bak"
SIM_BAK="$RUN_ROOT/simulation.yaml.bak"
cp "$GEN_YAML" "$GEN_BAK"
cp "$SIM_YAML" "$SIM_BAK"

restore_configs() {
  cp "$GEN_BAK" "$GEN_YAML"
  cp "$SIM_BAK" "$SIM_YAML"
}
trap restore_configs EXIT

set_yaml_value() {
  local yaml_path="$1"
  local key="$2"
  local value="$3"
  python3 - "$yaml_path" "$key" "$value" <<'PY'
import sys
import yaml

path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r") as f:
    data = yaml.safe_load(f) or {}

parsed = value
if value.lower() in {"true", "false"}:
    parsed = (value.lower() == "true")
else:
    try:
        parsed = int(value)
    except ValueError:
        try:
            parsed = float(value)
        except ValueError:
            parsed = value

data[key] = parsed
with open(path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
}

delete_yaml_key() {
  local yaml_path="$1"
  local key="$2"
  python3 - "$yaml_path" "$key" <<'PY'
import sys
import yaml

path, key = sys.argv[1], sys.argv[2]
with open(path, "r") as f:
    data = yaml.safe_load(f) or {}
if isinstance(data, dict) and key in data:
    data.pop(key, None)
with open(path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
}

get_yaml_value() {
  local yaml_path="$1"
  local key="$2"
  python3 - "$yaml_path" "$key" <<'PY'
import sys
import yaml

path, key = sys.argv[1], sys.argv[2]
with open(path, "r") as f:
    data = yaml.safe_load(f) or {}
value = data.get(key, "")
if value is None:
    value = ""
print(value)
PY
}

latest_result_dir() {
  if ls -1d "$RESULT_DIR"/* >/dev/null 2>&1; then
    ls -dt "$RESULT_DIR"/* | head -n 1
  else
    return 1
  fi
}

run_dir_from_sim_log() {
  local log_path="$1"
  awk -F': ' '/^\[test\] mismatch logs directory: /{print $2}' "$log_path" | tail -n 1
}

simulation_status() {
  local run_dir="$1"
  local runtime_file="$run_dir/simulation_runtime.txt"
  if [[ ! -f "$runtime_file" ]]; then
    echo "unknown"
    return
  fi
  awk -F': ' '/^status:/{print $2}' "$runtime_file" | tail -n 1
}

list_utils_from_run_dir() {
  local run_dir="$1"
  find "$run_dir/rta_logs" -maxdepth 1 -type f -name "rta_task_set_list_u*.log" \
    | sed -E 's#.*_u([0-9]+(\.[0-9]+)?)\.log#\1#' \
    | sort -n
}

run_rbest_trace_for_utils() {
  local run_dir="$1"
  local label="$2"
  local trace_log="$RUN_ROOT/${label}_trace_rbest.log"
  local out_root="$TRACE_DIR/$(basename "$run_dir")"
  local rbest_method="RTA_SS_tol_fb_rbest"

  local util=""
  while IFS= read -r util; do
    [[ -z "$util" ]] && continue
    echo "[trace:r_best] case=$label util=$util run=$(basename "$run_dir")"
    python3 "$TRACE_PY" \
      --mode r_best \
      --run-dir "$run_dir" \
      --utilization "$util" \
      --tol-method "$rbest_method" \
      --output-root "$out_root" \
      >> "$trace_log" 2>&1
  done < <(list_utils_from_run_dir "$run_dir")
}

run_case() {
  local cfg_path="$1"
  local label
  label="$(basename "$cfg_path" .yaml)"
  local task_set_dir
  task_set_dir="$(get_yaml_value "$cfg_path" "output_dir")"

  local generate_log="$RUN_ROOT/${label}_generate.log"
  local sim_log="$RUN_ROOT/${label}_simulation.log"
  local sim_resume_log="$RUN_ROOT/${label}_simulation_resume1.log"
  local result_path_file="$RUN_ROOT/${label}_result_dir.txt"

  echo "============================================================"
  echo "[case] $label"
  echo "============================================================"

  cp "$cfg_path" "$GEN_YAML"
  delete_yaml_key "$SIM_YAML" "resume_run_log_dir"
  delete_yaml_key "$SIM_YAML" "replot_only_run_log_dir"
  set_yaml_value "$SIM_YAML" "result_dir_prefix" "${label}"
  set_yaml_value "$SIM_YAML" "enable_RTA_SS_tol_fb_rbest" "true"
  if [[ -n "$task_set_dir" ]]; then
    set_yaml_value "$SIM_YAML" "task_set_list_dir_path" "$task_set_dir"
  fi

  echo "[generate] config=$(basename "$cfg_path")"
  python3 "$GEN_PY" > "$generate_log" 2>&1

  local before_latest=""
  before_latest="$(latest_result_dir || true)"

  echo "[simulate] first attempt"
  set +e
  python3 -u "$SIM_PY" 2>&1 | tee "$sim_log"
  local sim_rc=${PIPESTATUS[0]}
  set -e

  local run_dir=""
  run_dir="$(run_dir_from_sim_log "$sim_log" || true)"
  if [[ -z "$run_dir" ]]; then
    run_dir="$(latest_result_dir || true)"
  fi

  if [[ -n "$before_latest" && -n "$run_dir" && "$run_dir" == "$before_latest" ]]; then
    echo "[warn] result dir unchanged after simulation: $run_dir"
  fi

  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    echo "[error] could not determine run_dir for case=$label"
    echo "[error] simulation rc=$sim_rc"
    echo "[case:$label] failed (no run_dir)"
    return 1
  fi

  local status=""
  status="$(simulation_status "$run_dir")"
  echo "[simulate] first attempt rc=$sim_rc status=$status run_dir=$run_dir task_set_list_dir_path=${task_set_dir:-<unchanged>}"

  if [[ "$status" != "completed" ]]; then
    echo "[simulate] resume attempt (max 1)"
    set_yaml_value "$SIM_YAML" "resume_run_log_dir" "$run_dir"
    set +e
    python3 -u "$SIM_PY" 2>&1 | tee "$sim_resume_log"
    local resume_rc=${PIPESTATUS[0]}
    set -e
    delete_yaml_key "$SIM_YAML" "resume_run_log_dir"
    status="$(simulation_status "$run_dir")"
    echo "[simulate] resume rc=$resume_rc status=$status run_dir=$run_dir"
  fi

  echo "$run_dir" > "$result_path_file"

  if [[ "$status" == "completed" ]]; then
    echo "[trace:r_best] run_dir=$run_dir"
    run_rbest_trace_for_utils "$run_dir" "$label"
    echo "[case:$label] done"
    echo "  generate log      : $generate_log"
    echo "  simulation log    : $sim_log"
    echo "  simulation resume : $sim_resume_log"
    echo "  r_best trace log  : $RUN_ROOT/${label}_trace_rbest.log"
    echo "  result dir        : $run_dir"
    return 0
  fi

  echo "[case:$label] failed (status=$status)"
  echo "  generate log      : $generate_log"
  echo "  simulation log    : $sim_log"
  echo "  simulation resume : $sim_resume_log"
  echo "  result dir        : $run_dir"
  return 1
}

main() {
  local fail_count=0
  local cfg=""

  while IFS= read -r cfg; do
    [[ -z "$cfg" ]] && continue
    if ! run_case "$cfg"; then
      fail_count=$((fail_count + 1))
    fi
  done < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.yaml" | sort -V)

  echo "============================================================"
  if [[ $fail_count -eq 0 ]]; then
    echo "All cases completed successfully."
  else
    echo "Completed with failures: $fail_count case(s) failed."
  fi
  echo "Run logs root : $RUN_ROOT"
  echo "Trace root    : $TRACE_DIR"
  echo "============================================================"
}

main "$@"
