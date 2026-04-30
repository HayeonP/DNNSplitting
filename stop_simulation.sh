#!/usr/bin/env bash
set -euo pipefail

TARGET_SCRIPTS=(
  "simulation.py"
)

collect_pids() {
  local script="$1"
  local pattern="python(3)? .*(^|[ /])${script}( |$)"
  pgrep -f "$pattern" || true
}

mapfile -t PIDS < <(
  {
    for script in "${TARGET_SCRIPTS[@]}"; do
      collect_pids "$script"
    done
  } | sort -n -u
)

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No running simulation process found."
  exit 0
fi

echo "Stopping simulation processes: ${PIDS[*]}"
kill "${PIDS[@]}" 2>/dev/null || true

for _ in {1..10}; do
  sleep 0.5
  STILL=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      STILL+=("$pid")
    fi
  done
  if [ "${#STILL[@]}" -eq 0 ]; then
    echo "All simulation processes stopped gracefully."
    exit 0
  fi
done

echo "Force killing remaining PIDs: ${STILL[*]}"
kill -9 "${STILL[@]}" 2>/dev/null || true

echo "Done."
