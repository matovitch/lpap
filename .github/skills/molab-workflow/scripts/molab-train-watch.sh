#!/usr/bin/env bash
# Read the notebook train cell's heartbeat without touching the kernel.
#
# The train cell writes /marimo/training_logs/notebook_train_status.json
# (state, stage, steps, GPU util, VRAM, heartbeat). This reads it through the
# marimo server's file API, so it is safe while training runs: a scratch exec
# (molab-exec) queues behind the busy cell, and abandoning it interrupts the run.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Usage: molab-train-watch.sh [--json]
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=molab_pair_lib.sh
source "$script_dir/molab_pair_lib.sh"

status="$(molab_read_server_file /marimo/training_logs/notebook_train_status.json)" || {
  echo "molab-train-watch: no train status yet (training never started this session?)" >&2
  exit 1
}
if [[ "${1:-}" == "--json" ]]; then
  echo "$status"
  exit 0
fi
jq -r '
  "state=\(.state) suffix=\(.run_suffix) stage=\(.stage) gpu=\(.gpu_util_pct // "?")% vram=\(.vram_mib // "?")MiB",
  "started=\(.started | strflocaltime("%Y-%m-%d %H:%M:%S %Z")) heartbeat_age=\((now - .heartbeat) | floor)s",
  (.runs | to_entries[] | "  \(.key) step=\(.value.step) loss=\(.value.loss) val=\(.value.val // "n/a") best=\(.value.best)")
' <<<"$status"
