#!/usr/bin/env bash
# Poll AE training status on molab (checkpoint + SQLite + bg).
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION (prefer unset unless multi-session)
#
# Usage:
#   bash .github/skills/molab-workflow/scripts/molab-train-status.sh
#   bash .github/skills/molab-workflow/scripts/molab-train-status.sh --json
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
molab_exec="$script_dir/molab-exec.sh"
as_json=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      as_json=1
      shift
      ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-train-status: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

"$molab_exec" <<PY
import json
import sys

sys.path.insert(0, "/marimo")
from molab.jobs import (
    AE_BIDIR_FLOW_BG_STEM,
    AE_BIDIR_FLOW_CHECKPOINT,
    AE_BIDIR_FLOW_LOG,
)
from lpap.training_status import format_training_status, summarize_training_status

summary = summarize_training_status(
    project_root="/marimo",
    checkpoint_name=AE_BIDIR_FLOW_CHECKPOINT,
    log_name=AE_BIDIR_FLOW_LOG,
    bg_stem=AE_BIDIR_FLOW_BG_STEM,
)
if ${as_json} == 1:
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
else:
    print(format_training_status(summary))
PY
