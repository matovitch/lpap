# Shared host helpers for scripts that call marimo-pair execute-code.sh.
# Source this file; do not execute it.
#
# Env:
#   MARIMO_PAIR_SCRIPTS  marimo-pair scripts dir
#                        (default: ~/.cursor/skills/marimo-pair/scripts)
#   MOLAB_PAIR_CALLER    prefix for error messages (optional)
#
# After molab_require_execute_code: MOLAB_EXECUTE_CODE is the absolute path.

molab_require_execute_code() {
  local scripts_dir="${MARIMO_PAIR_SCRIPTS:-$HOME/.cursor/skills/marimo-pair/scripts}"
  local caller="${MOLAB_PAIR_CALLER:-molab}"
  MOLAB_EXECUTE_CODE="$scripts_dir/execute-code.sh"
  if [[ ! -x "$MOLAB_EXECUTE_CODE" ]]; then
    echo "${caller}: execute-code.sh not found or not executable: $MOLAB_EXECUTE_CODE" >&2
    echo "Install marimo-pair with: npx skills add marimo-team/marimo-pair" >&2
    echo "Set MARIMO_PAIR_SCRIPTS to that skill's scripts directory if needed." >&2
    return 1
  fi
  return 0
}

# Print a file of the molab box through the marimo server file API
# (POST /api/files/file_details). Runs in the server process, not the kernel,
# so it never queues behind a busy cell. Needs MOLAB_URL + MOLAB_TOKEN/MARIMO_TOKEN.
molab_read_server_file() {
  local path="$1"
  local base="${MOLAB_URL%/}"
  local token="${MOLAB_TOKEN:-${MARIMO_TOKEN:-}}"
  local resp
  resp="$(
    curl -sf -m 15 -X POST "${base}/api/files/file_details" \
      -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
      -d "$(jq -n --arg p "$path" '{path: $p}')"
  )" || return 1
  jq -er '.contents // empty' <<<"$resp"
}

# Refuse a scratch exec while the notebook train cell reports "running": the
# exec would queue behind the cell, and abandoning it interrupts the run.
# Heartbeats older than MOLAB_BUSY_STALE_S (default 1800 s) count as dead.
# Override with MOLAB_EXEC_FORCE=1.
molab_guard_busy_kernel() {
  [[ "${MOLAB_EXEC_FORCE:-0}" == "1" ]] && return 0
  local status age
  status="$(molab_read_server_file /marimo/training_logs/notebook_train_status.json 2>/dev/null)" || return 0
  [[ "$(jq -r '.state' <<<"$status")" == "running" ]] || return 0
  age="$(jq -r 'now - .heartbeat | floor' <<<"$status")"
  (( age < ${MOLAB_BUSY_STALE_S:-1800} )) || return 0
  echo "${MOLAB_PAIR_CALLER:-molab}: the notebook train cell is running (heartbeat ${age}s ago)." >&2
  echo "A scratch exec would queue behind it, and abandoning it interrupts training." >&2
  echo "Monitor with molab-train-watch.sh; set MOLAB_EXEC_FORCE=1 only to stop the run on purpose." >&2
  return 1
}
