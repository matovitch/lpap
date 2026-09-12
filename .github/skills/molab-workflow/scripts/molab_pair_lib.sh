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
