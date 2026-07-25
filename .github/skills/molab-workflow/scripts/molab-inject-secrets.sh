#!/usr/bin/env bash
# Inject local configs/secrets.toml into the paired molab kernel as env vars.
#
# Single source of truth: configs/secrets.toml (gitignored). Sets HF_TOKEN,
# PUSHOVER_*, LPAP_NOTIFY_ON_FINISHED in the kernel. Does not write secret
# files under /marimo/ (and removes any legacy ones).
#
# Required env (same as molab-exec): MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN,
# optional MOLAB_SESSION.
#
# Optional:
#   LPAP_SECRETS_TOML   path to secrets file
#                       (default: <repo>/configs/secrets.toml)
#
# Usage:
#   bash .github/skills/molab-workflow/scripts/molab-inject-secrets.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
secrets_path="${LPAP_SECRETS_TOML:-$repo_root/configs/secrets.toml}"
molab_exec="$script_dir/molab-exec.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-inject-secrets: unknown arg: $1 (secrets.toml → env only)" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$molab_exec" ]]; then
  echo "molab-inject-secrets: molab-exec.sh not executable: $molab_exec" >&2
  exit 1
fi
if [[ ! -f "$secrets_path" ]]; then
  echo "molab-inject-secrets: missing $secrets_path" >&2
  echo "Copy configs/secrets.toml.example → configs/secrets.toml and fill values." >&2
  exit 1
fi

payload="$(
  SECRETS_PATH="$secrets_path" python3 - <<'PY'
import base64
import json
import os
import sys
import tomllib
from pathlib import Path

path = Path(os.environ["SECRETS_PATH"])
with path.open("rb") as handle:
    data = tomllib.load(handle)

env: dict[str, str] = {}
hf = data.get("huggingface") or {}
if isinstance(hf, dict):
    token = str(hf.get("token") or "").strip()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token

pushover = data.get("pushover") or {}
if isinstance(pushover, dict):
    app_token = str(pushover.get("token") or "").strip()
    user = str(pushover.get("user") or "").strip()
    if app_token:
        env["PUSHOVER_TOKEN"] = app_token
    if user:
        env["PUSHOVER_USER"] = user
    if app_token and user:
        env["LPAP_NOTIFY_ON_FINISHED"] = "1"

if not env:
    print(
        "molab-inject-secrets: no non-empty secrets in "
        f"{path} (fill huggingface.token and/or pushover.*)",
        file=sys.stderr,
    )
    sys.exit(2)

sys.stdout.write(base64.b64encode(json.dumps(env).encode("utf-8")).decode("ascii"))
PY
)"

"$molab_exec" -c "
import base64, json, os
from pathlib import Path
_root = Path('/marimo')
_env = json.loads(base64.b64decode('${payload}').decode('utf-8'))
for _key, _value in _env.items():
    os.environ[_key] = _value
# Drop legacy secret files if any remain from earlier workflows.
_removed = []
for _name in (
    '.hf_token',
    '.pushover_token',
    '.pushover_user',
    '.lpap_notify_on_finished',
):
    _path = _root / _name
    if _path.is_file():
        _path.unlink()
        _removed.append(_name)
print('injected_keys', ' '.join(sorted(_env)))
if _removed:
    print('removed_legacy_files', ' '.join(_removed))
"
