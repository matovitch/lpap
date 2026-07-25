#!/usr/bin/env bash
# Inject local configs/secrets.toml into the paired molab kernel.
#
# Reads secrets on the agent host, sets os.environ in the remote kernel without
# printing secret values. When huggingface.token is set, also writes
# /marimo/.hf_token (mode 600) for artifact_sync token_files.
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
#   LPAP_SECRETS_TOML=~/.config/lpap/secrets.toml bash .../molab-inject-secrets.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
secrets_path="${LPAP_SECRETS_TOML:-$repo_root/configs/secrets.toml}"
molab_exec="$script_dir/molab-exec.sh"

if [[ ! -f "$secrets_path" ]]; then
  echo "molab-inject-secrets: missing $secrets_path" >&2
  echo "Copy configs/secrets.toml.example → configs/secrets.toml and fill values." >&2
  exit 1
fi
if [[ ! -x "$molab_exec" ]]; then
  echo "molab-inject-secrets: molab-exec.sh not executable: $molab_exec" >&2
  exit 1
fi

# Build a base64 JSON map of env vars on the agent host (values never echoed).
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
        # Enable TrainingRun.mark_finished → Pushover without per-run flags.
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

# Remote: decode, set env, optionally write HF token file; print keys only.
"$molab_exec" -c "
import base64, json, os
from pathlib import Path
_env = json.loads(base64.b64decode('${payload}').decode('utf-8'))
for _key, _value in _env.items():
    os.environ[_key] = _value
if 'HF_TOKEN' in _env:
    _path = Path('/marimo/.hf_token')
    _path.write_text(_env['HF_TOKEN'].rstrip() + '\n', encoding='utf-8')
    _path.chmod(0o600)
print('injected_keys', ' '.join(sorted(_env)))
print('hf_token_file', Path('/marimo/.hf_token').is_file())
"
