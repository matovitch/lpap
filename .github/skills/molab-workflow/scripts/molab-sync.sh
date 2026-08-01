#!/usr/bin/env bash
# Sync local molab-summer package + configs into the paired molab kernel.
#
# Steps:
#   1. Force-reinstall lpap from git (default ref: molab-summer)
#   2. Copy configs/storage.toml → /marimo/configs/storage.toml
#   3. Copy repo molab/ helpers → /marimo/molab/ (not part of lpap)
#   4. Inject configs/secrets.toml → kernel env (molab-inject-secrets.sh)
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION (prefer unset unless the server has multiple sessions)
#
# Optional env / flags:
#   LPAP_GIT_REF=molab-summer     git ref / branch / SHA to install
#   --ref REF                     same as LPAP_GIT_REF for this call
#   --skip-install                skip pip reinstall
#   --skip-secrets                skip inject
#   --skip-storage                skip storage.toml copy
#   --skip-helpers                skip molab/ helper copy
#
# Usage:
#   bash .github/skills/molab-workflow/scripts/molab-sync.sh
#   bash .github/skills/molab-workflow/scripts/molab-sync.sh --ref abd69b3
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
molab_exec="$script_dir/molab-exec.sh"
inject_secrets="$script_dir/molab-inject-secrets.sh"
storage_toml="$repo_root/configs/storage.toml"
molab_helpers="$repo_root/molab"
git_ref="${LPAP_GIT_REF:-molab-summer}"
do_install=1
do_secrets=1
do_storage=1
do_helpers=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      git_ref="$2"
      shift 2
      ;;
    --skip-install)
      do_install=0
      shift
      ;;
    --skip-secrets)
      do_secrets=0
      shift
      ;;
    --skip-storage)
      do_storage=0
      shift
      ;;
    --skip-helpers)
      do_helpers=0
      shift
      ;;
    -h|--help)
      sed -n '2,26p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-sync: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$molab_exec" ]]; then
  echo "molab-sync: molab-exec.sh not executable: $molab_exec" >&2
  exit 1
fi

if [[ "$do_install" -eq 1 ]]; then
  echo "molab-sync: installing lpap @ ${git_ref} ..."
  # Embed ref: molab-exec runs on the remote kernel (no local env forward).
  "$molab_exec" <<PY
import subprocess, sys
ref = $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$git_ref")
spec = f"lpap @ git+https://github.com/matovitch/lpap.git@{ref}"
print("pip install --no-deps --force-reinstall", spec, flush=True)
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", spec],
    check=False,
)
print("pip_rc", r.returncode, flush=True)
if r.returncode != 0:
    raise SystemExit(r.returncode)
# lpap installs with --no-deps; pull runtime extras the molab image may lack.
for spec in ("jaxtyping>=0.3.7", "zstandard>=0.25.0"):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", spec],
        check=False,
    )
for name in list(sys.modules):
    if name == "lpap" or name.startswith("lpap."):
        del sys.modules[name]
import lpap
print("lpap", getattr(lpap, "__file__", "?"), flush=True)
PY
fi

if [[ "$do_storage" -eq 1 ]]; then
  if [[ ! -f "$storage_toml" ]]; then
    echo "molab-sync: missing $storage_toml" >&2
    exit 1
  fi
  echo "molab-sync: copying configs/storage.toml → /marimo/configs/storage.toml"
  payload="$(python3 - <<PY
import base64
from pathlib import Path
raw = Path("$storage_toml").read_bytes()
print(base64.b64encode(raw).decode("ascii"))
PY
)"
  "$molab_exec" -c "
import base64
from pathlib import Path
path = Path('/marimo/configs/storage.toml')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(base64.b64decode('${payload}'))
print('wrote', path, 'bytes', path.stat().st_size)
"
fi

if [[ "$do_helpers" -eq 1 ]]; then
  if [[ ! -d "$molab_helpers" ]]; then
    echo "molab-sync: missing $molab_helpers" >&2
    exit 1
  fi
  echo "molab-sync: copying molab/ helpers → /marimo/molab/"
  helpers_b64="$(
    python3 - <<PY
import base64
import io
import tarfile
from pathlib import Path

root = Path("$molab_helpers")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix == ".py":
            tar.add(path, arcname=str(Path("molab") / path.relative_to(root)))
print(base64.b64encode(buf.getvalue()).decode("ascii"))
PY
  )"
  "$molab_exec" <<PY
import base64
import io
import tarfile
from pathlib import Path

raw = base64.b64decode("${helpers_b64}")
dest = Path("/marimo")
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
    tar.extractall(dest)
# Drop stale modules so the next import picks up the synced files.
import sys
for name in list(sys.modules):
    if name == "molab" or name.startswith("molab."):
        del sys.modules[name]
sys.path.insert(0, "/marimo")
import molab
print("molab helpers", Path("/marimo/molab").resolve(), "files", sorted(p.name for p in Path("/marimo/molab").glob("*.py")))
PY
fi

if [[ "$do_secrets" -eq 1 ]]; then
  echo "molab-sync: injecting secrets.toml → kernel env"
  bash "$inject_secrets"
fi

echo "molab-sync: done"
