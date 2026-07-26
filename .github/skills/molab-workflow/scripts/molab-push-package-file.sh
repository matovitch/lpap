#!/usr/bin/env bash
# Hot-patch one local src/lpap/*.py into the paired molab kernel site-packages.
#
# Use for short “try before commit” probes. Durable path remains:
#   commit + push + molab-sync.sh
# Sync / force-reinstall will overwrite this patch.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION
#
# Usage:
#   bash …/molab-push-package-file.sh src/lpap/training_plots.py
#   bash …/molab-push-package-file.sh training_plots.py
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
molab_exec="$script_dir/molab-exec.sh"
lib_py="$script_dir/molab_lib.py"

path_arg=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    -*)
      echo "molab-push-package-file: unknown arg: $1" >&2
      exit 1
      ;;
    *)
      if [[ -n "$path_arg" ]]; then
        echo "molab-push-package-file: pass exactly one path" >&2
        exit 1
      fi
      path_arg="$1"
      shift
      ;;
  esac
done

if [[ -z "$path_arg" ]]; then
  echo "Usage: molab-push-package-file.sh src/lpap/<module>.py" >&2
  exit 1
fi

if [[ ! -x "$molab_exec" ]]; then
  echo "molab-push-package-file: missing $molab_exec" >&2
  exit 1
fi

meta="$(
  PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" python3 - <<PY
import json
from pathlib import Path
from molab_lib import repo_root_from_script, resolve_lpap_package_file

root = repo_root_from_script(Path(r"$lib_py"))
path, module, rel = resolve_lpap_package_file(r"$path_arg", repo_root=root)
print(json.dumps({"path": str(path), "module": module, "rel": str(rel)}))
PY
)"

local_path="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["path"])' "$meta")"
module_name="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["module"])' "$meta")"
rel_path="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["rel"])' "$meta")"

payload_b64="$(
  python3 - <<PY
import base64
from pathlib import Path
print(base64.b64encode(Path(r"$local_path").read_bytes()).decode("ascii"))
PY
)"

echo "molab-push-package-file: $local_path → kernel $module_name ($rel_path)"

"$molab_exec" <<PY
import base64
import importlib
import sys
from pathlib import Path

module_name = $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$module_name")
rel = Path($(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$rel_path"))
payload = base64.b64decode($(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$payload_b64"))

import lpap

package_root = Path(lpap.__file__).resolve().parent
dest = package_root / rel
if not dest.parent.is_dir():
    raise SystemExit(f"destination parent missing: {dest.parent}")
dest.write_bytes(payload)

for name in list(sys.modules):
    if name == module_name or name.startswith(module_name + "."):
        del sys.modules[name]
# Also drop bare package cache so ``import lpap.X`` re-binds cleanly.
if "lpap" in sys.modules and module_name != "lpap":
    # Keep ``lpap`` itself; only refresh the target module.
    pass
mod = importlib.import_module(module_name)
print(
    "patched",
    module_name,
    "->",
    getattr(mod, "__file__", dest),
    "bytes",
    dest.stat().st_size,
    flush=True,
)
print(
    "note: notebook ``from … import`` bindings stay stale until you re-run the owning cell",
    flush=True,
)
PY
