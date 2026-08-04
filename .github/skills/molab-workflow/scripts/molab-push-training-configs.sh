#!/usr/bin/env bash
# Copy repo configs/training/*.toml into the paired molab /marimo tree.
#
# Used by molab-sync and molab-launch-image-energy-flow so TOML edits on the
# agent host reach the worker without a pip reinstall of lpap.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
molab_exec="$script_dir/molab-exec.sh"
training_dir="$repo_root/configs/training"

if [[ ! -d "$training_dir" ]]; then
  echo "molab-push-training-configs: missing $training_dir" >&2
  exit 1
fi

payload="$(
  python3 - <<PY
import base64
import io
import tarfile
from pathlib import Path

root = Path("$training_dir")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for path in sorted(root.glob("*.toml")):
        tar.add(path, arcname=str(Path("configs/training") / path.name))
print(base64.b64encode(buf.getvalue()).decode("ascii"))
PY
)"

"$molab_exec" <<PY
import base64
import io
import tarfile
from pathlib import Path

raw = base64.b64decode("${payload}")
dest = Path("/marimo")
(dest / "configs" / "training").mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
    tar.extractall(dest)
files = sorted(p.name for p in (dest / "configs" / "training").glob("*.toml"))
print("training configs", dest / "configs" / "training", "files", files)
PY
