#!/usr/bin/env bash
# Export AE loss, best-checkpoint gallery, and teachers PNGs from a molab run.
#
# Safe while training: does not call molab-exec or otherwise touch the kernel.
# Fresh SQLite logs are pulled through the marimo file API. The best AE
# checkpoint is downloaded from the Hugging Face dual-slot pointer — the
# 600 MB .pt is too large for file_details to be reliable (jq/timeouts
# truncated it). Rendering runs locally on CPU.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# HF token: HF_TOKEN, or LPAP_SECRETS_TOML / configs/secrets.toml / SECRETS_TOML
# Python: LPAP_PYTHON if set, otherwise python3 (needs torch, matplotlib, lpap)
#
# Usage:
#   bash .github/skills/molab-workflow/scripts/molab-export-pics.sh \
#     --suffix -v3 --output-dir ./lpap_viz --project-root /path/to/cache
#   --run v3 is the same as --suffix -v3
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,20p' "$0" | sed 's/^# \?//'
  exit 0
fi

py="${LPAP_PYTHON:-python3}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$py" "$script_dir/molab_export_pics.py" "$@"
