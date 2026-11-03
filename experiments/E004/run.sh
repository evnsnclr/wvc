#!/usr/bin/env bash
# E004 (experiments/E004-em-bisect.md): training, per-step merged commits, and the judge sweep.
# The bisect is not part of this script. Run from anywhere:  bash experiments/E004/run.sh
# Stages: fetch models -> data -> train -> merge and commit -> sweep; receipt.py records the
# start and, on exit (also after a failure), writes results/receipt.json.
export WVC_AUTHOR="${WVC_AUTHOR:-evansmith}"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
E=experiments/E004
export HF_HUB_DISABLE_TELEMETRY=1

# The package pins every script runs with (keep in sync with UV_PYTHON in e004_config.py).
PY=(uv run --with torch==2.15.0 --with transformers==5.22.1 --with peft==0.21.2
    --with safetensors==0.8.0 --with accelerate==1.15.0 python)

# The plan's stop rule: abort when less than 20 GB is free.
check_disk() {
  local free_kb
  free_kb=$(df -k /System/Volumes/Data | awk 'NR == 2 {print $4}')
  echo "free disk before $1: ${free_kb} KB"
  if (( free_kb < 20 * 1024 * 1024 )); then
    echo "abort: under 20 GB free before $1" >&2
    exit 3
  fi
}

finish() {
  local status=$?
  "${PY[@]}" "$E/receipt.py" finish --exit-status "$status" || true
  exit "$status"
}

uv sync --locked
check_disk "the run"
"${PY[@]}" "$E/receipt.py" start
trap finish EXIT

check_disk "downloads"
"${PY[@]}" "$E/fetch_models.py"
"${PY[@]}" "$E/data.py"

check_disk "training"
rm -rf "$E/checkpoints"
"${PY[@]}" "$E/train.py"

check_disk "merging and committing"
rm -rf "$E/merged"
"${PY[@]}" "$E/merge_and_commit.py" --fresh

check_disk "the sweep"
rm -rf "$E/work" "$E/results/labels" "$E/raw/generations"
"${PY[@]}" "$E/sweep.py"
