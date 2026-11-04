#!/usr/bin/env bash
#SBATCH --job-name=wvc-census
#SBATCH --account=paceship-srdsgt26
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# E009 on PACE: the large-pace group of experiments/E009/census.py, CPU only, embers only.
# Arguments: OUT_DIR CACHE_DIR SOURCE_COMMIT [census.py arguments, default --group large-pace].
# OUT_DIR must already hold selection/ for each base.
set -euo pipefail

job_record=$(scontrol show job -o "$SLURM_JOB_ID")
case "$job_record" in
  *"Account=paceship-srdsgt26"*"QOS=embers"*) ;;
  *) printf 'JOB_ROUTE_MISMATCH|%s\n' "$job_record" >&2; exit 90 ;;
esac

out_dir=$1
cache_dir=$2
export WVC_CENSUS_SOURCE_COMMIT=$3
shift 3
if [ "$#" -eq 0 ]; then set -- --group large-pace; fi
# the submitter passes --chdir to the synced repository; SLURM_SUBMIT_DIR is the login directory

module purge
module load python/3.12.5
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$HOME/scratch/cache/pip"
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$PIP_CACHE_DIR" "$cache_dir" "$out_dir"

venv="${TMPDIR:-/tmp}/wvc-census-venv"
python3 -m venv "$venv"
. "$venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r experiments/E009/requirements-pace.txt
pip install --quiet --no-deps .
python -c 'import platform, wvc, huggingface_hub, numpy; print("python", platform.python_version(), "wvc", wvc.__version__, "hub", huggingface_hub.__version__, "numpy", numpy.__version__)'

# Gated base: fail early, before any weight is fetched, if the token on this account lacks access.
python -c 'import sys, tempfile; from huggingface_hub import hf_hub_download; [hf_hub_download(r, "config.json", cache_dir=tempfile.mkdtemp()) for r in ("google/gemma-2-9b-it", "google/gemma-3-1b-it")]; print("gated access ok")' || exit 91

echo "start $(date -u +%FT%TZ) host $(hostname)"
set +e
python -u experiments/E009/census.py --out "$out_dir" --cache "$cache_dir" \
  --host pace --max-bytes 200e9 --min-free 100e9 "$@"
rc=$?
set -e
echo "exit $rc end $(date -u +%FT%TZ)"
exit "$rc"
