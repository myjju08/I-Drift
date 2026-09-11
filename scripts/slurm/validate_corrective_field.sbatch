#!/usr/bin/env bash
#SBATCH --job-name=CF-validation
#SBATCH --account=osilab
#SBATCH --qos=normal
#SBATCH --partition=srv02
#SBATCH --nodelist=srv02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
CF_SNAPSHOT=${1:?Usage: sbatch validate_corrective_field.sbatch SNAPSHOT_ROOT [PYTHON] [RUN_ROOT]}
CF_PYTHON=${2:-/home/juhyeong/.venvs/replay-drift/bin/python}
CF_RUN_ROOT=${3:-/home/juhyeong/corrective-field-runs}
CF_SOURCE="$CF_SNAPSHOT/source"
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign two GPUs}"
test -r "$CF_SNAPSHOT/source-manifest.json"
test -x "$CF_PYTHON"
CF_VALIDATION_ROOT=${CORRECTIVE_VALIDATION_ROOT:-$CF_RUN_ROOT/validation}
mkdir -p "$CF_VALIDATION_ROOT"
CF_WORKDIR=$(mktemp -d "$CF_VALIDATION_ROOT/job${SLURM_JOB_ID}-XXXXXX")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NUMPY_MADVISE_HUGEPAGE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME
CF_SUITE_REL=$("$CF_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("execution", {}).get("suite_dir", "configs/corrective_field"))' "$CF_SNAPSHOT/source-manifest.json")
echo "[validation] source=$CF_SOURCE suite=$CF_SUITE_REL workdir=$CF_WORKDIR job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cd "$CF_SOURCE"
exec "$CF_PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/validate_corrective_field.py --snapshot-root "$CF_SNAPSHOT" --workdir "$CF_WORKDIR" \
  --suite-dir "$CF_SOURCE/$CF_SUITE_REL"
