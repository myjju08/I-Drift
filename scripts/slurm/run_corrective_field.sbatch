#!/usr/bin/env bash
#SBATCH --job-name=CF-suite
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
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
CF_SNAPSHOT=${1:?Use scripts/submit_corrective_field.py --submit}
CF_VARIANT=${2:?A suite variant is required}
CF_RUN_ROOT=${3:-/home/juhyeong/corrective-field-runs}
CF_PYTHON=${4:-/home/juhyeong/.venvs/replay-drift/bin/python}
case "$CF_VARIANT" in baseline|replay_double|replay_double_gan) ;; *) echo "Unknown variant: $CF_VARIANT" >&2; exit 2 ;; esac
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign the requested two GPUs}"
CF_SOURCE="$CF_SNAPSHOT/source"
CF_CONFIG="$CF_SOURCE/configs/corrective_field/$CF_VARIANT.yaml"
test -r "$CF_SNAPSHOT/source-manifest.json"
test -x "$CF_PYTHON"
export PYTHONDONTWRITEBYTECODE=1
CF_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
CF_WORKDIR=$("$CF_PYTHON" "$CF_SOURCE/scripts/corrective_field_runtime.py" workdir \
  --snapshot "$CF_SNAPSHOT" --variant "$CF_VARIANT" --run-root "$CF_RUN_ROOT" \
  --job-id "$SLURM_JOB_ID" --restart-count "$CF_RESTART_COUNT")
cp "$CF_SNAPSHOT/source-manifest.json" "$CF_WORKDIR/run_metadata/source-manifest.json"
cp "$CF_CONFIG" "$CF_WORKDIR/run_metadata/config.yaml"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NUMPY_MADVISE_HUGEPAGE=0
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
export WANDB_PROJECT='Corrective Field'
export WANDB_ENTITY=a01065522071-kaist-digital-humanities-and-social-science
export WANDB_RUN_GROUP="corrective-field-$(basename "$CF_SNAPSHOT")"
export WANDB_JOB_TYPE=training
export WANDB_DIR="$CF_WORKDIR"
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME IDRIFT_WANDB_NAME

echo "[launch] variant=$CF_VARIANT job=$SLURM_JOB_ID host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[launch] source=$CF_SOURCE config=$CF_CONFIG workdir=$CF_WORKDIR"
"$CF_PYTHON" - "$CF_SNAPSHOT" "$CF_VARIANT" "$CF_WORKDIR" <<'PY'
import json, os, pathlib, socket, sys
snapshot, variant, workdir = sys.argv[1:]
metadata = {
    "snapshot": snapshot, "variant": variant, "workdir": workdir,
    "host": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "wandb_group": os.environ["WANDB_RUN_GROUP"],
    "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
}
pathlib.Path(workdir, "run_metadata", f"launch-restart{metadata['restart_count']}.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
cd "$CF_SOURCE"
"$CF_PYTHON" scripts/preflight_corrective_field.py \
  --variant "$CF_VARIANT" --config "$CF_CONFIG" \
  --suite-dir "$CF_SOURCE/configs/corrective_field" \
  --runtime-source "$CF_SOURCE" --snapshot-manifest "$CF_SNAPSHOT/source-manifest.json" \
  --output "$CF_WORKDIR/run_metadata/input-preflight-restart${CF_RESTART_COUNT}.json"
if (( CF_RESTART_COUNT > 0 )); then
  "$CF_PYTHON" scripts/corrective_field_runtime.py checkpoint --workdir "$CF_WORKDIR" --config "$CF_CONFIG"
fi

CF_TRAIN_PID=''
cf_time_limit_requeue() {
  trap '' USR1
  echo "[requeue] Slurm time-limit warning; stopping training to resume its last completed checkpoint"
  if [[ -n "$CF_TRAIN_PID" ]]; then
    kill -TERM -- "-$CF_TRAIN_PID" 2>/dev/null || true
    wait "$CF_TRAIN_PID" || true
  fi
  "$CF_PYTHON" scripts/corrective_field_runtime.py checkpoint --workdir "$CF_WORKDIR" --config "$CF_CONFIG"
  echo "[requeue] Validated same-arm checkpoint; requeueing job=$SLURM_JOB_ID workdir=$CF_WORKDIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap cf_time_limit_requeue USR1
# A separate process group lets the time-limit trap stop torchrun and both ranks.
# Other training failures propagate normally and are not automatically requeued.
setsid "$CF_PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_imagenet_gen.py --config "$CF_CONFIG" --workdir "$CF_WORKDIR" &
CF_TRAIN_PID=$!
wait "$CF_TRAIN_PID"
