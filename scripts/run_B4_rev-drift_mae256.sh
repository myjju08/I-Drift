#!/usr/bin/env bash
# B/4 generator + official full-resolution latent MAE-256 + reverse drift.
# Matched to run_B2_rev-drift.sh except for generator patch size and MAE width.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"

CONFIG="${CONFIG:-${ROOT}/configs/gen/B4_rev-drift_mae256.yaml}"
LAYER_TEMPERATURE_PROFILE="${LAYER_TEMPERATURE_PROFILE:-uniform}"
case "${LAYER_TEMPERATURE_PROFILE}" in
  uniform|shallow_hot|deep_sharp|depth_profile) ;;
  *)
    echo "[error] unknown LAYER_TEMPERATURE_PROFILE=${LAYER_TEMPERATURE_PROFILE}"
    echo "[error] choose: uniform, shallow_hot, deep_sharp, depth_profile"
    exit 1
    ;;
esac
FEATURE_LOSS_PROFILE="${FEATURE_LOSS_PROFILE:-all}"
case "${FEATURE_LOSS_PROFILE}" in
  all|no_global|no_norm_x|no_stage1|no_stage2|no_stage3|no_stage4|global_x8|no_stage1_norm_x2|no_stage2_norm_x2|no_stage12|no_stage12_norm_x2) ;;
  *)
    echo "[error] unknown FEATURE_LOSS_PROFILE=${FEATURE_LOSS_PROFILE}"
    echo "[error] choose: all, no_global, no_norm_x, no_stage1, no_stage2, no_stage3, no_stage4, global_x8, no_stage1_norm_x2, no_stage2_norm_x2, no_stage12, no_stage12_norm_x2"
    exit 1
    ;;
esac
DEFAULT_WORKDIR="${ROOT}/runs/gen_B4_revdrift_mae256_layerT_${LAYER_TEMPERATURE_PROFILE}"
if [[ "${FEATURE_LOSS_PROFILE}" != "all" ]]; then
  DEFAULT_WORKDIR="${DEFAULT_WORKDIR}_lossW_${FEATURE_LOSS_PROFILE}"
fi
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
RUN_NAME="${RUN_NAME:-$(basename "${WORKDIR}")}"
LOG_DIR="${ROOT}/runs/launch_logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pid_${RUN_NAME}}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29626}"
LAUNCH_MODE="${LAUNCH_MODE:-foreground}"  # foreground | background | follow
REV_DRIFT_TOP_P="${REV_DRIFT_TOP_P:-1.0}"
DRIFT_TOP_P_MIN_KEEP="${DRIFT_TOP_P_MIN_KEEP:-1}"
DRIFT_TOP_K_POS="${DRIFT_TOP_K_POS:-0}"
DRIFT_TOP_K_NEG="${DRIFT_TOP_K_NEG:-0}"
DRIFT_TOP_K_GROUPS="${DRIFT_TOP_K_GROUPS:-all}"
TRAIN_SEED="${TRAIN_SEED:--1}"
SEED_HOST_RNG="${SEED_HOST_RNG:-keep}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-0}"
POS_PER_SAMPLE="${POS_PER_SAMPLE:-0}"
NEG_PER_SAMPLE="${NEG_PER_SAMPLE:-0}"
TOTAL_GENERATED_EPOCHS="${TOTAL_GENERATED_EPOCHS:-0}"
SAVE_PER_GENERATED_EPOCHS="${SAVE_PER_GENERATED_EPOCHS:-0}"
EVAL_PER_STEP="${EVAL_PER_STEP:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
THROUGHPUT_OPT_LEVEL="${THROUGHPUT_OPT_LEVEL:--1}"
GENERATOR_REMAT="${GENERATOR_REMAT:-keep}"
MAE_REMAT="${MAE_REMAT:-keep}"
HISTORICAL_GEN_REPLAY="${HISTORICAL_GEN_REPLAY:-keep}"
HISTORICAL_GEN_REPLAY_RATIO="${HISTORICAL_GEN_REPLAY_RATIO:--1}"
HISTORICAL_GEN_REPLAY_COUNT="${HISTORICAL_GEN_REPLAY_COUNT:-0}"
HISTORICAL_GEN_REPLAY_BANK_COUNT="${HISTORICAL_GEN_REPLAY_BANK_COUNT:--1}"
HISTORICAL_GEN_REPLAY_START_EPOCH="${HISTORICAL_GEN_REPLAY_START_EPOCH:-0}"
HISTORICAL_GEN_REPLAY_SOURCE="${HISTORICAL_GEN_REPLAY_SOURCE:-keep}"
HISTORICAL_GEN_REPLAY_POLICY="${HISTORICAL_GEN_REPLAY_POLICY:-keep}"
HISTORICAL_GEN_REPLAY_UPDATE_COUNT="${HISTORICAL_GEN_REPLAY_UPDATE_COUNT:-0}"
HISTORICAL_GEN_REPLAY_UPDATE_INTERVAL_STEPS="${HISTORICAL_GEN_REPLAY_UPDATE_INTERVAL_STEPS:-0}"
HISTORICAL_GEN_REPLAY_USAGE_BUDGET="${HISTORICAL_GEN_REPLAY_USAGE_BUDGET:-0}"
HISTORICAL_GEN_CURRENT_WEIGHT="${HISTORICAL_GEN_CURRENT_WEIGHT:-keep}"
HISTORICAL_GEN_HISTORY_WEIGHT="${HISTORICAL_GEN_HISTORY_WEIGHT:-keep}"
HISTORICAL_GEN_REPLAY_RATIO_START="${HISTORICAL_GEN_REPLAY_RATIO_START:-keep}"
HISTORICAL_GEN_REPLAY_RAMP_START_STEP="${HISTORICAL_GEN_REPLAY_RAMP_START_STEP:-keep}"
HISTORICAL_GEN_REPLAY_RAMP_END_STEP="${HISTORICAL_GEN_REPLAY_RAMP_END_STEP:-keep}"

IMAGENET_PATH="${IMAGENET_PATH:-/home1/irteam/data-vol1/osilab/hojung/data/imagenet_kaggle/ILSVRC2012}"
IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-/home1/irteam/data-vol1/osilab/hojung/data/image_test/image_latents}"
IMAGENET_CACHE_FORMAT="${IMAGENET_CACHE_FORMAT:-pt_imagefolder}"
SD_VAE_VARIANT="${SD_VAE_VARIANT:-mse}"
MAE_CKPT="${MAE_CKPT:-${ROOT}/weights/pt/mae_latent_256/ckpt_latest.pt}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] config not found: ${CONFIG}"; exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "[error] torchrun executable not found: ${TORCHRUN_BIN}"; exit 1
fi
if [[ ! -f "${MAE_CKPT}" ]]; then
  echo "[error] MAE checkpoint not found: ${MAE_CKPT}"; exit 1
fi
if [[ ! -d "${IMAGENET_CACHE_PATH}/train" || ! -d "${IMAGENET_CACHE_PATH}/val" ]]; then
  echo "[error] latent cache path invalid: ${IMAGENET_CACHE_PATH}"; exit 1
fi

mkdir -p "${WORKDIR}" "${LOG_DIR}"
: > "${LOG_FILE}"

echo "[launch] B/4 + official full-resolution latent MAE-256 + reverse drift"
echo "[launch] config=${CONFIG}"
echo "[launch] workdir=${WORKDIR}"
echo "[launch] GPUs=${GPU_IDS} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "[launch] mode=${LAUNCH_MODE}"
echo "[launch] torchrun=${TORCHRUN_BIN}"
echo "[launch] layer_temperature_profile=${LAYER_TEMPERATURE_PROFILE}"
echo "[launch] feature_loss_profile=${FEATURE_LOSS_PROFILE}"
echo "[launch] train_seed=${TRAIN_SEED} (negative keeps config)"
echo "[launch] seed_host_rng=${SEED_HOST_RNG}"
echo "[launch] batch=${TRAIN_BATCH_SIZE:-config} pos=${POS_PER_SAMPLE:-config} neg=${NEG_PER_SAMPLE:-config}"
echo "[launch] generated_epochs total=${TOTAL_GENERATED_EPOCHS:-config} save_every=${SAVE_PER_GENERATED_EPOCHS:-config}"
echo "[launch] eval_per_step=${EVAL_PER_STEP} (0 keeps config)"
echo "[launch] max_steps=${MAX_STEPS} (0 runs the full config)"
echo "[launch] throughput_opt_level=${THROUGHPUT_OPT_LEVEL} (-1 keeps config)"
echo "[launch] remat generator=${GENERATOR_REMAT} mae=${MAE_REMAT}"
echo "[launch] historical_replay=${HISTORICAL_GEN_REPLAY} ratio=${HISTORICAL_GEN_REPLAY_RATIO} count=${HISTORICAL_GEN_REPLAY_COUNT} bank=${HISTORICAL_GEN_REPLAY_BANK_COUNT} start_epoch=${HISTORICAL_GEN_REPLAY_START_EPOCH}"
echo "[launch] historical_source=${HISTORICAL_GEN_REPLAY_SOURCE} policy=${HISTORICAL_GEN_REPLAY_POLICY} update_count=${HISTORICAL_GEN_REPLAY_UPDATE_COUNT} update_interval_steps=${HISTORICAL_GEN_REPLAY_UPDATE_INTERVAL_STEPS} usage_budget=${HISTORICAL_GEN_REPLAY_USAGE_BUDGET}"
echo "[launch] historical_weights current=${HISTORICAL_GEN_CURRENT_WEIGHT} history=${HISTORICAL_GEN_HISTORY_WEIGHT} ramp_ratio=${HISTORICAL_GEN_REPLAY_RATIO_START} ramp_steps=${HISTORICAL_GEN_REPLAY_RAMP_START_STEP}:${HISTORICAL_GEN_REPLAY_RAMP_END_STEP}"
echo "[launch] top-p rev=${REV_DRIFT_TOP_P} min_keep=${DRIFT_TOP_P_MIN_KEEP}"
echo "[launch] top-k row-wise pos=${DRIFT_TOP_K_POS} neg=${DRIFT_TOP_K_NEG} groups=${DRIFT_TOP_K_GROUPS}"
echo "[launch] cache=${IMAGENET_CACHE_PATH}"
echo "[launch] cache_format=${IMAGENET_CACHE_FORMAT} vae_variant=${SD_VAE_VARIANT}"
echo "[launch] mae_checkpoint=${MAE_CKPT}"
echo "[launch] log=${LOG_FILE}"

RUN_CMD=(
  env
  PYTHONUNBUFFERED=1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IMAGENET_PATH="${IMAGENET_PATH}"
  IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH}"
  IMAGENET_CACHE_FORMAT="${IMAGENET_CACHE_FORMAT}"
  SD_VAE_VARIANT="${SD_VAE_VARIANT}"
  "${TORCHRUN_BIN}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT}/train_imagenet_gen.py"
  --config "${CONFIG}"
  --workdir "${WORKDIR}"
  --mae_checkpoint "${MAE_CKPT}"
  --seed "${TRAIN_SEED}"
  --seed_host_rng "${SEED_HOST_RNG}"
  --rev_drift_top_p "${REV_DRIFT_TOP_P}"
  --drift_top_p_min_keep "${DRIFT_TOP_P_MIN_KEEP}"
  --drift_top_k_pos "${DRIFT_TOP_K_POS}"
  --drift_top_k_neg "${DRIFT_TOP_K_NEG}"
  --drift_top_k_groups "${DRIFT_TOP_K_GROUPS}"
  --layer_temperature_profile "${LAYER_TEMPERATURE_PROFILE}"
  --feature_loss_profile "${FEATURE_LOSS_PROFILE}"
  --batch_size "${TRAIN_BATCH_SIZE}"
  --pos_per_sample "${POS_PER_SAMPLE}"
  --neg_per_sample "${NEG_PER_SAMPLE}"
  --total_generated_epochs "${TOTAL_GENERATED_EPOCHS}"
  --save_per_generated_epochs "${SAVE_PER_GENERATED_EPOCHS}"
  --eval_per_step "${EVAL_PER_STEP}"
  --max_steps "${MAX_STEPS}"
  --throughput_opt_level "${THROUGHPUT_OPT_LEVEL}"
  --generator_remat "${GENERATOR_REMAT}"
  --mae_remat "${MAE_REMAT}"
  --historical_gen_replay "${HISTORICAL_GEN_REPLAY}"
  --historical_gen_replay_ratio "${HISTORICAL_GEN_REPLAY_RATIO}"
  --historical_gen_replay_count "${HISTORICAL_GEN_REPLAY_COUNT}"
  --historical_gen_replay_bank_count "${HISTORICAL_GEN_REPLAY_BANK_COUNT}"
  --historical_gen_replay_start_generated_epochs "${HISTORICAL_GEN_REPLAY_START_EPOCH}"
  --historical_gen_replay_source "${HISTORICAL_GEN_REPLAY_SOURCE}"
  --historical_gen_replay_policy "${HISTORICAL_GEN_REPLAY_POLICY}"
  --historical_gen_replay_update_count "${HISTORICAL_GEN_REPLAY_UPDATE_COUNT}"
  --historical_gen_replay_update_interval_steps "${HISTORICAL_GEN_REPLAY_UPDATE_INTERVAL_STEPS}"
  --historical_gen_replay_usage_budget "${HISTORICAL_GEN_REPLAY_USAGE_BUDGET}"
)

if [[ "${HISTORICAL_GEN_CURRENT_WEIGHT}" != "keep" ]]; then
  RUN_CMD+=(--historical_gen_current_weight "${HISTORICAL_GEN_CURRENT_WEIGHT}")
fi
if [[ "${HISTORICAL_GEN_HISTORY_WEIGHT}" != "keep" ]]; then
  RUN_CMD+=(--historical_gen_history_weight "${HISTORICAL_GEN_HISTORY_WEIGHT}")
fi
if [[ "${HISTORICAL_GEN_REPLAY_RATIO_START}" != "keep" ]]; then
  RUN_CMD+=(--historical_gen_replay_ratio_start "${HISTORICAL_GEN_REPLAY_RATIO_START}")
fi
if [[ "${HISTORICAL_GEN_REPLAY_RAMP_START_STEP}" != "keep" ]]; then
  RUN_CMD+=(--historical_gen_replay_ratio_ramp_start_step "${HISTORICAL_GEN_REPLAY_RAMP_START_STEP}")
fi
if [[ "${HISTORICAL_GEN_REPLAY_RAMP_END_STEP}" != "keep" ]]; then
  RUN_CMD+=(--historical_gen_replay_ratio_ramp_end_step "${HISTORICAL_GEN_REPLAY_RAMP_END_STEP}")
fi

if [[ "${LAUNCH_MODE}" == "foreground" ]]; then
  "${RUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

setsid -f "${RUN_CMD[@]}" >> "${LOG_FILE}" 2>&1
sleep 2

PID="$(pgrep -f "torchrun.*--master_port=${MASTER_PORT}.*${ROOT}/train_imagenet_gen.py.*--config ${CONFIG}.*--workdir ${WORKDIR}" | head -n 1 || true)"
if [[ -z "${PID}" ]]; then
  PID="$(pgrep -f "${ROOT}/train_imagenet_gen.py.*--config ${CONFIG}.*--workdir ${WORKDIR}" | head -n 1 || true)"
fi
if [[ -n "${PID}" ]]; then
  echo "${PID}" > "${PID_FILE}"
  echo "[launch] pid=${PID}"
  echo "[launch] pid file=${PID_FILE}"
else
  echo "[warn] detached launch completed, but PID auto-detect failed."
  echo "[warn] monitor with: pgrep -fa \"${ROOT}/train_imagenet_gen.py\""
fi

if [[ "${LAUNCH_MODE}" == "follow" ]]; then
  echo "[launch] following ${LOG_FILE} (Ctrl-C stops log following only)"
  exec tail -n 50 -f "${LOG_FILE}"
fi

echo "[launch] tail -f ${LOG_FILE}"
