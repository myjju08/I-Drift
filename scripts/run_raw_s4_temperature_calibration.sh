#!/bin/bash
# Verify a supplied production DINO profile, optionally capture three seeds.
# Outputs stay outside this checkout; the historical tau artifact stays external.
# No new temperatures are fitted or installed by this wrapper.
set -euo pipefail

IDRIFT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IDRIFT_PYTHON=${IDRIFT_PYTHON:-python}
CALIBRATION_TOOL=${IDRIFT_ROOT}/scripts/calibrate_feature_encoder_temperatures.py

usage() {
  cat <<'USAGE'
Usage:
  run_raw_s4_temperature_calibration.sh verify CONFIG [OUTPUT_JSON]
  run_raw_s4_temperature_calibration.sh capture CONFIG OUTPUT_DIRECTORY GPU

verify runs the same CPU provenance/temperature guard as production training.
capture verifies first, then captures DINO statistics for seeds 43, 44, and 45
sequentially on one explicitly selected GPU. It performs no optimizer updates,
reference-encoder runs, or production temperature fitting. The config must
reference the existing external calibrated artifact and DINO checkpoint.
All outputs must be new paths outside this code checkout. Relative paths in
CONFIG are interpreted relative to the checkout, as in a normal training launch.
Set IDRIFT_PYTHON to select a Python interpreter with the project dependencies.
USAGE
}

if (($# == 0)); then usage >&2; exit 2; fi
case "$1" in
  -h|--help) usage; exit 0 ;;
  verify)
    if (($# < 2 || $# > 3)); then usage >&2; exit 2; fi
    config=$(realpath -- "$2")
    output_args=()
    if (($# == 3)); then output_args=(--output "$(realpath -m -- "$3")"); fi
    cd -- "${IDRIFT_ROOT}"
    exec "${IDRIFT_PYTHON}" "${CALIBRATION_TOOL}" verify-artifact \
      --config "${config}" "${output_args[@]}"
    ;;
  capture)
    if (($# != 4)); then usage >&2; exit 2; fi
    config=$(realpath -- "$2")
    output_root=$(realpath -m -- "$3")
    gpu=$4
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "GPU must be one physical GPU index" >&2
      exit 2
    fi
    case "${output_root}/" in
      "${IDRIFT_ROOT}/"*) echo "Output directory must be outside the checkout" >&2; exit 2 ;;
    esac
    if [[ -e "${output_root}" ]]; then
      echo "Refusing to reuse calibration output directory: ${output_root}" >&2
      exit 1
    fi
    cd -- "${IDRIFT_ROOT}"
    "${IDRIFT_PYTHON}" "${CALIBRATION_TOOL}" verify-artifact \
      --config "${config}"
    mkdir -p -- "${output_root}"
    for seed in 43 44 45; do
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
        "${IDRIFT_PYTHON}" -u "${CALIBRATION_TOOL}" capture \
          --config "${config}" \
          --workdir "${output_root}/work_dino_s${seed}" \
          --output "${output_root}/dino_s${seed}.json" \
          --seed "${seed}" --max-token-rows 64 \
          --grid-min 1 --grid-max 1 --grid-steps 1
    done
    ;;
  *) usage >&2; exit 2 ;;
esac
