#!/bin/bash
# torchrun --no_python entrypoint for the two-rank mixed GAN experiments.
# Optional: PYTHON_BIN, IDRIFT_RANK0_CPUS and IDRIFT_RANK1_CPUS.
# Supply dataset, pretrained encoder and calibration paths in the YAML locally.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rank_python="${PYTHON_BIN:-python}"
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
case "${LOCAL_RANK:?torchrun LOCAL_RANK required}" in
  0) rank_cpus="${IDRIFT_RANK0_CPUS:-}" ;;
  1) rank_cpus="${IDRIFT_RANK1_CPUS:-}" ;;
  *) echo "Expected exactly two ranks" >&2; exit 1 ;;
esac
if [[ -n "${rank_cpus}" ]]; then
  exec taskset -c "${rank_cpus}" "${rank_python}" -u "${repo_root}/train_imagenet_gen.py" "$@"
fi
exec "${rank_python}" -u "${repo_root}/train_imagenet_gen.py" "$@"
