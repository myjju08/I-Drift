#!/bin/bash
set -euo pipefail
case "${LOCAL_RANK:?torchrun LOCAL_RANK required}" in
  0) rank_cpus="${IDRIFT_RANK0_CPUS:?}" ;;
  1) rank_cpus="${IDRIFT_RANK1_CPUS:?}" ;;
  *) echo "Expected exactly two ranks" >&2; exit 1 ;;
esac
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec taskset -c "${rank_cpus}" "${PYTHON_BIN:-python}" -u "${repo_root}/train_imagenet_gen.py" "$@"
