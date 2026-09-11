"""Opt-in evaluation integrity guards, independent of metric implementation."""
from contextlib import contextmanager
import math
import traceback

import numpy as np
import torch
import torch.distributed as dist


@contextmanager
def preserve_evaluation_rng(enabled=False, *, device=None):
    """Restore the parent-process RNG streams after eval and its cleanup.

    CUDA training has already initialized its own device. Only snapshot that
    device: querying every visible GPU can create extra DDP CUDA contexts.
    """
    if not enabled:
        yield
        return
    cpu_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    device = torch.device(device) if device is not None else None
    cuda_state = (
        torch.cuda.get_rng_state(device)
        if device is not None and device.type == "cuda" and torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        np.random.set_state(numpy_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def validate_evaluation_metrics(stats, *, required=False, step, cfg_scale):
    """Require a complete finite FID/IS triplet without changing metric math."""
    if not required:
        return
    invalid = []
    for name in ("fid", "is_mean", "is_std"):
        value = stats.get(name)
        try:
            valid = value is not None and math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            invalid.append(f"{name}={value!r}")
    if invalid:
        raise RuntimeError(
            f"Required evaluation metrics invalid at step {step}, CFG={cfg_scale}: "
            + ", ".join(invalid)
        )


def run_rank_zero_evaluation(
    evaluate, *, rank, world_size, device, step,
    require_eval_metrics=False, preserve_rng_during_eval=False,
):
    """Wait, run rank-0 evaluation/cleanup, then share a strict failure flag.

    The callback must restore training models in its finally block. Catching
    outside that callback covers restoration failures as well as metric errors,
    so rank 1 cannot be stranded at the next training collective by rank 0.
    """
    distributed = world_size > 1
    if distributed:
        dist.barrier()
    error = None
    if rank == 0:
        try:
            with preserve_evaluation_rng(preserve_rng_during_eval, device=device):
                evaluate()
        except Exception as exc:
            error = exc
            print(f"[eval] Evaluation failed at step {step}: {exc}", flush=True)
            traceback.print_exc()

    failed = error is not None
    if distributed:
        if require_eval_metrics:
            # NCCL requires a CUDA flag; Gloo CPU tests need no CUDA context.
            flag_device = device if dist.get_backend() == "nccl" else torch.device("cpu")
            flag = torch.tensor(int(failed), dtype=torch.int32, device=flag_device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            failed = bool(flag.item())
        else:
            dist.barrier()
    if require_eval_metrics and failed:
        detail = f": {error}" if error is not None else "; see rank-0 evaluation error"
        raise RuntimeError(f"Required rank-0 evaluation failed at step {step}{detail}") from error
