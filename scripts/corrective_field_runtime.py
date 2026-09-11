"""Bind Slurm restarts to the same experiment and validate resume checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight_corrective_field import (
    DATASET_ROWS, GENERATED_PER_STEP, VARIANTS, read_config, selected_suite_dir, sha256, snapshot_execution,
)


def prepare_workdir(snapshot, variant, run_root, job_id, restart_count):
    snapshot = Path(snapshot).resolve(strict=True)
    source = snapshot / "source"
    manifest = json.loads((snapshot / "source-manifest.json").read_text())
    config = selected_suite_dir(source, manifest) / f"{variant}.yaml"
    identity = {"commit": manifest["commit"], "config_sha256": sha256(config),
                "variant": variant, "job_id": str(job_id), "snapshot": str(snapshot)}
    if "execution" in manifest:
        binding = snapshot_execution(manifest)
        if Path(run_root).resolve() != Path(binding["run_root"]).resolve():
            raise ValueError("Run root does not match the immutable execution binding")
        identity["execution"] = binding
    pointer = snapshot / "workdirs" / f"job{job_id}-{variant}.json"
    arm_root = Path(run_root).resolve() / variant
    if restart_count:
        if not pointer.is_file():
            raise ValueError("Slurm restart has no original workdir binding; refusing a fresh run")
        prior = json.loads(pointer.read_text())
        if any(prior.get(key) != value for key, value in identity.items()):
            raise ValueError("Slurm restart source/config/variant identity mismatch")
        workdir = Path(prior["workdir"]).resolve(strict=True)
        if workdir.parent != arm_root:
            raise ValueError("Slurm restart workdir belongs to another arm/root")
        if sha256(workdir / "run_metadata/config.yaml") != identity["config_sha256"]:
            raise ValueError("Slurm restart stored config checksum mismatch")
        return workdir
    if pointer.exists():
        raise ValueError("Job already has a workdir; refusing accidental reuse without Slurm restart")
    arm_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"job{job_id}-", dir=arm_root))
    (workdir / "run_metadata").mkdir()
    (workdir / "run_metadata/config.yaml").write_bytes(config.read_bytes())
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with pointer.open("x") as handle:
        json.dump({**identity, "workdir": str(workdir)}, handle, indent=2)
        handle.write("\n")
    return workdir


def validate_checkpoint(workdir, config):
    import torch
    workdir = Path(workdir)
    cfg = read_config(config)
    latest = workdir / "checkpoints/ckpt_latest.pt"
    if not latest.is_file():
        raise ValueError("No completed checkpoint is available for time-limit requeue")
    state = torch.load(latest, map_location="cpu", weights_only=False)
    if not {"step", "model", "ema", "optimizer", "config"}.issubset(state):
        raise ValueError("Resume checkpoint is incomplete")
    changed = [key for key, value in cfg.items() if state["config"].get(key) != value]
    if changed:
        raise ValueError(f"Resume checkpoint configuration mismatch: {changed}")
    step = int(state["step"])
    if not 0 < step <= int(cfg["total_steps"]):
        raise ValueError(f"Resume checkpoint step is invalid: {step}")
    if any(key in state for key in ("adversarial_system", "feature_discriminator", "feature_discriminator_optimizer")):
        raise ValueError("This suite disables GAN; refusing checkpoint discriminator state")
    run_id = workdir / "wandb_run_id.txt"
    if not run_id.is_file() or not run_id.read_text().strip():
        raise ValueError("Resume requires the original W&B run ID")
    replay_source = str(cfg.get("historical_gen_replay_source", "frozen_snapshot")).lower().strip()
    if cfg["historical_gen_replay"] and replay_source == "frozen_snapshot":
        freeze_step = math.ceil(DATASET_ROWS * cfg["historical_gen_replay_start_generated_epochs"] / GENERATED_PER_STEP)
        policy = str(cfg.get("historical_gen_replay_policy", "frozen")).lower().strip()
        for rank in range(2):
            if step < freeze_step:
                filename = f"historical_gen_replay_capture_step{step:07d}_rank{rank:02d}.npz"
            else:
                filename = f"historical_gen_replay_state_step{step:07d}_rank{rank:02d}.npz"
                # Match the trainer: prefer checkpoint-paired bank/telemetry,
                # with legacy immutable-bank fallback only for frozen replay.
                if policy == "frozen" and not (workdir / filename).is_file():
                    filename = f"historical_gen_replay_rank{rank:02d}.npz"
            path = workdir / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Resume requires matching replay state: {filename}")
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("workdir")
    prepare.add_argument("--snapshot", type=Path, required=True)
    prepare.add_argument("--variant", choices=VARIANTS, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--job-id", required=True)
    prepare.add_argument("--restart-count", type=int, default=0)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--workdir", type=Path, required=True)
    checkpoint.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "workdir":
        print(prepare_workdir(args.snapshot, args.variant, args.run_root, args.job_id, args.restart_count))
    else:
        step = validate_checkpoint(args.workdir, args.config)
        print(f"[resume] Valid checkpoint: step={step}; workdir={args.workdir}")


if __name__ == "__main__":
    main()
