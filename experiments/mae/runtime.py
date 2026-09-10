"""Local entry point for the live MAE256 control and spatial latent GAN recipes.

Run with ``torchrun --nproc_per_node=2 --module experiments.mae.runtime ...``.
Use ``python -m experiments.mae.runtime --config ... --preflight`` for a CPU,
read-only asset/cache check. Relative asset paths are rooted at this repository;
no imports or fallback paths point into the original experiment directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from . import latent_cache


REPO = Path(__file__).resolve().parents[2]
PATH_ENVIRON = {
    "imagenet_path": "IDRIFT_IMAGENET_PATH",
    "cache_path": "IDRIFT_MAE_CACHE",
    "feature_checkpoint": "IDRIFT_MAE_CHECKPOINT",
    "mae_checkpoint": "IDRIFT_MAE_CHECKPOINT",
    "feature_vae_model_id": "IDRIFT_MAE_VAE",
}
METRIC_NAMES = {
    "adversarial/dino_pixel_grad_norm": "adversarial/mae_latent_grad_norm",
    "adversarial/g_pixel_grad_norm": "adversarial/auxiliary_latent_grad_norm",
    "adversarial/g_to_dino_pixel_grad_ratio": "adversarial/auxiliary_to_mae_latent_grad_ratio",
}


def resolve_config_paths(cfg, *, repo_root=REPO, environ=None):
    """Resolve path fields without changing the scientific configuration."""
    environ = os.environ if environ is None else environ
    result = copy.deepcopy(cfg)
    for key, variable in PATH_ENVIRON.items():
        value = environ.get(variable) or result.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(repo_root) / path
        result[key] = str(path.resolve())
        # Logger/checkpoint config snapshots must show the paths actually used.
        for section in result.get("_raw", {}).values():
            if isinstance(section, dict) and key in section:
                section[key] = result[key]
    return result


def validate_recipe(cfg):
    """Reject incompatible domains before opening a cache or allocating a GPU."""
    if cfg.get("cache_format") != "packed_npy_latents_v1":
        raise ValueError("MAE runtime requires cache_format=packed_npy_latents_v1")
    if not cfg.get("use_cache") or not cfg.get("use_latent"):
        raise ValueError("MAE runtime requires the packed latent cache")
    if cfg.get("feature_extractor") not in {"mae", "mae_resnet", "mae_resnet256"}:
        raise ValueError("MAE runtime requires the frozen MAE feature extractor")
    if int(cfg.get("input_size", 0)) != 32 or int(cfg.get("in_channels", 0)) != 4:
        raise ValueError("Expected a generator producing four-channel 32x32 latents")
    if cfg.get("adversarial_mode", "none") not in {"none", "raw_gan", "feature_drift"}:
        raise ValueError("Choose control (none), raw_gan, or feature_drift")
    if cfg.get("adversarial_mode") != "none":
        if cfg.get("adversarial_architecture") != "latent_spatial_844":
            raise ValueError("The current MAE GAN architecture is latent_spatial_844")
        if cfg.get("adversarial_input_space") != "latent":
            raise ValueError("The current MAE discriminator consumes latent inputs")


def configure_affinity(environ=None):
    """Optional rank CPU pinning; hardware-specific CPU lists are not defaults."""
    environ = os.environ if environ is None else environ
    encoded = environ.get("MAE256_RANK_CPUSETS")
    if not encoded:
        return
    plans = json.loads(encoded)
    rank = int(environ.get("LOCAL_RANK", "0"))
    if rank < 0 or rank >= len(plans):
        raise ValueError("MAE256_RANK_CPUSETS has no entry for this local rank")
    cpus = set(plans[rank])
    if not cpus or not cpus.issubset(os.sched_getaffinity(0)):
        raise ValueError("Requested CPU affinity is outside the process allocation")
    os.sched_setaffinity(0, cpus)


def install(trainer, cfg):
    """Install only the same loader, discriminator, metrics and logging hooks."""
    from .latent_spatial_gan import build_adversarial_system

    validate_recipe(cfg)
    trainer.build_adversarial_system = build_adversarial_system
    trainer.create_imagenet_split = latent_cache.make_create_imagenet_split(
        trainer.create_imagenet_split, cfg["cache_path"]
    )
    original_load = trainer.load_yaml_config
    trainer.load_yaml_config = lambda path: resolve_config_paths(original_load(path))
    original_step = trainer.train_step

    def train_step(*args, **kwargs):
        loss, metrics, extras = original_step(*args, **kwargs)
        for old, new in METRIC_NAMES.items():
            if old in metrics:
                metrics[new] = metrics.pop(old)
        return loss, metrics, extras

    trainer.train_step = train_step
    original_logger = trainer.Logger

    class OnlineLogger(original_logger):
        def __init__(self, workdir, cfg, rank):
            super().__init__(workdir, cfg, rank)
            if rank == 0 and cfg.get("use_wandb"):
                import wandb
                if not self.use_wandb or getattr(self, "_wandb", None) is None or wandb.run is None:
                    raise RuntimeError("Online W&B initialization failed; refusing an unlogged run")
                if wandb.run.name != cfg["name"]:
                    raise RuntimeError("W&B run name does not match the selected experiment")

    trainer.Logger = OnlineLogger


def preflight(cfg, *, verify_asset_hashes=False):
    """Read manifests/files on CPU; never construct an encoder, VAE or GAN."""
    validate_recipe(cfg)
    checkpoint = Path(cfg["feature_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen MAE checkpoint missing: {checkpoint}")
    if Path(cfg["mae_checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("feature_checkpoint and mae_checkpoint must identify the same frozen MAE")
    vae = Path(cfg["feature_vae_model_id"])
    required = [checkpoint, vae / "config.json", vae / "diffusion_pytorch_model.safetensors"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required local evaluation asset missing: {path}")
    index = latent_cache.validate_complete(cfg["cache_path"])
    report = {
        "training_started": False, "mode": cfg["adversarial_mode"],
        "cache_complete": True, "cache_counts": index["expected_counts"],
        "checkpoint": str(checkpoint), "cache_path": cfg["cache_path"],
        "vae_path": str(vae), "adversarial_vae_decode": False,
        "architecture": cfg.get("adversarial_architecture"),
        "asset_hashes_verified": False,
    }
    if verify_asset_hashes:
        pins = json.loads((Path(__file__).parent / "provenance.json").read_text())["asset_sha256"]
        for key, path in zip(("mae_checkpoint", "vae_config", "vae_weights"), required):
            if latent_cache._sha(path) != pins[key]:
                raise RuntimeError(f"Asset differs from the live experiment: {path}")
        report["asset_hashes_verified"] = True
    return report


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-asset-hashes", action="store_true")
    options, _ = parser.parse_known_args()
    if options.verify_asset_hashes and not options.preflight:
        parser.error("--verify-asset-hashes requires --preflight")
    import train_imagenet_gen as trainer
    if not options.config:
        # The shared parser provides its full --help and required-argument error.
        trainer.main()
        return
    cfg = resolve_config_paths(trainer.load_yaml_config(options.config))
    if options.preflight:
        print(json.dumps(preflight(cfg, verify_asset_hashes=options.verify_asset_hashes), indent=2))
        return
    configure_affinity()
    install(trainer, cfg)
    print("[MAE256] Frozen MAE drifting + latent D32 native stages 8/4/4; VAE only for evaluation/images", flush=True)
    trainer.main()


if __name__ == "__main__":
    main()
