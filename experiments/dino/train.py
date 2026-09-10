"""Run or CPU-check the portable DINO adversarial experiment snapshots.

Without --train this checks configuration and lossless I/O on CPU. Full runs
require torchrun with two workers, a fresh workdir and the calibrated assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRESETS = ('dino_only_replay_feature', 'replay_feature_hardcopy',
           'replay_raw_gan_d32', 'replay_raw_gan_d128')
PATH_FIELDS = {'imagenet_path': 'env', 'feature_checkpoint': 'feature',
               'temperature_calibration_artifact': 'train', 'cache_path': 'env'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_config(preset='dino_only_replay_feature', config=None, **overrides):
    """Resolve relative asset paths against this checkout, never the caller cwd."""
    import train_imagenet_gen as trainer
    path = Path(config).expanduser().resolve() if config else ROOT / 'configs' / f'{preset}.yaml'
    if not config:
        provenance = json.loads((ROOT / 'provenance.json').read_text())['presets'][preset]
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['ported_sha256']:
            raise RuntimeError('Bundled preset changed; use --config for a deliberate custom experiment')
    cfg = trainer.load_yaml_config(str(path))
    for field, section in PATH_FIELDS.items():
        value = overrides.get(field) or cfg.get(field)
        if value:
            candidate = Path(value).expanduser()
            value = str((REPO / candidate if not candidate.is_absolute() else candidate).resolve())
            cfg[field] = value
            cfg['_raw'].setdefault(section, {})[field] = value
    return cfg


def describe(cfg, preset=None):
    return {
        'preset': preset, 'name': cfg.get('name'), 'world_size': 2,
        'generator_seed_per_rank': [int(cfg.get('seed', 43)), int(cfg.get('seed', 43)) + 1],
        'generated_per_step': 2 * int(cfg['batch_size']) * int(cfg['gen_per_label']),
        'feature_extractor': cfg.get('feature_extractor'),
        'adversarial': {k: v for k, v in cfg.items() if k.startswith('adversarial_')},
        'replay': {k: v for k, v in cfg.items() if k.startswith('historical_gen_')},
        'samples_per_rank': {k: cfg[k] for k in ('batch_size', 'gen_per_label', 'pos_per_sample', 'neg_per_sample')},
        'schedule': {k: cfg.get(k) for k in ('total_generated_epochs', 'save_per_generated_epochs',
                                           'eval_per_generated_epochs', 'eval_samples', 'eval_at_start', 'cfg_list')},
    }


def preflight(cfg, *, preset=None, io_backend='packed', decode_backend='native',
              check_assets=False, check_generator=False):
    import torch
    import train_imagenet_gen as trainer
    from . import runtime
    report = describe(cfg, preset)
    artifact = Path(cfg['temperature_calibration_artifact'])
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual != cfg['temperature_calibration_artifact_sha256']:
        raise RuntimeError('Calibration artifact differs from the pinned snapshot')
    report['calibration_artifact_sha256'] = actual
    boundary = trainer._steps_for_generated_epochs(
        dataset_size=1281167, generated_per_step=report['generated_per_step'],
        epochs=float(cfg.get('historical_gen_replay_start_generated_epochs', 10)))
    report['replay_start_step'] = boundary
    report['replay_ratio_before'] = trainer._historical_replay_ratio_for_step(cfg, boundary - 1, active=False)
    report['replay_ratio_after'] = trainer._historical_replay_ratio_for_step(cfg, boundary, active=True)
    report['io_backend'] = io_backend
    report['assets_checked'] = bool(check_assets)
    if io_backend == 'packed':
        report['packed_bank'] = runtime.check_factory(decode_backend)
        report['decode_backend'] = decode_backend
        report['train_prefetch_factor_runtime'] = 1
    if check_assets:
        trainer._validate_raw_temperature_calibration(cfg)
        report['raw_imagenet_expected'] = trainer._load_complete_raw_imagenet_manifest(cfg['imagenet_path'])['expected']
    if check_generator:
        report['generator_initial_state'] = runtime.check_initial_generator(trainer, cfg)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    report.update(passed=True, cuda_initialized=False, training_started=False)
    return report


def _validate_step_contract(trainer, cfg, system, args, kwargs, number, preset):
    import torch
    if system is None or system.mode != cfg['adversarial_mode']:
        raise RuntimeError('Unexpected adversarial system')
    if number == 0:
        if system.online.base_channels != int(cfg['adversarial_base_channels']):
            raise RuntimeError('Discriminator width differs from preset')
        if system.ema_decay != 0 or system.structure_weight != 0:
            raise RuntimeError('The replay presets use a hard-copy target and no structure loss')
        if any(p.requires_grad for p in system.target.parameters()):
            raise RuntimeError('Critic target parameters must remain frozen')
        if tuple(args[3].shape) != (4,) or tuple(args[4].shape[:2]) != (4, 64) or tuple(args[5].shape[:2]) != (4, 32):
            raise RuntimeError('Expected per-rank B4/P64/N32 inputs')
        if kwargs.get('historical_samples') is not None:
            raise RuntimeError('Historical replay started before epoch 10')
    if number >= 50046:
        history = kwargs.get('historical_samples')
        if history is None or tuple(history.shape[:2]) != (4, 16):
            raise RuntimeError('Expected frozen replay particles H16 after epoch 10')


def require_fresh_wandb(logger, cfg, rank, *, matched_preset=False):
    """Preserve the live wrapper's refusal to silently drop online logging."""
    if rank != 0 or not cfg.get('use_wandb', False):
        return
    import wandb
    if not logger.use_wandb or wandb.run is None or wandb.run.resumed:
        raise RuntimeError('A fresh W&B run was required but did not initialize successfully')
    if matched_preset and wandb.run.settings.mode != 'online':
        raise RuntimeError('The matched DINO presets require fresh online W&B logging')


def train(cfg, workdir, *, preset=None, io_backend='packed', decode_backend='native'):
    import torch
    import torch.distributed as dist
    import train_imagenet_gen as trainer
    from models.dino_rf_tuning import module_fingerprint
    from . import runtime
    if int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise RuntimeError('Use torchrun --nproc_per_node=2 for the matched DINO experiments')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if 'IDRIFT_RANK_CPUSETS' in os.environ:
        plans = json.loads(os.environ['IDRIFT_RANK_CPUSETS'])
        if len(plans) != 2 or not plans[local_rank] or not set(plans[local_rank]).issubset(os.sched_getaffinity(0)):
            raise RuntimeError('IDRIFT_RANK_CPUSETS must contain two valid CPU sets')
        os.sched_setaffinity(0, set(plans[local_rank]))
    # Fail before allocating CUDA resources if assets or native compilation are missing.
    trainer._validate_raw_temperature_calibration(cfg)
    trainer._load_complete_raw_imagenet_manifest(cfg['imagenet_path'])
    if io_backend == 'packed' and decode_backend == 'native':
        from .io import native_color_decoder  # noqa: F401
    workdir = Path(workdir).expanduser().resolve()
    if workdir.exists():
        raise RuntimeError(f'A fresh workdir is required; refusing checkpoint or W&B reuse: {workdir}')
    rank, world_size, device = trainer.setup_distributed()
    factory = None
    try:
        if rank == 0:
            workdir.mkdir(parents=True, exist_ok=False)
            atomic_json(workdir / 'effective_config.json', cfg)
            atomic_json(workdir / 'runtime_identity.json', {
                **describe(cfg, preset), 'generator_from_scratch': True, 'checkpoint_resumed': False,
                'io_backend': io_backend, 'decode_backend': decode_backend if io_backend == 'packed' else None,
                'prefetch_factor_runtime': 1 if io_backend == 'packed' else cfg.get('prefetch_factor'),
                'runtime_code_root': str(REPO),
                'provenance_sha256': hashlib.sha256((ROOT / 'provenance.json').read_bytes()).hexdigest(),
            })
        dist.barrier()
        atomic_json(workdir / f'process_rank{rank}.json', {
            'pid': os.getpid(), 'rank': rank, 'device': str(device),
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
        })
        timings = runtime.StageTimings()
        if io_backend == 'packed':
            runtime.install_compact_timed_loader(trainer, timings)
            factory = runtime.make_memory_bank_factory(workdir, rank, 8, decode_backend=decode_backend)
            trainer.ArrayMemoryBank = factory
        if preset:
            original_build = trainer.build_ditgen_from_config

            def build(*args, **kwargs):
                model = original_build(*args, **kwargs)
                fingerprint = module_fingerprint(model)
                if fingerprint != runtime.INITIAL_GENERATOR_SHA256[rank]:
                    raise RuntimeError('Generator initialization differs from the source experiment')
                atomic_json(workdir / f'generator_initial_rank{rank}.json',
                            {'sha256': fingerprint, 'seed': 43 + rank, 'matches_source': True})
                return model

            trainer.build_ditgen_from_config = build
            original_step = trainer.train_step

            def step(*args, **kwargs):
                number = int(args[7])
                system = kwargs.get('adversarial_system')
                _validate_step_contract(trainer, cfg, system, args, kwargs, number, preset)
                result = original_step(*args, **kwargs)
                metrics = result[1]
                if preset == 'dino_only_replay_feature' and (
                    metrics['adversarial/replay_enabled'] != 0 or metrics['adversarial/history_count'] != 0
                ):
                    raise RuntimeError('CNN replay must remain disabled; DINO alone receives historical particles')
                if number in (0, 50046):
                    equal = all(torch.equal(a, b) for a, b in zip(
                        system.online.state_dict().values(), system.target.state_dict().values()))
                    if not equal:
                        raise RuntimeError('Critic target is not an exact post-D-update copy')
                    if any(p.requires_grad for p in args[1].parameters()):
                        raise RuntimeError('DINO feature teacher is no longer frozen')
                    atomic_json(workdir / f'contract_step{number}_rank{rank}.json', {
                        'passed': True, 'target_exact_copy': equal, 'dino_frozen': True,
                        'direct_generator_gan_loss': 'adversarial/raw_gan_loss' in metrics,
                        'cnn_feature_drift': 'adversarial/feature_drift_loss' in metrics,
                        'cnn_replay': bool(cfg.get('adversarial_apply_replay', True)),
                        'historical_count': 0 if number == 0 else 16,
                    })
                return result

            trainer.train_step = step
        original_logger = trainer.Logger

        class Logger(original_logger):
            def __init__(self, workdir, active, local_rank):
                super().__init__(workdir, active, local_rank)
                require_fresh_wandb(self, active, local_rank, matched_preset=preset is not None)

            def set_step(self, number):
                super().set_step(number)
                timings.reset(number)

            def log(self, metrics, *args, **kwargs):
                if self.rank == 0 and io_backend == 'packed':
                    metrics = dict(metrics)
                    metrics.update({'time_cpu_rank0/' + key: value for key, value in timings.values.items()})
                return super().log(metrics, *args, **kwargs)

        trainer.Logger = Logger
        trainer.train_gen(cfg, str(workdir), rank, world_size, device)
    finally:
        if factory:
            for bank in factory.compressed_banks:
                bank.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--preset', choices=PRESETS, default='dino_only_replay_feature')
    source.add_argument('--config', type=Path, help='Custom YAML; bypasses preset identity/step guards')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--imagenet-path', type=Path)
    parser.add_argument('--feature-checkpoint', type=Path)
    parser.add_argument('--temperature-calibration-artifact', type=Path)
    parser.add_argument('--io-backend', choices=('auto', 'packed', 'standard'), default='auto')
    parser.add_argument('--decode-backend', choices=('native', 'numpy'), default='native')
    parser.add_argument('--check-assets', action='store_true')
    parser.add_argument('--check-generator', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if not args.train:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
    import torch
    if not args.train:
        torch.set_num_threads(1)
    preset = None if args.config else args.preset
    cfg = load_config(args.preset, args.config, imagenet_path=args.imagenet_path,
                      feature_checkpoint=args.feature_checkpoint,
                      temperature_calibration_artifact=args.temperature_calibration_artifact)
    io_backend = args.io_backend
    if io_backend == 'auto':
        io_backend = 'packed' if preset in PRESETS[:2] else 'standard'
    if args.train:
        if not args.workdir:
            parser.error('--train requires --workdir pointing to a new directory')
        train(cfg, args.workdir, preset=preset, io_backend=io_backend, decode_backend=args.decode_backend)
    else:
        report = preflight(cfg, preset=preset, io_backend=io_backend, decode_backend=args.decode_backend,
                           check_assets=args.check_assets, check_generator=args.check_generator)
        if args.report:
            atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
