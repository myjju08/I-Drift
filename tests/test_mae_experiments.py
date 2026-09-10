"""MAE spatial-GAN math, loader, runtime and optional live-source parity checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mae.latent_direct_gan import build_adversarial_system as build_direct
from experiments.mae.latent_spatial_gan import build_adversarial_system


def config(mode='raw_gan'):
    return dict(adversarial_mode=mode, adversarial_architecture='latent_spatial_844',
                adversarial_input_space='latent', adversarial_in_channels=4,
                use_latent=True, in_channels=4, feature_extractor='mae',
                historical_gen_replay=False, num_classes=2,
                adversarial_base_channels=32, adversarial_seed=43,
                adversarial_ema_decay=0., adversarial_structure_weight=0.,
                adversarial_r1_gamma=1., adversarial_r1_interval=16,
                adversarial_d_chunk_size=2, adversarial_g_chunk_size=2,
                adversarial_loss_weight=.15 if mode == 'raw_gan' else 0.,
                adversarial_drift_weight=.1 if mode == 'feature_drift' else 0.)


def build(mode='raw_gan'):
    return build_adversarial_system(config(mode), torch.device('cpu'))


def batch():
    rng = torch.Generator().manual_seed(2674)
    real = torch.randn(4, 4, 32, 32, generator=rng).requires_grad_()
    fake = torch.randn(4, 4, 32, 32, generator=rng).requires_grad_()
    return real, fake, torch.tensor([0, 0, 1, 1])


def equal_tensors(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            equal_tensors(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal_tensors(test, x, y)
    else:
        test.assertEqual(a, b)


class SpatialLatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_geometry_retains_independent_spatial_tokens_and_original_parameters(self):
        spatial = build('feature_drift')
        direct = build_direct(config('feature_drift'), torch.device('cpu'))
        equal_tensors(self, spatial.online.state_dict(), direct.online.state_dict())
        self.assertEqual(sum(p.numel() for p in spatial.online.parameters()),
                         sum(p.numel() for p in direct.online.parameters()))
        self.assertEqual({id(p) for p in spatial.online.parameters()},
                         {id(p) for g in spatial.optimizer.param_groups for p in g['params']})
        real, _, _ = batch()
        with torch.no_grad():
            maps = spatial.target._maps(real)
            tokens = spatial.target_features(real)
        self.assertEqual([tuple(v.shape[-2:]) for v in maps.values()], [(8, 8), (4, 4), (4, 4)])
        self.assertEqual([spatial.online.blocks[i][0].stride for i in range(4)],
                         [(2, 2), (2, 2), (2, 2), (1, 1)])
        for stage, values in tokens.items():
            expected = F.adaptive_avg_pool2d(maps[stage], (4, 4)).flatten(2).transpose(1, 2)
            expected = F.normalize(expected, dim=-1, eps=1e-8) * expected.shape[-1]**.5
            torch.testing.assert_close(values, expected, rtol=0, atol=0)
            self.assertEqual(values.shape[1], 16)
            self.assertEqual(torch.unique(values[0], dim=0).shape[0], 16)

    def test_both_modes_preserve_image_gradient_and_freeze_target_without_vae(self):
        from train_imagenet_gen import compute_drift_loss_from_features
        for mode in ('raw_gan', 'feature_drift'):
            with self.subTest(mode=mode), \
                 patch('vae_imagenet.load_vae', side_effect=AssertionError('VAE loaded')), \
                 patch('vae_imagenet.vae_decode', side_effect=AssertionError('VAE decoded')):
                system = build(mode)
                real, fake, labels = batch()
                if mode == 'raw_gan':
                    loss = .15 * F.softplus(-system.target_logits(fake, labels)).mean()
                else:
                    with torch.no_grad():
                        positive = system.target_features(real)
                    loss, _ = compute_drift_loss_from_features(
                        system.target_features(fake), positive, B=2, G=2, P=2, N=0,
                        neg_feats=None, weight_neg=None,
                        global_scale_stats=False, global_fnorm_stats=False)
                    loss = .1 * loss
                loss.backward()
                self.assertTrue(torch.isfinite(fake.grad).all())
                self.assertGreater(float(fake.grad.norm()), 0.)
                self.assertIsNone(real.grad)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in system.target.parameters()))
                self.assertTrue(all(p.grad is None for p in system.online.parameters()))
                self.assertFalse(hasattr(system, 'vae'))

    def test_raw_and_feature_D_updates_optimizer_and_hardcopy_are_identical(self):
        raw, feature = build(), build('feature_drift')
        real, fake, labels = batch()
        initial = {k: v.clone() for k, v in raw.online.state_dict().items()}
        for step in (0, 1, 2, 16):
            with self.subTest(step=step):
                a = raw.discriminator_step(real, fake, labels, labels, step)
                b = feature.discriminator_step(real, fake, labels, labels, step)
                self.assertEqual(a, b)
                self.assertEqual(a['adversarial/r1_applied'], float(step in (0, 16)))
                self.assertEqual(a['adversarial/structure_loss'], 0.)
                self.assertEqual(a['adversarial/d_updated'], 1.)
                equal_tensors(self, raw.online.state_dict(), feature.online.state_dict())
                equal_tensors(self, raw.optimizer.state_dict(), feature.optimizer.state_dict())
                equal_tensors(self, raw.online.state_dict(), raw.target.state_dict())
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in raw.target.parameters()))
        self.assertTrue(any(not torch.equal(initial[k], v) for k, v in raw.online.state_dict().items()))
        self.assertIsNone(real.grad)
        self.assertIsNone(fake.grad)

    def test_D_objective_is_logistic_plus_lazy_R1_in_latent_coordinates(self):
        system = build()
        real, fake, labels = batch()
        for step in (1, 16):
            with self.subTest(step=step):
                reference = copy.deepcopy(system.online)
                reference.train()
                rr = real.detach().requires_grad_(step == 16)
                lr = reference(rr, labels)
                lf = reference(fake.detach(), labels)
                logistic = F.softplus(-lr).mean() + F.softplus(lf).mean()
                r1 = (torch.autograd.grad(lr.sum(), rr, create_graph=True)[0]
                      .square().flatten(1).sum(1).mean()) if step == 16 else logistic.new_zeros(())
                metrics = system.discriminator_step(real, fake, labels, labels, step)
                self.assertAlmostEqual(metrics['adversarial/d_logistic_loss'], float(logistic.detach()), places=6)
                self.assertAlmostEqual(metrics['adversarial/r1'], float(r1.detach()), places=6)
                self.assertAlmostEqual(metrics['adversarial/d_loss'], float((logistic + 8 * r1).detach()), places=6)

    def test_state_rejects_old_geometry_and_roundtrips_new_geometry(self):
        system = build()
        real, fake, labels = batch()
        system.discriminator_step(real, fake, labels, labels, 0)
        state = copy.deepcopy(system.state_dict())
        restored = build()
        restored.load_state_dict(state)
        equal_tensors(self, restored.state_dict(), state)
        self.assertEqual(state['adversarial_input_identity']['architecture'], 'latent_spatial_844')
        with self.assertRaises(ValueError):
            restored.load_state_dict(build_direct(config(), torch.device('cpu')).state_dict())
        for kind in ('missing_identity', 'old_grids', 'RGB'):
            changed = copy.deepcopy(state)
            if kind == 'missing_identity':
                changed.pop('adversarial_input_identity')
            elif kind == 'old_grids':
                changed['adversarial_input_identity']['native_feature_grids'] = [[4, 4], [2, 2], [1, 1]]
            else:
                changed['adversarial_input_identity']['input_space'] = 'RGB'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                restored.load_state_dict(changed)
        with self.assertRaises(ValueError):
            build('feature_drift').load_state_dict(state)

    def test_builder_preserves_rng_and_cfg_and_rejects_unrequested_conditions(self):
        cfg = config()
        before, original = torch.get_rng_state().clone(), copy.deepcopy(cfg)
        with patch('vae_imagenet.load_vae', side_effect=AssertionError('VAE loaded')):
            build_adversarial_system(cfg, torch.device('cpu'))
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(cfg, original)
        for update in ({'adversarial_architecture': 'direct'}, {'adversarial_input_space': 'rgb'},
                       {'adversarial_in_channels': 3}, {'use_latent': False},
                       {'historical_gen_replay': True}, {'adversarial_ema_decay': .99},
                       {'adversarial_structure_weight': 1.}, {'adversarial_base_channels': 128},
                       {'adversarial_drift_weight': .1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                build_adversarial_system(cfg | update, torch.device('cpu'))



from experiments.mae import latent_cache as cache
from train.train_data import _LatentCacheDataset, create_imagenet_split


class PackedCacheChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "packed"
        self.ptroot = Path(self.tmp.name) / "original"
        self.root.mkdir()
        self.counts = {"train": 5, "val": 2}
        self.count_patch = patch.object(cache, "EXPECTED_COUNTS", self.counts)
        self.count_patch.start()
        self.index = {"schema": 1, "expected_counts": self.counts, "chunk_size": 3,
                      "raw_root": str(Path(self.tmp.name) / "raw"), "splits": {}}
        self.arrays = {}
        for split, count in self.counts.items():
            dest = self.root / split
            (dest / "chunks").mkdir(parents=True)
            labels = np.array([0] * ((count + 1) // 2) + [1] * (count // 2), dtype=np.int64)
            np.save(dest / "labels.npy", labels)
            offsets = [0]
            with open(dest / "paths.txt", "wb") as paths:
                for i, label in enumerate(labels):
                    paths.write(f"class{label}/{i}.JPEG\n".encode())
                    offsets.append(paths.tell())
            np.save(dest / "path_offsets.npy", np.asarray(offsets, dtype=np.int64))
            self.index["splits"][split] = {
                "count": count, "classes": [f"class{i}" for i in range(1000)],
                "class_to_idx": {f"class{i}": i for i in range(1000)},
                "index_sha256": {name: cache._sha(dest / name) for name in
                                  ("labels.npy", "paths.txt", "path_offsets.npy")}}
            values = np.empty((count, 2, 4, 32, 32), np.float32)
            for i, label in enumerate(labels):
                values[i, 0] = i * 100
                values[i, 1] = i * 100 + 17
                pdest = self.ptroot / split / f"class{label}"
                pdest.mkdir(parents=True, exist_ok=True)
                torch.save({"moments": values[i, 0], "moments_flip": values[i, 1]}, pdest / f"{i}.pt")
            self.arrays[split] = values
        cache._json(self.root / "index_manifest.json", self.index)
        cache._json(self.root / "encoding_recipe.json", {"test": True, "posterior": "sample", "scale": .18215})
        self.recipe_sha = cache._sha(self.root / "encoding_recipe.json")

    def tearDown(self):
        self.count_patch.stop()
        self.tmp.cleanup()

    def write_chunks(self, exclude=None):
        for split in self.counts:
            for number, start, end in cache._chunks(self.index, split):
                if (split, number) == exclude:
                    continue
                path, marker = cache._chunk_paths(self.root, split, number)
                np.save(path, self.arrays[split][start:end])
                cache._json(marker, {"range": [start, end], "recipe_sha256": self.recipe_sha,
                                     "bytes": path.stat().st_size, "sha256": cache._sha(path)})

    def complete(self):
        self.write_chunks()
        cache.finalize(self.root)

    def test_partial_cache_and_missing_chunk_fail_closed(self):
        self.write_chunks(exclude=("train", 1))
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            cache.PackedLatentDataset(self.root, "train")
        with self.assertRaisesRegex(RuntimeError, "partial cache"):
            cache.finalize(self.root)
        self.assertFalse((self.root / "complete_manifest.json").exists())

    def test_full_checksum_validation_and_truncation_gate(self):
        self.complete()
        cache.validate_complete(self.root)
        path, _ = cache._chunk_paths(self.root, "train", 0)
        with open(path, "r+b") as handle:
            handle.seek(-4, 2)
            handle.write(b"fail")
        with self.assertRaisesRegex(RuntimeError, "checksum differs"):
            cache.finalize(self.root)
        with open(path, "r+b") as handle:
            handle.truncate(256)
        with self.assertRaisesRegex(RuntimeError, "chunk changed"):
            cache.validate_complete(self.root)

    def test_item_sampling_matches_pt_and_preserves_rng_state(self):
        self.complete()
        for split in self.counts:
            expected = _LatentCacheDataset(str(self.ptroot / split))
            actual = cache.PackedLatentDataset(self.root, split, max_open_chunks=1)
            for seed in range(8):
                for item in range(len(expected)):
                    torch.manual_seed(seed)
                    x, label = expected[item]
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(seed)
                    y, got_label = actual[item]
                    self.assertEqual(label, got_label)
                    np.testing.assert_array_equal(x, y)
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                    self.assertTrue(y.flags.writeable)

    def test_loader_sampler_worker_seed_and_collation_equivalence(self):
        self.complete()
        hook = cache.make_create_imagenet_split(create_imagenet_split, self.root)
        for workers in (0, 2):
            for split in self.counts:
                for rank in (0, 1):
                    common = dict(imagenet_path="", resolution=256, use_latent=True,
                                  use_cache=True, batch_size=2, split=split,
                                  num_workers=workers, pin_memory=False,
                                  distributed=True, rank=rank, world_size=2,
                                  persistent_workers=False)
                    expected, exp_pre, _ = create_imagenet_split(cache_path=str(self.ptroot), **common)
                    actual, act_pre, _ = hook(cache_path=str(self.root), **common)
                    for epoch in (0, 7):
                        expected.sampler.set_epoch(epoch)
                        actual.sampler.set_epoch(epoch)
                        self.assertEqual(list(expected.sampler), list(actual.sampler))
                        torch.manual_seed(991)
                        exp_batches = [exp_pre(batch) for batch in expected]
                        torch.manual_seed(991)
                        act_batches = [act_pre(batch) for batch in actual]
                        self.assertEqual(len(exp_batches), len(act_batches))
                        for exp, act in zip(exp_batches, act_batches):
                            self.assertTrue(torch.equal(exp["images"], act["images"]))
                            self.assertTrue(torch.equal(exp["labels"], act["labels"]))


class _TinyLatentGenerator(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.images = torch.nn.Parameter(torch.randn(4, 4, 32, 32) * .2)

    def forward(self, labels, cfg_scale=None, train=True):
        return {'samples': self.images[:labels.numel()] + .01 * labels[:, None, None, None]}


class _TinyFrozenMAE(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(4, 4, 1).requires_grad_(False)

    def get_activations(self, images, return_stage_features=False, **kwargs):
        if return_stage_features:
            raise AssertionError('Disabled structure teacher must not request DINO terminal maps')
        projected = self.projection(images)
        features = {f'stage{stage}': F.adaptive_avg_pool2d(projected, side).flatten(2).transpose(1, 2)
                    for stage, side in ((3, 4), (4, 2))}
        features['norm_x'] = F.adaptive_avg_pool2d(images, 2).flatten(2).transpose(1, 2)
        return features


def _tiny_mae_inputs(mode):
    from experiments.mae import runtime
    from train_imagenet_gen import load_yaml_config

    cfg = load_yaml_config(runtime.REPO / 'experiments' / 'mae' / 'configs' / f'{mode}.yaml')
    cfg.update(gen_per_label=2, feature_use_bf16=False, feature_use_remat=False,
               num_classes=2, compute_wpos_stats=False, global_scale_stats=False,
               global_fnorm_stats=False, feature_microbatch_size=2,
               adversarial_d_chunk_size=2, adversarial_g_chunk_size=2,
               adversarial_samples_per_class=2, use_wandb=False)
    rng = torch.Generator().manual_seed(781)
    return dict(labels=torch.tensor([0, 1]),
                pos_samples=torch.randn(2, 2, 4, 32, 32, generator=rng),
                neg_samples=torch.randn(2, 1, 4, 32, 32, generator=rng),
                historical_samples=None, device=torch.device('cpu'), step=0, cfg=cfg)


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_full_trainer_modes_run_without_structure_teacher_and_keep_metric_domains(self):
        import io
        from types import SimpleNamespace
        from experiments.mae import runtime
        import train_imagenet_gen as trainer

        for mode in ('control', 'raw_gan', 'feature_drift'):
            with self.subTest(mode=mode):
                torch.manual_seed(43)
                generator, mae = _TinyLatentGenerator(), _TinyFrozenMAE()
                initial = generator.images.detach().clone()
                optimizer = torch.optim.AdamW(generator.parameters(), lr=4e-4, betas=(.9, .95), weight_decay=0.)
                inputs = _tiny_mae_inputs(mode)
                local = SimpleNamespace(**trainer.__dict__)
                runtime.install(local, runtime.resolve_config_paths(inputs['cfg'], environ={}))
                system = local.build_adversarial_system(inputs['cfg'], torch.device('cpu'))
                with patch('vae_imagenet.load_vae', side_effect=AssertionError('VAE loaded')), \
                     patch('vae_imagenet.vae_decode', side_effect=AssertionError('VAE decoded')):
                    loss, metrics, _ = local.train_step(generator, mae, optimizer, adversarial_system=system, **inputs)
                self.assertTrue(torch.isfinite(loss))
                self.assertFalse(torch.equal(initial, generator.images))
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in mae.parameters()))
                if mode == 'control':
                    self.assertIsNone(system)
                    continue
                self.assertEqual(metrics['adversarial/r1_applied'], 1.)
                self.assertEqual(metrics['adversarial/d_updated'], 1.)
                self.assertEqual(metrics['adversarial/structure_loss'], 0.)
                self.assertGreater(metrics['adversarial/auxiliary_latent_grad_norm'], 0.)
                self.assertIn('adversarial/mae_latent_grad_norm', metrics)
                self.assertIn('adversarial/auxiliary_to_mae_latent_grad_ratio', metrics)
                self.assertNotIn('adversarial/dino_pixel_grad_norm', metrics)
                wanted = 'raw_gan_loss' if mode == 'raw_gan' else 'feature_drift_loss'
                forbidden = 'feature_drift_loss' if mode == 'raw_gan' else 'raw_gan_loss'
                self.assertIn('adversarial/' + wanted, metrics)
                self.assertNotIn('adversarial/' + forbidden, metrics)
                equal_tensors(self, system.online.state_dict(), system.target.state_dict())
                stream = io.BytesIO()
                torch.save(system.state_dict(), stream)
                stream.seek(0)
                restored = local.build_adversarial_system(inputs['cfg'], torch.device('cpu'))
                restored.load_state_dict(torch.load(stream, weights_only=False))
                equal_tensors(self, restored.state_dict(), system.state_dict())

    def test_portable_configs_preserve_comparison_and_resolve_explicit_overrides(self):
        from experiments.mae import runtime
        from train_imagenet_gen import load_yaml_config

        configs = {}
        for mode in ('control', 'raw_gan', 'feature_drift'):
            cfg = load_yaml_config(runtime.REPO / 'experiments' / 'mae' / 'configs' / f'{mode}.yaml')
            before = copy.deepcopy(cfg)
            relocated = runtime.resolve_config_paths(cfg, repo_root='/tmp/relocated-idrift', environ={})
            self.assertEqual(cfg, before)
            runtime.validate_recipe(relocated)
            for key in runtime.PATH_ENVIRON:
                self.assertTrue(relocated[key].startswith('/tmp/relocated-idrift/'))
            overridden = runtime.resolve_config_paths(cfg, environ={
                'IDRIFT_MAE_CACHE': '/tmp/cache', 'IDRIFT_MAE_CHECKPOINT': '/tmp/mae.pt',
                'IDRIFT_MAE_VAE': '/tmp/vae', 'IDRIFT_IMAGENET_PATH': '/tmp/images'})
            self.assertEqual(overridden['feature_checkpoint'], overridden['mae_checkpoint'])
            self.assertEqual(overridden['_raw']['env']['cache_path'], '/tmp/cache')
            self.assertEqual(overridden['_raw']['feature']['mae_checkpoint'], '/tmp/mae.pt')
            self.assertEqual(overridden['_raw']['feature']['feature_vae_model_id'], '/tmp/vae')
            configs[mode] = cfg
        raw, feature, control = (configs[m] for m in ('raw_gan', 'feature_drift', 'control'))
        self.assertEqual(raw['adversarial_loss_weight'], .15)
        self.assertEqual(feature['adversarial_drift_weight'], .1)
        self.assertEqual(control['adversarial_mode'], 'none')
        # These are the intended three-way scientific differences; all other
        # flattened settings (including loader, G, MAE, drift and eval) match.
        excluded = {'_raw', 'name', 'adversarial_mode', 'adversarial_loss_weight',
                    'adversarial_drift_weight', 'adversarial_input_space',
                    'adversarial_in_channels', 'adversarial_architecture',
                    'latent_rgb_vae_weights_sha256', 'latent_rgb_decode_chunk_size'}
        common = lambda cfg: {k: v for k, v in cfg.items() if k not in excluded}
        self.assertEqual(common(raw), common(feature))
        self.assertEqual(common(raw), common(control))

    def test_hooks_use_local_cache_builder_and_rename_only_metric_labels(self):
        from types import SimpleNamespace
        from experiments.mae import runtime
        from experiments.mae.latent_spatial_gan import build_adversarial_system as local_build
        from train_imagenet_gen import load_yaml_config

        path = runtime.REPO / 'experiments' / 'mae' / 'configs' / 'feature_drift.yaml'
        cfg = runtime.resolve_config_paths(load_yaml_config(path), environ={})
        sentinel = object()
        trainer = SimpleNamespace(
            load_yaml_config=load_yaml_config, create_imagenet_split=lambda **kwargs: sentinel,
            train_step=lambda: (sentinel, {'adversarial/dino_pixel_grad_norm': 2., 'loss': 3.}, sentinel),
            Logger=object)
        runtime.install(trainer, cfg)
        self.assertIs(trainer.build_adversarial_system, local_build)
        loss, metrics, extras = trainer.train_step()
        self.assertIs(loss, sentinel)
        self.assertIs(extras, sentinel)
        self.assertEqual(metrics, {'adversarial/mae_latent_grad_norm': 2., 'loss': 3.})
        self.assertIs(trainer.create_imagenet_split(use_cache=False), sentinel)
        self.assertIsNone(trainer.build_adversarial_system({'adversarial_mode': 'none'}, torch.device('cpu')))

    def test_cpu_affinity_is_optional_and_rejects_unallocated_cpus(self):
        from experiments.mae import runtime

        with patch.object(runtime.os, 'sched_setaffinity') as setter:
            runtime.configure_affinity({})
            setter.assert_not_called()
            with patch.object(runtime.os, 'sched_getaffinity', return_value={2, 3, 4}):
                runtime.configure_affinity({'MAE256_RANK_CPUSETS': '[[2, 3], [4]]', 'LOCAL_RANK': '1'})
                setter.assert_called_once_with(0, {4})
                with self.assertRaises(ValueError):
                    runtime.configure_affinity({'MAE256_RANK_CPUSETS': '[[9]]'})


class LiveSourceParity(unittest.TestCase):
    """Optional independent parity against the original, unmodified live files.

    IDRIFT_MAE_REFERENCE_ROOT enables this audit on the source instance. Normal
    portable test runs do not require any original experiment directory.
    """
    @classmethod
    def setUpClass(cls):
        import importlib.util
        import os

        reference_root = os.environ.get('IDRIFT_MAE_REFERENCE_ROOT')
        if not reference_root:
            raise unittest.SkipTest('Set IDRIFT_MAE_REFERENCE_ROOT for optional live-source parity')
        cls.reference_root = Path(reference_root)
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        direct = load('_live_mae_direct', cls.reference_root / 'latent_direct_gan.py')
        with patch.dict(sys.modules, {'latent_direct_gan': direct}):
            cls.reference = load('_live_mae_spatial', cls.reference_root / 'latent_spatial_gan.py')
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_literal_live_forward_gradient_D_step_optimizer_R1_and_checkpoint_parity(self):
        from train_imagenet_gen import compute_drift_loss_from_features

        for mode in ('raw_gan', 'feature_drift'):
            with self.subTest(mode=mode):
                cfg = config(mode)
                old = self.reference.build_adversarial_system(cfg, torch.device('cpu'))
                new = build_adversarial_system(cfg, torch.device('cpu'))
                equal_tensors(self, old.state_dict(), new.state_dict())
                real, fake, labels = batch()
                for step in (0, 1, 16):
                    # Compare every native activation and pooled feature/logit,
                    # G input gradient, D optimizer slot and target hard copy.
                    equal_tensors(self, old.target._maps(fake), new.target._maps(fake))
                    losses, gradients = [], []
                    for system in (old, new):
                        generated = fake.detach().clone().requires_grad_()
                        if mode == 'raw_gan':
                            output = system.target_logits(generated, labels)
                            loss = .15 * F.softplus(-output).mean()
                        else:
                            output = system.target_features(generated)
                            with torch.no_grad():
                                positive = system.target_features(real)
                            loss, _ = compute_drift_loss_from_features(
                                output, positive, B=2, G=2, P=2, N=0,
                                neg_feats=None, weight_neg=None,
                                global_scale_stats=False, global_fnorm_stats=False)
                            loss = .1 * loss
                        losses.append((output, loss.detach()))
                        gradients.append(torch.autograd.grad(loss, generated)[0])
                    equal_tensors(self, losses[0], losses[1])
                    equal_tensors(self, gradients[0], gradients[1])
                    a = old.discriminator_step(real, fake, labels, labels, step)
                    b = new.discriminator_step(real, fake, labels, labels, step)
                    self.assertEqual(a, b)
                    equal_tensors(self, old.state_dict(), new.state_dict())
                # State produced by the live class loads into the port unchanged.
                new.load_state_dict(copy.deepcopy(old.state_dict()))
                equal_tensors(self, old.state_dict(), new.state_dict())

    def test_all_live_config_fields_match_except_explicit_path_relocation(self):
        from experiments.mae import runtime
        from train_imagenet_gen import load_yaml_config

        for mode in ('control', 'raw_gan', 'feature_drift'):
            relative = 'configs/control.yaml' if mode == 'control' else f'latent_spatial/configs/{mode}.yaml'
            old = load_yaml_config(self.reference_root / relative)
            new = load_yaml_config(runtime.REPO / 'experiments' / 'mae' / 'configs' / f'{mode}.yaml')
            for key in set(old) | set(new):
                if key not in {'_raw', *runtime.PATH_ENVIRON}:
                    with self.subTest(mode=mode, field=key):
                        self.assertEqual(old[key], new[key])

    def test_full_G_D_step_matches_live_patched_trainer_for_all_three_modes(self):
        import importlib.util
        import os
        from types import SimpleNamespace
        from experiments.mae import runtime
        import train_imagenet_gen as trainer

        source_repo = Path(os.environ.get('IDRIFT_MAE_REFERENCE_REPO', self.reference_root.parent / 'I-Drift'))
        source_path = source_repo / 'train_imagenet_gen.py'
        if not source_path.is_file():
            self.skipTest('Original trainer absent; set IDRIFT_MAE_REFERENCE_REPO')
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        original_trainer = load('_live_mae_trainer', source_path)
        bridge = load('_live_mae_bridge', self.reference_root / 'trainer_bridge.py')
        original_step = bridge.patched_train_step(original_trainer)
        for mode in ('control', 'raw_gan', 'feature_drift'):
            with self.subTest(mode=mode):
                torch.manual_seed(43)
                generator, mae = _TinyLatentGenerator(), _TinyFrozenMAE()
                inputs = _tiny_mae_inputs(mode)
                local = SimpleNamespace(**trainer.__dict__)
                runtime.install(local, runtime.resolve_config_paths(inputs['cfg'], environ={}))
                records = []
                for train_fn, build_fn in ((original_step, self.reference.build_adversarial_system),
                                           (local.train_step, local.build_adversarial_system)):
                    gen = copy.deepcopy(generator)
                    optimizer = torch.optim.AdamW(gen.parameters(), lr=4e-4, betas=(.9, .95), weight_decay=0.)
                    system = build_fn(inputs['cfg'], torch.device('cpu'))
                    torch.manual_seed(917)
                    output = train_fn(gen, mae, optimizer, adversarial_system=system, **inputs)
                    metrics = output[1]
                    for old, new in runtime.METRIC_NAMES.items():
                        if old in metrics:
                            metrics[new] = metrics.pop(old)
                    # The integrated trainer additionally logs the auxiliary
                    # replay switch and count. No historical samples are used
                    # by any of these MAE presets; these are logging-only keys.
                    if 'adversarial/history_count' in metrics:
                        self.assertEqual(metrics.pop('adversarial/history_count'), 0.)
                        self.assertEqual(metrics.pop('adversarial/replay_enabled'), 1.)
                    records.append((output, gen.state_dict(), optimizer.state_dict(),
                                    None if system is None else system.state_dict(), torch.get_rng_state()))
                equal_tensors(self, records[0], records[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
