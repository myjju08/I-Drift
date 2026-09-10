"""CPU checks for the portable DINO presets, live replay scope and lossless I/O."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.dino import runtime
from experiments.dino.train import PRESETS, ROOT, load_config, preflight
from experiments.dino.io.compact_imagefolder import PackedImageSamples


class TinyGenerator(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.images = torch.nn.Parameter(torch.randn(4, 3, 32, 32) * .2)

    def forward(self, labels, cfg_scale=None, train=True):
        return {'samples': self.images[:labels.numel()] + .01 * labels[:, None, None, None]}


class TinyDino(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, 1).requires_grad_(False)

    def get_activations(self, images, return_stage_features=False, **kwargs):
        projected = self.projection(images)
        maps = {f'layer{stage}': torch.nn.functional.adaptive_avg_pool2d(projected, size)
                for stage, size in ((3, 4), (4, 2))}
        features = {name: value.flatten(2).transpose(1, 2) for name, value in maps.items()}
        features['norm_x'] = torch.nn.functional.adaptive_avg_pool2d(images, 2).flatten(2).transpose(1, 2)
        return (features, maps) if return_stage_features else features


class DinoPresetTests(unittest.TestCase):
    def test_all_presets_pinned_and_source_scientific_settings_preserved(self):
        manifest = json.loads((ROOT / 'provenance.json').read_text())
        for preset in PRESETS:
            with self.subTest(preset=preset):
                cfg = load_config(preset)
                self.assertEqual(cfg['feature_extractor'], 'dino_resnet50')
                self.assertEqual([cfg[k] for k in ('batch_size', 'gen_per_label', 'pos_per_sample', 'neg_per_sample')],
                                 [4, 32, 64, 32])
                self.assertEqual(cfg['historical_gen_replay_ratio'], .35)
                self.assertEqual(cfg['historical_gen_replay_source'], 'frozen_snapshot')
                self.assertEqual(cfg['historical_gen_replay_count'], 16)
                self.assertEqual(cfg['historical_gen_replay_start_generated_epochs'], 10.)
                self.assertEqual(cfg['adversarial_structure_weight'], 0)
                self.assertEqual(cfg['adversarial_ema_decay'], 0)
                self.assertEqual(cfg['adversarial_r1_gamma'], 1)
                self.assertEqual(cfg['adversarial_r1_interval'], 16)
                self.assertEqual(cfg['adversarial_base_channels'], 128 if preset.endswith('d128') else 32)
                self.assertEqual(cfg['adversarial_loss_weight'], .1 if 'raw_gan' in preset else 0)
                self.assertEqual(cfg['adversarial_drift_weight'], 0 if 'raw_gan' in preset else 1)
                path = ROOT / 'configs' / f'{preset}.yaml'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest['presets'][preset]['ported_sha256'])
                self.assertNotIn('/workspace/I-Drift/', json.dumps(cfg))

    def test_paths_resolve_from_repository_and_overrides_update_nested_config(self):
        with tempfile.TemporaryDirectory() as folder:
            override = Path(folder) / 'checkpoint.pth'
            cfg = load_config(feature_checkpoint=override)
        self.assertEqual(cfg['feature_checkpoint'], str(override))
        self.assertEqual(cfg['_raw']['feature']['feature_checkpoint'], str(override))
        self.assertEqual(Path(cfg['temperature_calibration_artifact']).parent, ROOT / 'calibration')

    def test_replay_boundary_and_source_preset_scope(self):
        for preset in PRESETS:
            cfg = load_config(preset)
            report = preflight(cfg, preset=preset, io_backend='standard')
            self.assertEqual(report['generated_per_step'], 256)
            self.assertEqual(report['replay_start_step'], 50046)
            self.assertEqual(report['replay_ratio_before'], 0)
            self.assertEqual(report['replay_ratio_after'], .35)
        self.assertFalse(load_config('dino_only_replay_feature')['adversarial_apply_replay'])
        self.assertTrue(load_config('replay_feature_hardcopy').get('adversarial_apply_replay', True))


class DinoIOTests(unittest.TestCase):
    def test_packed_ring_samples_and_rng_equal_original(self):
        self.assertTrue(runtime.check_factory('numpy')['passed'])

    def test_compact_metadata_preserves_paths_labels_and_order(self):
        samples = [('/tmp/images/n0002/a.jpg', 1), ('/tmp/images/n0001/한글.jpg', 0)]
        packed = PackedImageSamples(samples, root='/tmp/images')
        self.assertEqual(list(packed), samples)
        self.assertEqual(packed[::-1], samples[::-1])
        self.assertEqual(packed[-1], samples[-1])
        with self.assertRaises(IndexError):
            _ = packed[2]

    def test_timing_does_not_change_rng_or_loader_sequence(self):
        class Loader:
            dataset = object()

            def __len__(self):
                return 4

            def __iter__(self):
                return iter(random.random() for _ in range(4))

        saved = random.getstate()
        try:
            random.seed(43)
            loader = Loader()
            expected = [list(loader), list(loader)]
            expected_rng = random.getstate()
            random.seed(43)
            timings = runtime.StageTimings()
            wrapped = runtime.TimedLoader(loader, timings)
            self.assertEqual([list(wrapped), list(wrapped)], expected)
            self.assertEqual(random.getstate(), expected_rng)
            self.assertIs(wrapped.dataset, loader.dataset)
            self.assertGreaterEqual(timings.values['loader_next_seconds'], 0)
        finally:
            random.setstate(saved)

    @unittest.skipUnless(shutil.which('cc') and platform.machine().lower() in {'x86_64', 'amd64'},
                         'Native codec requires an x86 C compiler')
    def test_native_codec_matches_numpy_and_roundtrips_tail_widths(self):
        from experiments.dino.io.build_native import build
        if not (ROOT / 'io/native_color_decode.so').is_file():
            build()
        from experiments.dino.io.native_color_decoder import filter_color_xy, inverse_color_xy
        from experiments.dino.io.color_compressed_memory_bank import ColorCompressedPixelMemoryBank
        source = np.random.default_rng(31)
        reference = ColorCompressedPixelMemoryBank(num_classes=1, max_size=2, predictor='xy', codec_workers=1)
        try:
            for h, w in ((1, 1), (3, 15), (2, 16), (5, 17), (3, 31), (2, 32), (5, 256)):
                pixels = source.integers(0, 256, size=(3, h, w), dtype=np.uint8)
                encoded = filter_color_xy(pixels)
                self.assertTrue(np.array_equal(encoded, reference._filter(pixels)))
                output = np.empty_like(pixels)
                inverse_color_xy(encoded.tobytes(), output)
                self.assertTrue(np.array_equal(output, pixels))
            self.assertTrue(runtime.check_factory('native')['passed'])
        finally:
            reference.suspend_codec_workers()

    def test_wholefile_loader_returns_identical_rgb_pixels(self):
        from PIL import Image
        from torchvision.datasets.folder import pil_loader
        from experiments.dino.io.wholefile_reader import WholeFilePILLoader
        pixels = np.random.default_rng(10).integers(0, 256, size=(19, 21, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.jpg'
            Image.fromarray(pixels).save(path)
            self.assertTrue(np.array_equal(np.asarray(pil_loader(path)), np.asarray(WholeFilePILLoader(pil_loader)(path))))
            # Oversize fallback uses the same original PIL loader.
            self.assertTrue(np.array_equal(np.asarray(pil_loader(path)), np.asarray(WholeFilePILLoader(pil_loader, max_file_bytes=1)(path))))


class DinoReplayObjectiveTests(unittest.TestCase):
    def test_live_scope_keeps_dino_replay_and_excludes_cnn_history_and_weights(self):
        import train_imagenet_gen as trainer
        from models.adversarial_drift import AdversarialDriftSystem
        from models.dino_rf_tuning import module_fingerprint
        prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for cnn_replay in (False, True):
                with self.subTest(cnn_replay=cnn_replay), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(43)
                    generator, teacher = TinyGenerator(), TinyDino()
                    optimizer = torch.optim.SGD(generator.parameters(), lr=.001)
                    system = AdversarialDriftSystem(device='cpu', mode='feature_drift', in_channels=3,
                        num_classes=4, base_channels=4, lr=.001, ema_decay=0, r1_gamma=0,
                        r1_interval=16, structure_weight=0, d_chunk_size=2, g_chunk_size=2, seed=43)
                    rng = torch.Generator().manual_seed(76)
                    inputs = dict(labels=torch.tensor([0, 1]),
                        pos_samples=torch.randn(2, 2, 3, 32, 32, generator=rng),
                        neg_samples=torch.randn(2, 1, 3, 32, 32, generator=rng),
                        historical_samples=torch.randn(2, 1, 3, 32, 32, generator=rng),
                        device=torch.device('cpu'), step=1,
                        cfg=dict(gen_per_label=2, cfg_min=1., cfg_max=2., R_list=[.2],
                            drift_matching='rev-drift', global_scale_stats=False, global_fnorm_stats=False,
                            compute_wpos_stats=False, activation_kwargs={}, adversarial_samples_per_class=2,
                            adversarial_diagnostics_every=1000, historical_gen_replay=True,
                            historical_gen_replay_ratio=.35, adversarial_apply_replay=cnn_replay))
                    before = module_fingerprint(system.target)
                    original_features, original_drift = system.target_features, trainer.compute_drift_loss_from_features
                    reads, calls = [], []

                    def features(images, labels=None):
                        self.assertEqual(module_fingerprint(system.target), before)
                        reads.append((len(images), torch.is_grad_enabled(), images.requires_grad))
                        return original_features(images, labels)

                    def drift(*args, **kwargs):
                        cnn = set(kwargs['gen_feats']) == {'stage2', 'stage3', 'stage4'}
                        with_replay = not cnn or cnn_replay
                        self.assertEqual(kwargs['historical_count'], 1 if with_replay else 0)
                        if with_replay:
                            self.assertIsNotNone(kwargs['historical_feats'])
                            self.assertTrue(torch.allclose(kwargs['weight_gen'], torch.full_like(kwargs['weight_gen'], .65)))
                            self.assertTrue(torch.allclose(kwargs['weight_history'], torch.full_like(kwargs['weight_history'], .7)))
                        else:
                            for key in ('historical_feats', 'weight_gen', 'weight_history'):
                                self.assertIsNone(kwargs[key])
                        calls.append('cnn' if cnn else 'dino')
                        return original_drift(*args, **kwargs)

                    with mock.patch.object(system, 'target_features', side_effect=features), \
                         mock.patch.object(trainer, 'compute_drift_loss_from_features', side_effect=drift), \
                         mock.patch.object(system, 'target_logits', side_effect=AssertionError('Unexpected direct GAN G loss')):
                        loss, metrics, _ = trainer.train_step(generator, teacher, optimizer, adversarial_system=system, **inputs)
                    self.assertTrue(torch.isfinite(loss).item())
                    self.assertGreater(generator.images.grad.abs().sum().item(), 0)
                    self.assertEqual(calls, ['dino', 'cnn'])
                    expected_reads = [(4, False, False), (2, False, False)]
                    if cnn_replay:
                        expected_reads.append((2, False, False))
                    expected_reads.append((4, True, True))
                    self.assertEqual(reads, expected_reads)
                    self.assertEqual(metrics['adversarial/replay_enabled'], float(cnn_replay))
                    self.assertEqual(metrics['adversarial/history_count'], float(cnn_replay))
                    self.assertEqual(module_fingerprint(system.online), module_fingerprint(system.target))
                    self.assertNotIn('adversarial/raw_gan_loss', metrics)
                    self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        finally:
            torch.set_num_threads(prior_threads)


if __name__ == '__main__':
    unittest.main()
