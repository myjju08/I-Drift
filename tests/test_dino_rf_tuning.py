"""Isolation and checkpoint checks for offline DINO real/fake tuning."""
import copy
import datetime
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from models.dino_rf_tuning import DinoRealFakeTuner, TRAINABLE_BLOCKS, atomic_torch_save
from scripts.tune_dino_real_fake import MatchedPairDataset, decode_images, real_fake_auc, reconcile_selected_export, train_update


class _Block(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, value):
        return torch.relu(value + self.bn(self.conv(value)))


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        self.layer1 = nn.Sequential(_Block())
        self.layer2 = nn.Sequential(_Block())
        self.layer3 = nn.Sequential(*[_Block() for _ in range(6)])
        self.layer4 = nn.Sequential(*[_Block() for _ in range(3)])
        self.fc = nn.Identity()


def _tuner():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        model = DinoRealFakeTuner(_Backbone(), num_classes=2, stage_channels=(4, 4))
    return model


def _inputs():
    rng = torch.Generator().manual_seed(25)
    return torch.rand(4, 3, 16, 16, generator=rng) * 2 - 1, torch.rand(4, 3, 16, 16, generator=rng) * 2 - 1, torch.tensor([0, 0, 1, 1])


def _distributed_probe_worker(rank, port):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank,
                            timeout=datetime.timedelta(seconds=45))
    try:
        model = _tuner()
        real, fake, labels = _inputs()
        real = real + rank * 0.05
        sums, counts = model.scale_statistics(real)
        dist.all_reduce(sums)
        dist.all_reduce(counts)
        model.set_scales(sums, counts)
        wrapper = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
        args = types.SimpleNamespace(microbatch_pairs=2, bf16=False, preservation_weight=10.0, max_grad_norm=1.0)
        initial_student = copy.deepcopy(model.student.state_dict())
        optimizer = torch.optim.AdamW(model.heads.parameters(), lr=0.001)
        train_update(wrapper, model, optimizer, (real, fake, labels), args, torch.device("cpu"), head_only=True)
        for key, value in model.student.state_dict().items():
            torch.testing.assert_close(value, initial_student[key], rtol=0, atol=0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        for _ in range(2):
            train_update(wrapper, model, optimizer, (real, fake, labels), args, torch.device("cpu"))
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class DinoRealFakeTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_only_allowed_non_bn_student_weights_and_head_change(self):
        model = _tuner().train()
        real, fake, labels = _inputs()
        sums, counts = model.scale_statistics(real)
        model.set_scales(sums, counts)
        original = copy.deepcopy(model.student.state_dict())
        teacher = copy.deepcopy(model.teacher.state_dict())
        heads = copy.deepcopy(model.heads.state_dict())
        real.requires_grad_()
        fake.requires_grad_()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        output = model(real, fake, labels)
        self.assertEqual(float(output["preservation"]), 0.0)
        output["loss"].backward()
        self.assertIsNone(real.grad)
        self.assertIsNone(fake.grad)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        optimizer.step()
        changed = {key for key, value in model.student.state_dict().items() if not torch.equal(value, original[key])}
        self.assertTrue(changed)
        self.assertTrue(all(any(key.startswith(block + ".") for block in TRAINABLE_BLOCKS) for key in changed))
        self.assertFalse(any(".bn." in key for key in changed))
        self.assert_state_equal(model.teacher.state_dict(), teacher)
        self.assertTrue(any(not torch.equal(value, heads[key]) for key, value in model.heads.state_dict().items()))
        self.assertTrue(all(not module.training for module in model.student.modules() if isinstance(module, nn.BatchNorm2d)))

    def test_fixed_scales_are_additive_and_preservation_detects_absolute_rescaling(self):
        model = _tuner()
        real, fake, labels = _inputs()
        full_sum, full_count = model.scale_statistics(real)
        first_sum, first_count = model.scale_statistics(real[:2])
        last_sum, last_count = model.scale_statistics(real[2:])
        torch.testing.assert_close(full_sum, first_sum + last_sum)
        torch.testing.assert_close(full_count, first_count + last_count)
        model.set_scales(full_sum, full_count)
        fixed = model.feature_scales.clone()
        with torch.no_grad():
            model.student.layer4[2].conv.weight.mul_(1.5)
        output = model(real, fake, labels)
        self.assertGreater(float(output["preservation"]), 0.0)
        torch.testing.assert_close(model.feature_scales, fixed, rtol=0, atol=0)

    def test_head_probe_has_no_student_gradient_and_is_resettable(self):
        model = _tuner().train()
        real, fake, labels = _inputs()
        model.set_scales(*model.scale_statistics(real))
        student = copy.deepcopy(model.student.state_dict())
        heads = copy.deepcopy(model.heads.state_dict())
        optimizer = torch.optim.SGD(model.heads.parameters(), lr=0.1)
        model(real, fake, labels, head_only=True)["loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.student.parameters()))
        optimizer.step()
        self.assert_state_equal(model.student.state_dict(), student)
        model.heads.load_state_dict(heads)
        self.assert_state_equal(model.heads.state_dict(), heads)

    def test_checkpoint_resume_and_flat_frozen_export(self):
        model = _tuner()
        real, fake, labels = _inputs()
        model.set_scales(*model.scale_statistics(real))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
        model(real, fake, labels)["loss"].backward()
        optimizer.step()
        restored = _tuner()
        restored_optimizer = torch.optim.Adam([p for p in restored.parameters() if p.requires_grad], lr=0.001)
        with tempfile.TemporaryDirectory(prefix="dino-tuning-test-") as directory:
            checkpoint = Path(directory) / "state.pt"
            atomic_torch_save({"tuning": model.tuning_state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
            state = torch.load(checkpoint, weights_only=False)
            restored.load_tuning_state_dict(state["tuning"])
            restored_optimizer.load_state_dict(state["optimizer"])
            for current, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
                current_optimizer.zero_grad(set_to_none=True)
                current(real, fake, labels)["loss"].backward()
                current_optimizer.step()
            self.assert_state_equal(model.student.state_dict(), restored.student.state_dict())
            self.assert_state_equal(model.heads.state_dict(), restored.heads.state_dict())
            export = Path(directory) / "dino_tuned.pth"
            restored.export_backbone(export)
            flat = torch.load(export, weights_only=True)
            frozen = _Backbone().eval().requires_grad_(False)
            frozen.load_state_dict(flat, strict=True)
            # A frozen exported encoder still carries gradients into G images.
            pixels = real.clone().requires_grad_()
            frozen.conv1(pixels).square().mean().backward()
            self.assertGreater(float(pixels.grad.abs().sum()), 0.0)
            self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))

    def test_dataset_matches_classes_and_keeps_raw_float_validation_separate(self):
        with tempfile.TemporaryDirectory(prefix="dino-pairs-test-") as directory:
            root = Path(directory)
            real = np.full((2, 3, 16, 16), 255, dtype=np.uint8)
            fake = np.zeros_like(real)
            raw = np.full(real.shape, 1.125, dtype=np.float16)
            np.save(root / "real.npy", real)
            np.save(root / "fake.npy", fake)
            np.save(root / "raw.npy", raw)
            np.save(root / "real_labels.npy", np.array([1, 0]))
            np.save(root / "fake_labels.npy", np.array([0, 1]))
            entry = {"real_images": "real.npy", "fake_images": "fake.npy", "raw_fake_images": "raw.npy",
                     "real_labels": "real_labels.npy", "fake_labels": "fake_labels.npy"}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "train": entry, "validation": entry}))
            pairs = MatchedPairDataset(manifest)
            self.assertEqual(pairs.real_indices.tolist(), [1, 0])
            real_tensor, fake_tensor, label = pairs[0]
            self.assertEqual(label, 0)
            self.assertEqual(float(real_tensor.min()), 1.0)
            self.assertEqual(float(fake_tensor.max()), -1.0)
            raw_pairs = MatchedPairDataset(manifest, "validation", raw_fake=True)
            self.assertEqual(float(raw_pairs[0][1].max()), 1.125)
            torch.testing.assert_close(decode_images(fake), torch.full(fake.shape, -1.0))

    def test_auc_uses_all_examples_and_average_tie_ranks(self):
        self.assertEqual(real_fake_auc([1, 2], [-1, 0]), 1.0)
        self.assertEqual(real_fake_auc([-1, 0], [1, 2]), 0.0)
        self.assertEqual(real_fake_auc([0, 0], [0, 0]), 0.5)
        self.assertEqual(real_fake_auc([0, 1], [0, 0]), 0.75)

    def test_resume_repairs_interrupted_best_export_publication(self):
        model = _tuner()
        real, _, _ = _inputs()
        model.set_scales(*model.scale_statistics(real))
        selection = {"step": 2, "quantized": {"loss": 0.5}}
        state = {"step": 2, "metadata": {"teacher_sha256": "example"},
                 "best_selection": selection, "tuning": model.tuning_state_dict()}
        with tempfile.TemporaryDirectory(prefix="dino-resume-repair-") as directory:
            root = Path(directory)
            atomic_torch_save(state, root / "tuning_latest.pt")
            # Simulate preemption immediately after latest is committed.
            reconcile_selected_export(root, state)
            self.assertEqual((root / "tuning_best.pt").stat().st_ino, (root / "tuning_latest.pt").stat().st_ino)
            self.assert_state_equal(torch.load(root / "dino_tuned.pth", weights_only=True), model.student.state_dict())
            later = {**state, "step": 3}
            atomic_torch_save(later, root / "tuning_latest.pt")
            (root / "dino_tuned.pth").unlink()
            reconcile_selected_export(root, later)
            self.assert_state_equal(torch.load(root / "dino_tuned.pth", weights_only=True), model.student.state_dict())
            (root / "tuning_best.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                reconcile_selected_export(root, later)

    def test_two_rank_head_probe_then_microbatch_tuning_remain_synchronized(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("Gloo is unavailable")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_distributed_probe_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
