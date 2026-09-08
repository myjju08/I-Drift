import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from scripts.prepare_dino_rf_tuning_data import (
    load_real_pixels,
    publish_manifest,
    quantize_fake_pixels,
    select_real_examples,
    stage_fake_split,
    stage_real_split,
)
from train.train_data import _build_transforms


class _TinyGenerator(torch.nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        self.scales = []

    def forward(self, labels, cfg_scale, train=False):
        self.scales.append(cfg_scale.detach().clone())
        shape = (len(labels), 3, self.resolution, self.resolution)
        # Fractional values deliberately preserve a difference between raw and
        # uint8-normalized samples, including a few out-of-range pixels.
        samples = torch.rand(shape, device=labels.device) * 2.4 - 1.2
        return {"samples": samples}


class DinoRFTuningDataTest(unittest.TestCase):
    def _make_imagenet(self, root):
        for split in ("train", "val"):
            for label in range(2):
                directory = root / split / f"n{label:08d}"
                directory.mkdir(parents=True)
                for index in range(6):
                    values = np.arange(11 * 17 * 3, dtype=np.uint16)
                    values = ((values * 37 + index * 29 + label * 17) % 256).astype(np.uint8)
                    Image.fromarray(values.reshape(11, 17, 3)).save(directory / f"{index}.png")

    def _selection(self, root):
        return select_real_examples(root, train_per_class=3, validation_per_class=2,
                                    train_seed=43, validation_seed=44, num_classes=2)

    def test_balanced_deterministic_selection_uses_disjoint_train_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_imagenet(root)
            classes, records = self._selection(root)
            self.assertEqual((classes, records), self._selection(root))
            self.assertEqual(classes, ["n00000000", "n00000001"])
            selected = {}
            for split, amount in (("train", 3), ("validation", 2)):
                selected[split] = {item["path"] for item in records[split]}
                self.assertTrue(all(item["relative_path"].startswith("train/") for item in records[split]))
                self.assertEqual(np.bincount([item["label"] for item in records[split]]).tolist(), [amount] * 2)
            self.assertFalse(selected["train"] & selected["validation"])

    def test_real_pixels_match_production_transform_and_preserve_cpu_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_imagenet(root)
            _, records = self._selection(root)
            record = records["train"][0]
            torch.random.default_generator.manual_seed(918)
            original_rng = torch.get_rng_state().clone()
            pixels, digest = load_real_pixels(record, resolution=8)
            self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(record["transform_seed"])
                with Image.open(record["path"]) as image:
                    expected = _build_transforms(8, False, "train", return_uint8=True)(image.convert("RGB"))
            self.assertTrue(np.array_equal(pixels, expected.numpy()))
            self.assertEqual(digest, hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest())

    def test_quantization_clamps_and_truncates_and_rejects_nonfinite(self):
        values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
        expected = torch.tensor([0, 0, 127, 255, 255], dtype=torch.uint8)
        self.assertTrue(torch.equal(quantize_fake_pixels(values)[0, 0, 0], expected))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(FloatingPointError):
                quantize_fake_pixels(torch.full((1, 3, 1, 1), invalid))

    def test_end_to_end_staging_records_cfg_raw_diagnostics_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_imagenet(root / "source")
            _, records = self._selection(root / "source")
            output = root / "staged"
            manifest = {"schema_version": 1}
            expected_hashes = {}
            for split, seed in (("train", 43), ("validation", 44)):
                directory = output / split
                manifest[split] = stage_real_split(directory, records[split], 8)
                generator = _TinyGenerator(8)
                metadata = stage_fake_split(directory, generator, seed=seed, cfg_scale=None,
                                            batch_size=4, device=torch.device("cpu"), resolution=8,
                                            save_raw_fake=True)
                manifest[split].update(metadata)
                quantized = np.load(manifest[split]["fake_images"])
                raw = np.load(metadata["raw_fake_images"])
                scales = np.load(metadata["fake_cfg_scales"])
                self.assertEqual(raw.dtype, np.float16)
                self.assertEqual(quantized.dtype, np.uint8)
                self.assertTrue(np.array_equal(scales, torch.cat(generator.scales).numpy()))
                self.assertTrue(np.all((scales >= 1) & (scales < 4)))
                self.assertGreater(np.unique(scales).size, 1)
                self.assertEqual(metadata["raw_fake_usage"], "diagnostic_only_not_primary_training")
                self.assertGreater(metadata["raw_fake_outside_minus1_plus1_fraction"], 0)
                expected_hashes[split] = metadata["fake_pixels_sha256"]
                sources = [json.loads(line) for line in Path(manifest[split]["real_sources"]).read_text().splitlines()]
                self.assertEqual(len(sources), len(records[split]))
                self.assertTrue(all("source_sha256" in source for source in sources))
            self.assertFalse((output / "manifest.json").exists())
            result = publish_manifest(output, manifest)
            self.assertEqual(json.loads(result.read_text()), manifest)
            self.assertFalse((output / "manifest.json.tmp").exists())
            self.assertNotEqual(expected_hashes["train"], expected_hashes["validation"])
            with self.assertRaises(FileExistsError):
                publish_manifest(output, manifest)

    def test_fixed_seed_batching_reproduces_fake_pixels_and_cfg_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            hashes = []
            for name in ("first", "second"):
                directory = Path(temporary) / name
                directory.mkdir()
                np.save(directory / "labels.npy", np.array([0, 1, 0, 1, 0], dtype=np.int64))
                result = stage_fake_split(directory, _TinyGenerator(3), seed=43, cfg_scale=None,
                                          batch_size=3, device=torch.device("cpu"), resolution=3,
                                          save_raw_fake=False)
                hashes.append((result["fake_pixels_sha256"], result["fake_cfg_scales_sha256"]))
                self.assertNotIn("raw_fake_images", result)
                self.assertFalse((directory / "fake_raw_float16.npy").exists())
            self.assertEqual(hashes[0], hashes[1])

    def test_invalid_primary_dtype_cannot_publish_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real, fake, labels = root / "real.npy", root / "fake.npy", root / "labels.npy"
            np.save(real, np.zeros((2, 3, 4, 4), dtype=np.uint8))
            np.save(fake, np.zeros((2, 3, 4, 4), dtype=np.float16))
            np.save(labels, np.zeros(2, dtype=np.int64))
            spec = {"real_images": str(real), "fake_images": str(fake),
                    "real_labels": str(labels), "fake_labels": str(labels)}
            with self.assertRaisesRegex(ValueError, "uint8"):
                publish_manifest(root, {"train": spec, "validation": spec})
            self.assertFalse((root / "manifest.json").exists())
            self.assertFalse((root / "manifest.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
