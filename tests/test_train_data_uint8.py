import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train.train_data import (
    _build_transforms,
    create_imagenet_split,
    create_raw_image_io_semaphore,
    infer_latent_cache_format,
)


def _normalized_from_uint8(source: torch.Tensor) -> torch.Tensor:
    return source.float().div(255.0).sub(0.5).div(0.5)


class RawTrainUint8LoaderTest(unittest.TestCase):
    @staticmethod
    def _image() -> Image.Image:
        height, width = 19, 27
        values = np.arange(height * width * 3, dtype=np.uint16)
        values = np.mod(values * 37 + 11, 256).astype(np.uint8)
        return Image.fromarray(values.reshape(height, width, 3), mode="RGB")

    def _assert_transform_equivalent(self, *, use_aug: bool, seed: int) -> None:
        image = self._image()
        legacy_transform = _build_transforms(
            12,
            use_aug=use_aug,
            split="train",
            return_uint8=False,
        )
        direct_transform = _build_transforms(
            12,
            use_aug=use_aug,
            split="train",
            return_uint8=True,
        )

        torch.manual_seed(seed)
        legacy = legacy_transform(image.copy())
        torch.manual_seed(seed)
        direct = direct_transform(image.copy())

        self.assertEqual(direct.dtype, torch.uint8)
        self.assertEqual(tuple(direct.shape), (3, 12, 12))
        self.assertTrue(torch.equal(legacy, _normalized_from_uint8(direct)))

    def test_center_crop_and_flip_match_legacy_float_path_exactly(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                self._assert_transform_equivalent(use_aug=False, seed=seed)

    def test_random_resized_crop_and_flip_match_legacy_float_path_exactly(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                self._assert_transform_equivalent(use_aug=True, seed=seed)

    def test_loader_exposes_prefetch_persistence_and_uint8_preprocess(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class_dir = Path(tmpdir) / "train" / "class0"
            class_dir.mkdir(parents=True)
            self._image().save(class_dir / "sample.png")

            loader, preprocess_fn, _ = create_imagenet_split(
                imagenet_path=tmpdir,
                resolution=12,
                batch_size=1,
                split="train",
                use_aug=False,
                use_latent=False,
                use_cache=False,
                num_workers=1,
                prefetch_factor=1,
                pin_memory=False,
                persistent_workers=False,
                return_uint8=True,
            )

            self.assertEqual(loader.prefetch_factor, 1)
            self.assertFalse(loader.persistent_workers)
            image, label = loader.dataset[0]
            processed = preprocess_fn((image.unsqueeze(0), torch.tensor([label])))
            self.assertEqual(processed["images"].dtype, torch.uint8)

    def test_prefetch_change_preserves_order_rng_and_exact_pixels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_root = Path(tmpdir) / "train"
            for class_index in range(2):
                class_dir = train_root / f"class{class_index}"
                class_dir.mkdir(parents=True)
                for image_index in range(4):
                    array = np.asarray(self._image(), dtype=np.uint8).copy()
                    array = np.bitwise_xor(
                        array,
                        np.uint8(class_index * 71 + image_index * 13),
                    )
                    Image.fromarray(array, mode="RGB").save(
                        class_dir / f"sample{image_index}.png"
                    )

            def collect(*, return_uint8: bool, prefetch_factor: int):
                torch.manual_seed(20260902)
                loader, _, _ = create_imagenet_split(
                    imagenet_path=tmpdir,
                    resolution=12,
                    batch_size=2,
                    split="train",
                    use_aug=False,
                    use_latent=False,
                    use_cache=False,
                    num_workers=2,
                    prefetch_factor=prefetch_factor,
                    pin_memory=False,
                    persistent_workers=False,
                    return_uint8=return_uint8,
                )
                images, labels = [], []
                for image_batch, label_batch in loader:
                    images.append(image_batch)
                    labels.append(label_batch)
                return torch.cat(images), torch.cat(labels)

            legacy_images, legacy_labels = collect(
                return_uint8=False,
                prefetch_factor=2,
            )
            direct_images, direct_labels = collect(
                return_uint8=True,
                prefetch_factor=1,
            )
            self.assertTrue(torch.equal(direct_labels, legacy_labels))
            self.assertTrue(
                torch.equal(legacy_images, _normalized_from_uint8(direct_images))
            )

    def test_io_gate_preserves_worker_order_rng_and_exact_pixels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_root = Path(tmpdir) / "train"
            for class_index in range(2):
                class_dir = train_root / f"class{class_index}"
                class_dir.mkdir(parents=True)
                for image_index in range(8):
                    array = np.asarray(self._image(), dtype=np.uint8).copy()
                    array = np.bitwise_xor(
                        array,
                        np.uint8(class_index * 71 + image_index * 13),
                    )
                    Image.fromarray(array, mode="RGB").save(
                        class_dir / f"sample{image_index}.png"
                    )

            def collect(io_concurrency: int):
                torch.manual_seed(20260902)
                loader, _, _ = create_imagenet_split(
                    imagenet_path=tmpdir,
                    resolution=12,
                    batch_size=2,
                    split="train",
                    use_aug=False,
                    use_latent=False,
                    use_cache=False,
                    num_workers=4,
                    prefetch_factor=2,
                    pin_memory=False,
                    persistent_workers=False,
                    return_uint8=True,
                    raw_image_io_concurrency=io_concurrency,
                )
                batches = list(loader)
                return (
                    torch.cat([batch[0] for batch in batches]),
                    torch.cat([batch[1] for batch in batches]),
                )

            ungated_images, ungated_labels = collect(0)
            gated_images, gated_labels = collect(2)
            self.assertTrue(torch.equal(gated_labels, ungated_labels))
            self.assertTrue(torch.equal(gated_images, ungated_images))

    def test_train_and_eval_can_share_one_io_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for split in ("train", "val"):
                class_dir = Path(tmpdir) / split / "class0"
                class_dir.mkdir(parents=True)
                self._image().save(class_dir / "sample.png")

            semaphore = create_raw_image_io_semaphore(2)
            train_loader, _, _ = create_imagenet_split(
                imagenet_path=tmpdir,
                resolution=12,
                batch_size=1,
                split="train",
                num_workers=0,
                pin_memory=False,
                return_uint8=True,
                raw_image_io_concurrency=2,
                raw_image_io_semaphore=semaphore,
            )
            eval_loader, _, _ = create_imagenet_split(
                imagenet_path=tmpdir,
                resolution=12,
                batch_size=1,
                split="val",
                num_workers=0,
                pin_memory=False,
                raw_image_io_concurrency=2,
                raw_image_io_semaphore=semaphore,
            )
            self.assertIs(
                train_loader.dataset.loader._semaphore,
                eval_loader.dataset.loader._semaphore,
            )

    def test_io_gate_rejects_invalid_modes(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            create_raw_image_io_semaphore(-1)
        with self.assertRaisesRegex(ValueError, "only valid for direct raw"):
            create_imagenet_split(
                imagenet_path="",
                cache_path="",
                split="train",
                use_cache=True,
                raw_image_io_concurrency=2,
            )

    def test_uint8_mode_rejects_non_training_or_non_raw_inputs(self):
        invalid_options = (
            {"split": "val", "use_latent": False, "use_cache": False},
            {"split": "train", "use_latent": True, "use_cache": False},
            {"split": "train", "use_latent": False, "use_cache": True},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError, "only valid for a direct raw training split"
            ):
                create_imagenet_split(
                    imagenet_path="",
                    cache_path="",
                    return_uint8=True,
                    **options,
                )

    def test_raw_loader_extensions_preserve_pt_cache_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class_dir = Path(tmpdir) / "train" / "class0"
            class_dir.mkdir(parents=True)
            moments = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(4, 32, 32)
            torch.save(
                {"moments": moments, "moments_flip": -moments},
                class_dir / "sample.pt",
            )
            self.assertEqual(infer_latent_cache_format(tmpdir), "pt_imagefolder")
            loader, preprocess, _ = create_imagenet_split(
                imagenet_path="",
                cache_path=tmpdir,
                cache_format="auto",
                use_cache=True,
                random_flip=False,
                shuffle=False,
                drop_last=False,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 1)
            processed = preprocess(batches[0])
            self.assertTrue(torch.equal(processed["images"], moments.unsqueeze(0)))
            self.assertEqual(processed["labels"].tolist(), [0])

    def test_validation_reference_does_not_change_with_flip_rng(self):
        transform = _build_transforms(12, use_aug=False, split="val")
        outputs = []
        for seed in range(8):
            torch.manual_seed(seed)
            outputs.append(transform(self._image()))
        self.assertTrue(all(torch.equal(outputs[0], output) for output in outputs[1:]))


if __name__ == "__main__":
    unittest.main()
