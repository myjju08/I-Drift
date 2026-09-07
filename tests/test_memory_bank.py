import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from memory_bank import ArrayMemoryBank, CompressedPixelMemoryBank


class ArrayMemoryBankPixelStorageTest(unittest.TestCase):
    def test_pixel_sample_owns_values_after_bank_wraparound_and_mutation(self):
        bank = ArrayMemoryBank(num_classes=1, max_size=2, storage_mode="pixel_uint8")
        first = np.stack([
            np.full((3, 4, 5), 17, dtype=np.uint8),
            np.full((3, 4, 5), 221, dtype=np.uint8),
        ])
        labels = np.zeros(2, dtype=np.int64)
        bank.add(first, labels)
        sampled = bank.sample(np.array([0]), n_samples=2, rng=np.random.default_rng(7))
        retained = sampled.clone()
        bank.add(np.full_like(first, 91), labels)
        self.assertTrue(torch.equal(sampled, retained))
        sampled.fill_(0)
        np.testing.assert_array_equal(bank.bank[0], np.full_like(first, 91))
        later = bank.sample(np.array([0]), n_samples=2, rng=np.random.default_rng(8))
        expected = torch.from_numpy(np.full((1, 2, 3, 4, 5), 91, dtype=np.uint8)).float().div_(255).sub_(0.5).div_(0.5)
        self.assertTrue(torch.equal(later, expected))

    def test_direct_decode_keeps_caller_array_independent(self):
        bank = ArrayMemoryBank(num_classes=1, max_size=2)
        source = np.arange(8, dtype=np.float32).reshape(2, 4)
        expected = source.copy()
        decoded = bank._decode_samples(source)
        source.fill(-1)
        np.testing.assert_array_equal(decoded.numpy(), expected)
        decoded.fill_(3)
        np.testing.assert_array_equal(source, np.full_like(source, -1))

    def test_raw_mode_preserves_legacy_dtype_and_values(self):
        bank = ArrayMemoryBank(num_classes=1, max_size=2, dtype=np.float16)
        values = torch.tensor([[[[0.125]]], [[[0.25]]]], dtype=torch.float32)
        bank.add(values, torch.zeros(2, dtype=torch.long))

        self.assertEqual(bank.storage_mode, "raw")
        self.assertEqual(bank.bank.dtype, np.dtype(np.float16))
        sampled = bank.sample(
            np.zeros(1, dtype=np.int64),
            n_samples=2,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(sampled.dtype, torch.float16)
        self.assertTrue(set(sampled.flatten().tolist()) <= {0.125, 0.25})

    def test_pixel_uint8_round_trips_normalized_source_bytes(self):
        source = torch.tensor(
            [
                [
                    [[0, 1], [127, 128]],
                    [[254, 255], [64, 192]],
                    [[32, 96], [160, 224]],
                ]
            ],
            dtype=torch.uint8,
        )
        normalized = source.float().div(255.0).sub(0.5).div(0.5)
        bank = ArrayMemoryBank(
            num_classes=1,
            max_size=1,
            storage_mode="pixel_uint8",
        )
        bank.add(normalized, torch.zeros(1, dtype=torch.long))

        self.assertEqual(bank.bank.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(bank.bank[0, 0], source[0].numpy())
        sampled = bank.sample(np.zeros(1, dtype=np.int64), n_samples=1)
        self.assertEqual(sampled.dtype, torch.float32)
        self.assertTrue(torch.equal(sampled[0, 0], normalized[0]))

    def test_direct_uint8_ingress_matches_legacy_normalized_ingress_exactly(self):
        source = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(
            2, 3, 4, 5
        )
        normalized = source.float().div(255.0).sub(0.5).div(0.5)
        labels = torch.tensor([0, 1], dtype=torch.long)

        legacy_bank = ArrayMemoryBank(
            num_classes=2,
            max_size=1,
            storage_mode="pixel_uint8",
        )
        direct_bank = ArrayMemoryBank(
            num_classes=2,
            max_size=1,
            storage_mode="pixel_uint8",
        )
        legacy_bank.add(normalized, labels)
        direct_bank.add(source, labels)

        np.testing.assert_array_equal(direct_bank.bank, legacy_bank.bank)
        np.testing.assert_array_equal(direct_bank.bank[:, 0], source.numpy())
        sampled_legacy = legacy_bank.sample(
            labels.numpy(),
            n_samples=1,
            rng=np.random.default_rng(19),
        )
        sampled_direct = direct_bank.sample(
            labels.numpy(),
            n_samples=1,
            rng=np.random.default_rng(19),
        )
        self.assertTrue(torch.equal(sampled_direct, sampled_legacy))
        self.assertTrue(
            torch.equal(sampled_direct[:, 0], normalized)
        )

    def test_pixel_uint8_rejects_wrong_shape_and_range(self):
        wrong_shape = ArrayMemoryBank(
            num_classes=1,
            max_size=1,
            storage_mode="pixel_uint8",
        )
        with self.assertRaisesRegex(ValueError, "CHW RGB"):
            wrong_shape.add(torch.zeros(1, 4, 2, 2), torch.zeros(1))

        wrong_range = ArrayMemoryBank(
            num_classes=1,
            max_size=1,
            storage_mode="pixel_uint8",
        )
        with self.assertRaisesRegex(ValueError, "normalized to"):
            wrong_range.add(torch.full((1, 3, 2, 2), 255.0), torch.zeros(1))

    def test_pixel_snapshot_round_trip_and_mode_guard(self):
        source = torch.arange(12, dtype=torch.uint8).reshape(1, 3, 2, 2)
        normalized = source.float().div(255.0).sub(0.5).div(0.5)
        bank = ArrayMemoryBank(
            num_classes=1,
            max_size=1,
            storage_mode="pixel_uint8",
        )
        bank.add(normalized, torch.zeros(1, dtype=torch.long))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pixel_bank.npz"
            metadata = {"rank": 0, "generated_epoch": 14, "step": 70100}
            bank.save_npz(path, metadata=metadata)
            restored = ArrayMemoryBank(
                num_classes=1,
                max_size=1,
                storage_mode="pixel_uint8",
            )
            self.assertEqual(
                restored.load_npz(path, expected_metadata=metadata), metadata
            )
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                restored.load_npz(path, expected_metadata={"rank": 1})
            with self.assertRaisesRegex(ValueError, "storage_mode"):
                ArrayMemoryBank(num_classes=1, max_size=1).load_npz(path)
            self.assertEqual(list(Path(tmpdir).iterdir()), [path])

        self.assertEqual(restored.bank.dtype, np.dtype(np.uint8))
        self.assertTrue(
            torch.equal(
                restored.sample(np.zeros(1, dtype=np.int64), n_samples=1),
                normalized.reshape(1, 1, 3, 2, 2),
            )
        )

    def test_legacy_raw_snapshot_without_storage_mode_preserves_float16(self):
        values = np.arange(12, dtype=np.float16).reshape(2, 3, 2)
        pointers = np.array([1, 2], dtype=np.int32)
        counts = np.array([3, 2], dtype=np.int32)
        restored = ArrayMemoryBank(num_classes=2, max_size=3, dtype=np.float16)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_bank.npz"
            np.savez(path, bank=values, ptr=pointers, count=counts)
            self.assertEqual(restored.load_npz(path), {})
        self.assertEqual(restored.bank.dtype, np.dtype(np.float16))
        np.testing.assert_array_equal(restored.bank, values)
        np.testing.assert_array_equal(restored.ptr, pointers)
        np.testing.assert_array_equal(restored.count, counts)

    def test_failed_snapshot_publish_preserves_previous_archive(self):
        bank = ArrayMemoryBank(num_classes=1, max_size=1, storage_mode="pixel_uint8")
        bank.add(np.zeros((1, 3, 2, 2), dtype=np.uint8), np.array([0]))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pixel_bank.npz"
            bank.save_npz(path, metadata={"step": 1})
            previous_archive = path.read_bytes()
            bank.add(np.full((1, 3, 2, 2), 255, dtype=np.uint8), np.array([0]))
            with mock.patch("memory_bank.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    bank.save_npz(path, metadata={"step": 2})
            self.assertEqual(path.read_bytes(), previous_archive)
            self.assertEqual(list(Path(tmpdir).iterdir()), [path])

    def test_pixel_storage_is_four_times_smaller_than_float32(self):
        sample = torch.zeros(1, 3, 8, 8)
        pixel_bank = ArrayMemoryBank(
            num_classes=2,
            max_size=3,
            storage_mode="pixel_uint8",
        )
        raw_bank = ArrayMemoryBank(num_classes=2, max_size=3)
        labels = torch.zeros(1, dtype=torch.long)
        pixel_bank.add(sample, labels)
        raw_bank.add(sample, labels)

        self.assertEqual(pixel_bank.bank.nbytes * 4, raw_bank.bank.nbytes)


class CompressedPixelMemoryBankTest(unittest.TestCase):
    def test_dense_trace_and_generator_state_are_bitwise_identical(self):
        dense = ArrayMemoryBank(
            num_classes=3, max_size=4, storage_mode="pixel_uint8"
        )
        compressed = CompressedPixelMemoryBank(
            num_classes=3, max_size=4, codec_workers=4
        )
        source_rng = np.random.default_rng(20260902)

        # Include wraparound and repeated labels in the exact same add trace.
        for labels in (
            np.array([0, 1, 2, 0]),
            np.array([2, 2, 1, 0, 1]),
            np.array([0, 0, 0, 0, 0]),
        ):
            pixels = source_rng.integers(
                0, 256, size=(len(labels), 3, 5, 7), dtype=np.uint8
            )
            dense.add(pixels, labels)
            compressed.add(pixels, labels)
            np.testing.assert_array_equal(compressed.ptr, dense.ptr)
            np.testing.assert_array_equal(compressed.count, dense.count)

        for n_samples in (2, 6):
            dense_rng = np.random.default_rng(91 + n_samples)
            compressed_rng = np.random.default_rng(91 + n_samples)
            labels = np.array([2, 0, 2, 1], dtype=np.int64)
            expected = dense.sample(labels, n_samples=n_samples, rng=dense_rng)
            actual = compressed.sample(
                labels, n_samples=n_samples, rng=compressed_rng
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(
                compressed_rng.integers(0, 2**31),
                dense_rng.integers(0, 2**31),
            )

    def test_global_numpy_rng_and_empty_class_match_dense(self):
        dense = ArrayMemoryBank(
            num_classes=2, max_size=2, storage_mode="pixel_uint8"
        )
        compressed = CompressedPixelMemoryBank(num_classes=2, max_size=2)
        pixels = np.arange(2 * 3 * 3 * 4, dtype=np.uint8).reshape(2, 3, 3, 4)
        labels = np.zeros(2, dtype=np.int64)
        dense.add(pixels, labels)
        compressed.add(pixels, labels)

        np.random.seed(701)
        expected = dense.sample(np.array([0, 1]), n_samples=3)
        expected_next = np.random.randint(0, 2**31)
        np.random.seed(701)
        actual = compressed.sample(np.array([0, 1]), n_samples=3)
        actual_next = np.random.randint(0, 2**31)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual_next, expected_next)

    def test_exact_normalized_ingress_compression_stats_and_wraparound(self):
        bank = CompressedPixelMemoryBank(num_classes=1, max_size=2)
        source = torch.zeros(3, 3, 16, 16, dtype=torch.uint8)
        source[1].fill_(127)
        source[2].fill_(255)
        normalized = source.float().div(255.0).sub(0.5).div(0.5)
        bank.add(normalized, torch.zeros(3, dtype=torch.long))

        self.assertEqual(len(bank), 2)
        self.assertEqual(bank.raw_equivalent_bytes, 2 * 3 * 16 * 16)
        self.assertGreater(bank.compression_ratio, 1.0)
        sampled = bank.sample(
            np.zeros(1, dtype=np.int64),
            n_samples=2,
            rng=np.random.default_rng(5),
        )
        stored = {tuple(row.flatten().tolist()) for row in normalized[1:]}
        self.assertTrue(
            all(tuple(row.flatten().tolist()) in stored for row in sampled[0])
        )

    def test_corrupt_payload_fails_closed(self):
        bank = CompressedPixelMemoryBank(num_classes=1, max_size=1)
        bank.add(torch.zeros(1, 3, 4, 4, dtype=torch.uint8), np.array([0]))
        payload = bytearray(bank.bank[0, 0])
        payload[len(payload) // 2] ^= 0x01
        bank.bank[0, 0] = bytes(payload)
        with self.assertRaisesRegex(RuntimeError, "decompress"):
            bank.sample(np.array([0]), n_samples=1)

    def test_raw_fallback_counter_overwrite_and_shape_guard(self):
        bank = CompressedPixelMemoryBank(
            num_classes=1, max_size=1, codec_workers=4
        )
        rng = np.random.default_rng(819)
        first = rng.integers(0, 256, size=(1, 3, 32, 32), dtype=np.uint8)
        second = rng.integers(0, 256, size=(1, 3, 32, 32), dtype=np.uint8)
        bank.add(first, np.array([0]))
        self.assertEqual(bank.bank[0, 0][0], bank._RAW_TAG)
        self.assertEqual(bank.payload_bytes, first[0].nbytes + 5)
        self.assertEqual(bank.raw_equivalent_bytes, first[0].nbytes)

        bank.add(second, np.array([0]))
        self.assertEqual(bank.bank[0, 0][0], bank._RAW_TAG)
        self.assertEqual(bank.payload_bytes, second[0].nbytes + 5)
        self.assertEqual(bank.raw_equivalent_bytes, second[0].nbytes)
        sampled = bank.sample(
            np.array([0]), n_samples=1, rng=np.random.default_rng(1)
        )
        expected = torch.from_numpy(second).float().div(255.0).sub(0.5).div(0.5)
        self.assertTrue(torch.equal(sampled[:, 0], expected))

        corrupt = bytearray(bank.bank[0, 0])
        corrupt[-1] ^= 0x01
        bank.bank[0, 0] = bytes(corrupt)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            bank.sample(np.array([0]), n_samples=1)
        bank.add(second, np.array([0]))

        same_bytes_different_shape = second.reshape(1, 3, 16, 64)
        with self.assertRaisesRegex(ValueError, "sample shape changed"):
            bank.add(same_bytes_different_shape, np.array([0]))


if __name__ == "__main__":
    unittest.main()
