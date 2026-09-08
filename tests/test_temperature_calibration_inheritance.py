import copy
import json
from pathlib import Path
import tempfile
import unittest

from temperature_calibration_inheritance import (
    MANIFEST_KEY, MANIFEST_SHA_KEY, sha256_file,
    validate_dino_temperature_inheritance,
)


class TemperatureInheritanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.teacher = self.root / "teacher.pth"
        self.teacher.write_bytes(b"original immutable teacher")
        self.tuned = self.root / "tuned.pth"
        self.tuned.write_bytes(b"separate tuned DINO backbone")
        self.artifact = {
            "encoder_provenance": {"dino": {
                "feature_extractor": "dino_resnet50",
                "checkpoint_sha256": sha256_file(self.teacher),
                "checkpoint_bytes": self.teacher.stat().st_size,
            }},
            "symmetric_profiles": {"dino": {"default": 1.0, "stage3": 1.01, "stage4": 1.02}},
        }
        artifact_path = self.write_json("calibration.json", self.artifact)
        self.baseline = {
            "name": "S4 / DINO", "feature_checkpoint": str(self.teacher),
            "feature_extractor": "dino_resnet50", "temperature_calibration_status": "ready",
            "temperature_calibration_role": "dino", "require_raw_temperature_calibration": True,
            "temperature_calibration_artifact": str(artifact_path),
            "temperature_calibration_artifact_sha256": sha256_file(artifact_path),
            "layer_temperature_profiles": self.artifact["symmetric_profiles"],
            "batch_size": 4, "seed": 43, "throughput_opt_level": 3,
            "use_latent": False, "use_cache": False, "num_workers": 8,
            "_raw": {"logging": {"name": "S4 / DINO"},
                     "model": {"depth": 12},
                     "dataset": {"num_workers": 8},
                     "feature": {"feature_checkpoint": str(self.teacher)}},
        }
        baseline_path = self.write_json("baseline.json", self.baseline)
        self.summary = {
            "best_step": 1500, "export_sha256": sha256_file(self.tuned),
            "original_teacher_unchanged": True, "in_memory_teacher_unchanged": True,
            "metadata": {"teacher_sha256": sha256_file(self.teacher),
                         "trainable_blocks": ["layer3.5", "layer4.2"]},
            "last_student_freeze_audit": {
                "trainable_parameter_names": [f"layer{s}.{b}.conv{c}.weight"
                                              for s, b in ((3, 5), (4, 2)) for c in (1, 2, 3)],
                "all_frozen_tensors_unchanged": True,
                "bn_parameters_and_buffers_unchanged": True, "changed_frozen_keys": [],
            },
        }
        summary_path = self.write_json("tuning_summary.json", self.summary)
        self.manifest = {
            "schema_version": 1, "kind": "weights_only_dino_ablation",
            "numeric_temperature_policy": "inherit_original_without_refit",
            "selected_tuning_step": 1500,
            "calibration_artifact_sha256": sha256_file(artifact_path),
            "baseline_config": self.identity(baseline_path),
            "original_teacher": self.identity(self.teacher, size=True),
            "tuned_encoder": self.identity(self.tuned, size=True),
            "tuning_summary": self.identity(summary_path),
        }
        self.cfg = copy.deepcopy(self.baseline)
        self.cfg.update(name="S4 / DINO(tunning)", feature_checkpoint=str(self.tuned))
        self.cfg["_raw"]["logging"]["name"] = self.cfg["name"]
        self.cfg["_raw"]["feature"]["feature_checkpoint"] = str(self.tuned)
        self.repin_manifest()

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True))
        return path

    def identity(self, path, *, size=False):
        value = {"path": str(path), "sha256": sha256_file(path)}
        if size:
            value["bytes"] = path.stat().st_size
        return value

    def repin_manifest(self):
        path = self.write_json("inheritance.json", self.manifest)
        self.cfg[MANIFEST_KEY] = str(path)
        self.cfg[MANIFEST_SHA_KEY] = sha256_file(path)
        self.cfg["_raw"]["feature"][MANIFEST_KEY] = str(path)
        self.cfg["_raw"]["feature"][MANIFEST_SHA_KEY] = sha256_file(path)

    def validate(self, cfg=None):
        return validate_dino_temperature_inheritance(cfg or self.cfg, self.artifact, self.tuned)

    def test_authenticates_changed_weights_without_mutating_original_taus_or_config(self):
        before = copy.deepcopy(self.cfg)
        result = self.validate()
        self.assertEqual(result["selected_tuning_step"], 1500)
        self.assertEqual(result["original_teacher_sha256"], sha256_file(self.teacher))
        self.assertEqual(result["tuned_encoder_sha256"], sha256_file(self.tuned))
        self.assertEqual(self.cfg, before)
        self.assertEqual(self.teacher.read_bytes(), b"original immutable teacher")

    def test_rejects_training_changes_even_when_hidden_in_raw_model(self):
        for key, value in (("batch_size", 8), ("seed", 44), ("throughput_opt_level", 4)):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg[key] = value
                with self.assertRaisesRegex(RuntimeError, "training config differs"):
                    self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["_raw"]["model"]["depth"] = 16
        with self.assertRaisesRegex(RuntimeError, "training config differs"):
            self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["_raw"]["model"]["name"] = "a different architecture"
        with self.assertRaisesRegex(RuntimeError, "training config differs"):
            self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["layer_temperature_profiles"]["dino"]["stage3"] += 1e-12
        with self.assertRaisesRegex(RuntimeError, "training config differs"):
            self.validate(cfg)

    def test_rejects_any_extra_loss_or_calibration_capture(self):
        for key, value in (("adversarial_mode", "raw_gan"), ("feature_adapter", True),
                           ("feature_gan", True), ("temperature_calibration_status", "calibration_capture")):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg[key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(cfg)

    def test_rejects_missing_or_changed_manifest_identity(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pop(MANIFEST_SHA_KEY)
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            self.validate(cfg)
        Path(self.cfg[MANIFEST_KEY]).write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            self.validate()

    def test_rejects_modified_teacher_tuned_export_or_baseline(self):
        for identity, message in (("original_teacher", "original teacher"),
                                  ("tuned_encoder", "tuned encoder"),
                                  ("baseline_config", "baseline config")):
            with self.subTest(identity=identity):
                path = Path(self.manifest[identity]["path"])
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(RuntimeError, message + " SHA256"):
                    self.validate()
                path.write_bytes(original)

    def test_rejects_mismatched_tuning_lineage_even_with_repinned_summary(self):
        for key, value in (("export_sha256", "f" * 64), ("original_teacher_unchanged", False),
                           ("in_memory_teacher_unchanged", False), ("best_step", 1499)):
            with self.subTest(key=key):
                summary = copy.deepcopy(self.summary)
                summary[key] = value
                path = self.write_json("tuning_summary.json", summary)
                self.manifest["tuning_summary"] = self.identity(path)
                self.repin_manifest()
                with self.assertRaises(RuntimeError):
                    self.validate()

    def test_rejects_changed_bn_and_original_calibration_relabeling(self):
        summary = copy.deepcopy(self.summary)
        summary["last_student_freeze_audit"]["bn_parameters_and_buffers_unchanged"] = False
        self.manifest["tuning_summary"] = self.identity(self.write_json("tuning_summary.json", summary))
        self.repin_manifest()
        with self.assertRaisesRegex(RuntimeError, "BN integrity"):
            self.validate()
        self.manifest["numeric_temperature_policy"] = "recalibrated"
        self.repin_manifest()
        with self.assertRaisesRegex(RuntimeError, "sidecar protocol"):
            self.validate()

    def test_rejects_unpinned_active_encoder_and_teacher_alias(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["feature_checkpoint"] = str(self.teacher)
        with self.assertRaisesRegex(RuntimeError, "active encoder path"):
            self.validate(cfg)
        self.tuned.unlink()
        self.tuned.symlink_to(self.teacher)
        self.manifest["tuned_encoder"] = self.identity(self.tuned, size=True)
        self.repin_manifest()
        with self.assertRaisesRegex(RuntimeError, "separate changed checkpoint"):
            self.validate()

    def enable_v2_io_cap(self):
        self.manifest["schema_version"] = 2
        self.manifest["runtime_overrides"] = {"raw_image_io_concurrency_per_rank": 1}
        self.cfg["raw_image_io_concurrency_per_rank"] = 1
        self.cfg["_raw"]["dataset"]["raw_image_io_concurrency_per_rank"] = 1
        self.repin_manifest()

    def test_v2_allows_only_declared_physical_io_cap_without_mutation(self):
        self.enable_v2_io_cap()
        before = copy.deepcopy(self.cfg)
        result = self.validate()
        self.assertEqual(result["runtime_overrides"], {"raw_image_io_concurrency_per_rank": 1})
        self.assertEqual(self.cfg, before)
        self.baseline["raw_image_io_concurrency_per_rank"] = 0
        self.baseline["_raw"]["dataset"]["raw_image_io_concurrency_per_rank"] = 0
        path = self.write_json("baseline.json", self.baseline)
        self.manifest["baseline_config"] = self.identity(path)
        self.repin_manifest()
        self.validate()

    def test_v1_stays_strict_and_rejects_runtime_declaration(self):
        self.cfg["raw_image_io_concurrency_per_rank"] = 1
        self.cfg["_raw"]["dataset"]["raw_image_io_concurrency_per_rank"] = 1
        with self.assertRaisesRegex(RuntimeError, "training config differs"):
            self.validate()
        self.manifest["runtime_overrides"] = {"raw_image_io_concurrency_per_rank": 1}
        self.repin_manifest()
        with self.assertRaisesRegex(RuntimeError, "version 1"):
            self.validate()

    def test_v2_rejects_incorrect_missing_or_extra_runtime_overrides(self):
        self.enable_v2_io_cap()
        for overrides in (None, {}, {"raw_image_io_concurrency_per_rank": 2},
                          {"raw_image_io_concurrency_per_rank": True},
                          {"raw_image_io_concurrency_per_rank": 1, "num_workers": 1}):
            with self.subTest(overrides=overrides):
                self.manifest["runtime_overrides"] = overrides
                self.repin_manifest()
                with self.assertRaisesRegex(RuntimeError, "only the declared physical I/O"):
                    self.validate()

    def test_v2_checks_flat_nested_values_and_keeps_other_runtime_settings_strict(self):
        self.enable_v2_io_cap()
        for nested in (False, True):
            for bad in (None, 0, 2, True, "1"):
                with self.subTest(nested=nested, bad=bad):
                    cfg = copy.deepcopy(self.cfg)
                    target = cfg["_raw"]["dataset"] if nested else cfg
                    target["raw_image_io_concurrency_per_rank"] = bad
                    with self.assertRaisesRegex(RuntimeError, "physical I/O cap must match"):
                        self.validate(cfg)
        for key, value in (("num_workers", 1), ("throughput_opt_level", 4),
                           ("positive_memory_bank_backend", "zstd_delta"), ("seed", 44)):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg[key] = value
                with self.assertRaisesRegex(RuntimeError, "training config differs"):
                    self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["_raw"]["model"]["depth"] = 16
        with self.assertRaisesRegex(RuntimeError, "training config differs"):
            self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["layer_temperature_profiles"]["dino"]["stage3"] += 1e-12
        with self.assertRaisesRegex(RuntimeError, "training config differs"):
            self.validate(cfg)

    def test_v2_rejects_nonraw_io_and_preexisting_baseline_throttle(self):
        self.enable_v2_io_cap()
        cfg = copy.deepcopy(self.cfg)
        cfg["use_cache"] = True
        with self.assertRaisesRegex(RuntimeError, "direct raw image"):
            self.validate(cfg)
        self.baseline["raw_image_io_concurrency_per_rank"] = 1
        self.manifest["baseline_config"] = self.identity(self.write_json("baseline.json", self.baseline))
        self.repin_manifest()
        with self.assertRaisesRegex(RuntimeError, "baseline flattened"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
