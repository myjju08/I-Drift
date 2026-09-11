"""Regression checks for experimental controls and immutable job provenance."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from scripts import preflight_corrective_field as preflight
from scripts import submit_corrective_field as submit
from scripts import corrective_field_runtime as runtime


class CorrectiveFieldSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suite = self.root / "configs/corrective_field"
        shutil.copytree(preflight.ROOT / "configs/corrective_field", self.suite)

    def edit(self, variant, section, key, value):
        path = self.suite / f"{variant}.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg[section][key] = value
        path.write_text(yaml.safe_dump(cfg))

    def test_all_three_arms_are_matched_and_have_requested_objectives(self):
        configs = preflight.validate_suite(self.suite)
        self.assertEqual(configs["baseline"]["total_steps"], 100092)
        self.assertFalse(configs["baseline"]["double_drift"])
        self.assertTrue(configs["replay_double"]["double_drift"])
        self.assertEqual(configs["replay_double_gan"]["adversarial_loss_weight"], 0.1)

    def test_unmatched_optimizer_rejected(self):
        self.edit("replay_double", "optimizer", "lr", 0.004)
        with self.assertRaisesRegex(ValueError, "Unmatched replay_double.*lr"):
            preflight.validate_suite(self.suite)

    def test_wrong_named_arm_rejected(self):
        self.edit("replay_double", "train", "double_drift", False)
        with self.assertRaisesRegex(ValueError, "Wrong double_drift for replay_double"):
            preflight.validate_suite(self.suite)

    def test_shared_replay_misconfiguration_is_rejected(self):
        for variant in preflight.VARIANTS:
            self.edit(variant, "train", "historical_gen_replay_ratio", 0.5)
        with self.assertRaisesRegex(ValueError, "historical_gen_replay_ratio"):
            preflight.validate_suite(self.suite)

    def test_double_counting_gan_loss_rejected(self):
        self.edit("baseline", "feature", "adversarial_loss_weight", 0.1)
        with self.assertRaisesRegex(ValueError, "Wrong adversarial_loss_weight for baseline"):
            preflight.validate_suite(self.suite)

    def test_duplicate_flattened_key_rejected(self):
        self.edit("baseline", "optimizer", "seed", 43)
        with self.assertRaisesRegex(ValueError, "Duplicate flattened config key: seed"):
            preflight.validate_suite(self.suite)

    def test_duplicate_yaml_key_rejected(self):
        path = self.suite / "baseline.yaml"
        path.write_text(path.read_text() + "\nlogging:\n  project: wrong\n")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML key: logging"):
            preflight.read_config(path)

    def test_changed_snapshot_and_untracked_code_are_rejected(self):
        source = self.root / "source"
        source.mkdir()
        script = source / "train.py"
        script.write_text("print('committed source')\n")
        manifest = self.root / "source-manifest.json"
        manifest.write_text(json.dumps({
            "kind": "corrective_field_source_snapshot", "commit": "a" * 40,
            "files_sha256": {"train.py": preflight.sha256(script)},
        }))
        preflight.validate_snapshot(source, manifest)
        script.write_text("print('changed after submission')\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            preflight.validate_snapshot(source, manifest)
        script.write_text("print('committed source')\n")
        (source / "injected.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "file set differs"):
            preflight.validate_snapshot(source, manifest)

    def test_online_environment_is_required_without_starting_wandb(self):
        cfg = preflight.validate_suite(self.suite)["baseline"]
        with patch.dict("os.environ", {"WANDB_MODE": "offline"}, clear=True):
            with self.assertRaisesRegex(ValueError, "WANDB_MODE=online"):
                preflight.validate_runtime_environment(cfg)

    def test_production_jobs_depend_on_successful_validation(self):
        def fake_snapshot(root, destination, suite_id):
            destination.mkdir(parents=True)
            return {"commit": "a" * 40}

        commands = []

        def fake_sbatch(command, **kwargs):
            commands.append(command)
            return str(9000 + len(commands)) + "\n"

        with (patch.object(submit, "ROOT", self.root),
              patch.object(submit, "make_snapshot", side_effect=fake_snapshot),
              patch.object(submit.subprocess, "check_output", side_effect=fake_sbatch),
              patch("sys.argv", ["submit_corrective_field.py", "--submit"]),
              redirect_stdout(io.StringIO())):
            submit.main()
        self.assertEqual(len(commands), 4)
        self.assertIn("--job-name=CF-validate", commands[0])
        for command in commands[1:]:
            self.assertIn("--dependency=afterok:9001", command)
        record = json.loads(next(self.root.rglob("submission.json")).read_text())
        self.assertEqual(record["status"], "submitted")
        self.assertEqual(record["validation"]["job_id"], "9001")
        self.assertEqual(set(record["jobs"]), set(preflight.VARIANTS))

    def test_requeue_reuses_only_original_workdir_and_configuration(self):
        snapshot = self.root / "snapshot"
        shutil.copytree(self.suite, snapshot / "source/configs/corrective_field")
        (snapshot / "source-manifest.json").write_text(json.dumps({"commit": "a" * 40}))
        runs = self.root / "runs"
        workdir = runtime.prepare_workdir(snapshot, "baseline", runs, "123", 0)
        self.assertEqual(workdir, runtime.prepare_workdir(snapshot, "baseline", runs, "123", 1))
        with self.assertRaisesRegex(ValueError, "accidental reuse"):
            runtime.prepare_workdir(snapshot, "baseline", runs, "123", 0)
        with self.assertRaisesRegex(ValueError, "no original workdir binding"):
            runtime.prepare_workdir(snapshot, "replay_double", runs, "123", 1)
        config = snapshot / "source/configs/corrective_field/baseline.yaml"
        config.write_text(config.read_text() + "\n# changed after submission\n")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            runtime.prepare_workdir(snapshot, "baseline", runs, "123", 1)

    def test_requeue_requires_checkpoint_gan_and_replay_state(self):
        workdir = self.root / "resume"
        (workdir / "checkpoints").mkdir(parents=True)
        (workdir / "checkpoints/ckpt_latest.pt").write_bytes(b"placeholder")
        (workdir / "wandb_run_id.txt").write_text("same-wandb-run")
        config = self.suite / "replay_double_gan.yaml"
        cfg = preflight.read_config(config)
        state = {"step": 100, "model": {}, "ema": {}, "optimizer": {}, "config": cfg,
                 "adversarial_system": {}}
        with patch.dict("sys.modules", {"torch": Mock(load=Mock(return_value=state))}):
            with self.assertRaisesRegex(ValueError, "requires matching replay state"):
                runtime.validate_checkpoint(workdir, config)
            for rank in range(2):
                (workdir / f"historical_gen_replay_capture_step0000100_rank{rank:02d}.npz").write_bytes(b"snapshot")
            self.assertEqual(runtime.validate_checkpoint(workdir, config), 100)
            del state["adversarial_system"]
            with self.assertRaisesRegex(ValueError, "wrong GAN state"):
                runtime.validate_checkpoint(workdir, config)


if __name__ == "__main__":
    unittest.main()
