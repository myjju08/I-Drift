"""Regression checks for experimental controls and immutable job provenance."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from scripts import preflight_corrective_field as preflight
from scripts import submit_corrective_field as submit
from scripts import corrective_field_runtime as runtime
from scripts import validate_corrective_field as gate


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
        self.assertEqual(configs["baseline"]["double_drift_mode"], "off")
        self.assertEqual(configs["replay_double"]["double_drift_mode"], "feature")
        self.assertEqual(configs["replay_double"]["double_drift_c0"], 0.75)
        self.assertEqual(configs["replay_double"]["double_drift_c1"], 0.25)
        self.assertEqual(configs["replay_only"]["double_drift_c1"], 0.0)
        self.assertTrue(configs["replay_only"]["historical_gen_replay"])
        self.assertTrue(all(not cfg["feature_gan"] and cfg["adversarial_mode"] == "none" for cfg in configs.values()))

    def test_unmatched_optimizer_rejected(self):
        self.edit("replay_double", "optimizer", "lr", 0.004)
        with self.assertRaisesRegex(ValueError, "Unmatched replay_double.*lr"):
            preflight.validate_suite(self.suite)

    def test_old_double_coefficients_and_changed_control_coefficients_are_rejected(self):
        self.edit("replay_double", "train", "double_drift_c0", 1.0)
        self.edit("replay_double", "train", "double_drift_c1", 1.0)
        with self.assertRaisesRegex(ValueError, "Wrong double_drift_c0 for replay_double"):
            preflight.validate_suite(self.suite)
        self.edit("replay_double", "train", "double_drift_c0", 0.75)
        with self.assertRaisesRegex(ValueError, "Wrong double_drift_c1 for replay_double"):
            preflight.validate_suite(self.suite)
        self.edit("replay_double", "train", "double_drift_c1", 0.25)
        for variant in ("baseline", "replay_only"):
            with self.subTest(variant=variant):
                self.edit(variant, "train", "double_drift_c0", 0.75)
                with self.assertRaisesRegex(ValueError, f"Wrong double_drift_c0 for {variant}"):
                    preflight.validate_suite(self.suite)
                self.edit(variant, "train", "double_drift_c0", 1.0)

    def test_wrong_named_arm_rejected(self):
        self.edit("replay_double", "train", "double_drift_mode", "off")
        with self.assertRaisesRegex(ValueError, "Wrong double_drift_mode for replay_double"):
            preflight.validate_suite(self.suite)

    def test_shared_replay_misconfiguration_is_rejected(self):
        for variant in preflight.VARIANTS:
            self.edit(variant, "train", "historical_gen_replay_ratio", 0.5)
        with self.assertRaisesRegex(ValueError, "historical_gen_replay_ratio"):
            preflight.validate_suite(self.suite)

    def test_any_gan_objective_is_rejected(self):
        for variant in preflight.VARIANTS:
            self.edit(variant, "feature", "feature_gan", True)
        with self.assertRaisesRegex(ValueError, "feature_gan"):
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
        def fake_snapshot(root, destination, suite_id, execution=None):
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
        self.assertEqual(record["launch_policy"], "slurm_only")
        self.assertEqual(record["selected_variants"], list(preflight.VARIANTS))
        self.assertEqual(record["validation"]["variants"], list(preflight.VARIANTS))

    def test_selected_replay_arms_never_submit_baseline_and_keep_full_validation(self):
        def fake_snapshot(root, destination, suite_id, execution=None):
            destination.mkdir(parents=True)
            return {"commit": "a" * 40}

        commands = []

        def fake_sbatch(command, **kwargs):
            commands.append(command)
            return str(9100 + len(commands)) + "\n"

        selected = ["replay_only", "replay_double"]
        with (patch.object(submit, "ROOT", self.root),
              patch.object(submit, "make_snapshot", side_effect=fake_snapshot),
              patch.object(submit.subprocess, "check_output", side_effect=fake_sbatch),
              patch("sys.argv", ["submit_corrective_field.py", "--submit", "--variants", *selected]),
              redirect_stdout(io.StringIO())):
            submit.main()
        self.assertEqual(len(commands), 3)
        self.assertIn("--job-name=CF-validate", commands[0])
        self.assertTrue(any(command.endswith("validate_corrective_field.sbatch") for command in commands[0]))
        self.assertNotIn("--variants", commands[0])
        for name, command in zip(selected, commands[1:]):
            self.assertIn(f"--job-name=CF-{name}", command)
            self.assertIn("--dependency=afterok:9101", command)
        self.assertFalse(any("--job-name=CF-baseline" in command for command in commands))
        record = json.loads(next(self.root.rglob("submission.json")).read_text())
        self.assertEqual(record["launch_policy"], "slurm_only")
        self.assertEqual(record["selected_variants"], selected)
        self.assertEqual(list(record["jobs"]), selected)
        self.assertEqual(record["validation"]["variants"], list(preflight.VARIANTS))

    def test_selected_plan_does_not_submit_and_duplicate_variants_are_rejected(self):
        output = io.StringIO()
        with (patch.object(submit, "ROOT", self.root),
              patch.object(submit, "make_snapshot") as snapshot,
              patch.object(submit.subprocess, "check_output") as run,
              patch("sys.argv", ["submit_corrective_field.py", "--variants", "replay_only"]),
              redirect_stdout(output)):
            submit.main()
        snapshot.assert_not_called()
        run.assert_not_called()
        self.assertIn("--job-name=CF-replay_only", output.getvalue())
        self.assertNotIn("--job-name=CF-baseline", output.getvalue())
        self.assertNotIn("--job-name=CF-replay_double", output.getvalue())
        with (patch("sys.argv", ["submit_corrective_field.py", "--variants", "baseline", "baseline"]),
              patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error):
            submit.main()
        self.assertEqual(error.exception.code, 2)

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

    def test_requeue_requires_replay_state_and_rejects_gan(self):
        workdir = self.root / "resume"
        (workdir / "checkpoints").mkdir(parents=True)
        (workdir / "checkpoints/ckpt_latest.pt").write_bytes(b"placeholder")
        (workdir / "wandb_run_id.txt").write_text("same-wandb-run")
        config = self.suite / "replay_only.yaml"
        cfg = preflight.read_config(config)
        state = {"step": 100, "model": {}, "ema": {}, "optimizer": {}, "config": cfg}
        with patch.dict("sys.modules", {"torch": Mock(load=Mock(return_value=state))}):
            with self.assertRaisesRegex(ValueError, "requires matching replay state"):
                runtime.validate_checkpoint(workdir, config)
            for rank in range(2):
                (workdir / f"historical_gen_replay_capture_step0000100_rank{rank:02d}.npz").write_bytes(b"snapshot")
            self.assertEqual(runtime.validate_checkpoint(workdir, config), 100)
            state["feature_discriminator"] = {}
            with self.assertRaisesRegex(ValueError, "discriminator state"):
                runtime.validate_checkpoint(workdir, config)

    def test_requeue_accepts_checkpoint_paired_postboundary_replay_state(self):
        workdir = self.root / "paired_resume"
        (workdir / "checkpoints").mkdir(parents=True)
        (workdir / "checkpoints/ckpt_latest.pt").write_bytes(b"placeholder")
        (workdir / "wandb_run_id.txt").write_text("same-wandb-run")
        config = self.suite / "replay_only.yaml"
        cfg = preflight.read_config(config)
        state = {"step": 30000, "model": {}, "ema": {}, "optimizer": {}, "config": cfg}
        with patch.dict("sys.modules", {"torch": Mock(load=Mock(return_value=state))}):
            # A state from another checkpoint cannot satisfy this resume.
            for rank in range(2):
                (workdir / f"historical_gen_replay_state_step0029999_rank{rank:02d}.npz").write_bytes(b"wrong-step")
            with self.assertRaisesRegex(ValueError, "requires matching replay state"):
                runtime.validate_checkpoint(workdir, config)
            for rank in range(2):
                (workdir / f"historical_gen_replay_state_step0030000_rank{rank:02d}.npz").write_bytes(b"paired-state")
            self.assertEqual(runtime.validate_checkpoint(workdir, config), 30000)
            # Prefer the paired state even if an older immutable bank exists.
            for rank in range(2):
                (workdir / f"historical_gen_replay_rank{rank:02d}.npz").write_bytes(b"legacy")
            (workdir / "historical_gen_replay_state_step0030000_rank00.npz").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "state_step0030000_rank00"):
                runtime.validate_checkpoint(workdir, config)
            # Frozen replay retains the trainer's old-checkpoint compatibility.
            for path in workdir.glob("historical_gen_replay_state_step0030000_rank*.npz"):
                path.unlink()
            self.assertEqual(runtime.validate_checkpoint(workdir, config), 30000)

    def test_rolling_replay_requires_paired_state_but_fresh_current_needs_no_bank(self):
        workdir = self.root / "rolling_resume"
        (workdir / "checkpoints").mkdir(parents=True)
        (workdir / "checkpoints/ckpt_latest.pt").write_bytes(b"placeholder")
        (workdir / "wandb_run_id.txt").write_text("same-wandb-run")
        config = self.suite / "replay_only.yaml"
        self.edit("replay_only", "train", "historical_gen_replay_policy", "fifo")
        cfg = preflight.read_config(config)
        state = {"step": 30000, "model": {}, "ema": {}, "optimizer": {}, "config": cfg}
        for rank in range(2):
            (workdir / f"historical_gen_replay_rank{rank:02d}.npz").write_bytes(b"legacy")
        with patch.dict("sys.modules", {"torch": Mock(load=Mock(return_value=state))}):
            with self.assertRaisesRegex(ValueError, "state_step0030000_rank00"):
                runtime.validate_checkpoint(workdir, config)
            for rank in range(2):
                (workdir / f"historical_gen_replay_state_step0030000_rank{rank:02d}.npz").write_bytes(b"paired-state")
            self.assertEqual(runtime.validate_checkpoint(workdir, config), 30000)
            self.edit("replay_only", "train", "historical_gen_replay_source", "fresh_current")
            state["config"] = preflight.read_config(config)
            for path in workdir.glob("historical_gen_replay_*.npz"):
                path.unlink()
            self.assertEqual(runtime.validate_checkpoint(workdir, config), 30000)

    def test_gpu_gate_checks_authentic_double_and_excludes_gan_metrics(self):
        metrics = {"loss": 1.0, "drift_loss": 1.0, "g_norm": 2.0}
        gate.validate_step_metrics(metrics, variant="replay_only", step=3)
        with self.assertRaisesRegex(AssertionError, "Missing gate metrics"):
            gate.validate_step_metrics(metrics, variant="replay_double", step=3)
        doubled = {**metrics, "double_drift/c0": 0.75, "double_drift/c1": 0.25}
        gate.validate_step_metrics(doubled, variant="replay_double", step=3)
        for c0, c1 in ((1.0, 1.0), (1.0, 0.25), (0.75, 1.0), (0.25, 0.75)):
            with self.subTest(c0=c0, c1=c1):
                with self.assertRaisesRegex(AssertionError, "coefficients"):
                    gate.validate_step_metrics(
                        {**metrics, "double_drift/c0": c0, "double_drift/c1": c1},
                        variant="replay_double", step=3,
                    )
        with self.assertRaisesRegex(AssertionError, "Double Drift activated"):
            gate.validate_step_metrics(doubled, variant="replay_only", step=3)
        with self.assertRaisesRegex(AssertionError, "GAN metrics appeared"):
            gate.validate_step_metrics({**metrics, "feature_gan/d_loss": 1.0}, variant="baseline", step=0)

    def test_production_requires_the_same_source_and_full_three_arm_gate(self):
        snapshot = self.root / "gate_snapshot"
        snapshot.mkdir()
        manifest = {"commit": "a" * 40, "suite_id": "unit-suite"}
        manifest_path = snapshot / "source-manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "successful two-GPU validation report"):
            preflight.validate_gate_result(snapshot, manifest, self.suite)
        outcomes = [
            {"variant": name, "history_count": 0 if name == "baseline" else 16,
             "double_drift_mode": "feature" if name == "replay_double" else "off"}
            for name in preflight.VARIANTS
        ]
        report = {
            "kind": "corrective_field_gpu_validation", "status": "passed", **manifest,
            "source_manifest_sha256": preflight.sha256(manifest_path),
            "config_sha256": {name: preflight.sha256(self.suite / f"{name}.yaml") for name in preflight.VARIANTS},
            "world_size": 2, "steps_per_variant": 6,
            "geometry_per_rank": {"B": 8, "P": 32, "N": 32, "G": 32, "H_if_replay": 16},
            "ranks": [{"rank": rank, "variants": outcomes} for rank in range(2)],
        }
        report_path = snapshot / "validation-success.json"
        report_path.write_text(json.dumps(report))
        preflight.validate_gate_result(snapshot, manifest, self.suite)
        report["commit"] = "b" * 40
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "does not match the production suite"):
            preflight.validate_gate_result(snapshot, manifest, self.suite)

    def test_srv06_submits_typed_a5000_jobs_from_the_bound_suite_and_environment(self):
        shutil.copytree(self.suite, self.root / "configs/corrective_field_srv06")
        captured = {}

        def fake_snapshot(root, destination, suite_id, execution=None):
            destination.mkdir(parents=True)
            captured.update(execution)
            return {"commit": "a" * 40, "execution": execution}

        commands = []

        def fake_sbatch(command, **kwargs):
            commands.append(command)
            return str(9300 + len(commands)) + "\n"

        with (patch.object(submit, "ROOT", self.root),
              patch.object(submit, "make_snapshot", side_effect=fake_snapshot),
              patch.object(submit.subprocess, "check_output", side_effect=fake_sbatch),
              patch("sys.argv", ["submit_corrective_field.py", "--submit", "--node", "srv06"]),
              redirect_stdout(io.StringIO())):
            submit.main()
        self.assertEqual(captured["suite_dir"], "configs/corrective_field_srv06")
        self.assertEqual(captured["python"], "/data/juhyeong/venvs/replay-drift/bin/python")
        self.assertEqual(captured["run_root"], "/data/juhyeong/corrective-field-runs")
        self.assertIs(captured["nccl_p2p_disable"], True)
        self.assertEqual(len(commands), 4)
        for command in commands:
            self.assertIn("--partition=srv06", command)
            self.assertIn("--nodelist=srv06", command)
            self.assertIn("--gres=gpu:a5000:2", command)
        for command in commands[1:]:
            self.assertIn("--dependency=afterok:9301", command)
        record = json.loads(next(self.root.rglob("submission.json")).read_text())
        self.assertEqual(record["execution"], captured)

    def test_hardware_binding_rejects_wrong_nodes_and_gpu_models(self):
        binding = preflight.execution_binding("srv06")
        preflight.validate_hardware_binding(binding, "srv06.osilab.work", ["NVIDIA RTX A5000"] * 2)
        with self.assertRaisesRegex(ValueError, "Expected allocated node srv06"):
            preflight.validate_hardware_binding(binding, "srv02", ["NVIDIA RTX A5000"] * 2)
        with self.assertRaisesRegex(ValueError, "Expected srv06 a5000 GPUs"):
            preflight.validate_hardware_binding(binding, "srv06", ["NVIDIA GeForce RTX 3090"] * 2)
        with self.assertRaisesRegex(ValueError, "exactly 2 allocated GPUs"):
            preflight.validate_hardware_binding(binding, "srv06", ["NVIDIA RTX A5000"] * 4)
        with self.assertRaisesRegex(ValueError, "binding is inconsistent"):
            preflight.snapshot_execution({"execution": {**binding, "gpu_type": "rtx3090"}})

    def test_nccl_transport_is_bound_to_the_node_with_legacy_srv02_compatibility(self):
        for node, disabled in (("srv02", False), ("srv06", True)):
            binding = preflight.execution_binding(node)
            self.assertIs(binding["nccl_p2p_disable"], disabled)
            self.assertEqual(preflight.snapshot_execution({"execution": binding}), binding)
            with self.assertRaisesRegex(ValueError, "binding is inconsistent"):
                preflight.snapshot_execution({"execution": {**binding, "nccl_p2p_disable": not disabled}})
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                preflight.snapshot_execution({"execution": {**binding, "nccl_p2p_disable": int(disabled)}})
            del binding["nccl_p2p_disable"]
            if node == "srv02":
                self.assertEqual(preflight.snapshot_execution({"execution": binding}), binding)
            else:
                with self.assertRaisesRegex(ValueError, "binding is inconsistent"):
                    preflight.snapshot_execution({"execution": binding})
        self.assertIs(preflight.snapshot_execution({})["nccl_p2p_disable"], False)

    def test_runtime_rejects_unbound_transport_before_gpu_queries_and_wrong_node(self):
        binding = preflight.execution_binding("srv06", python=sys.executable)
        fake_cuda = Mock(device_count=Mock(return_value=2),
                         get_device_name=Mock(return_value="NVIDIA RTX A5000"))
        env = {"SLURM_JOB_ID": "123", "SLURMD_NODENAME": "srv06"}
        with (patch.dict("sys.modules", {"torch": Mock(cuda=fake_cuda)}),
              patch.dict(os.environ, env, clear=True)):
            with self.assertRaisesRegex(ValueError, "NCCL_P2P_DISABLE.*expected 1"):
                preflight.validate_allocated_hardware({"execution": binding})
            fake_cuda.device_count.assert_not_called()
            with patch.dict(os.environ, {"NCCL_P2P_DISABLE": "1"}):
                self.assertEqual(preflight.validate_allocated_hardware({"execution": binding}),
                                 ["NVIDIA RTX A5000"] * 2)
                with patch.dict(os.environ, {"SLURMD_NODENAME": "srv02"}):
                    with self.assertRaisesRegex(ValueError, "Expected allocated node srv06"):
                        preflight.validate_allocated_hardware({"execution": binding})
        binding = preflight.execution_binding("srv02", python=sys.executable)
        del binding["nccl_p2p_disable"]
        fake_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3090"
        with (patch.dict("sys.modules", {"torch": Mock(cuda=fake_cuda)}),
              patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURMD_NODENAME": "srv02"}, clear=True)):
            self.assertEqual(len(preflight.validate_allocated_hardware({"execution": binding})), 2)
            with patch.dict(os.environ, {"NCCL_P2P_DISABLE": "1"}):
                with self.assertRaisesRegex(ValueError, "NCCL_P2P_DISABLE.*expected 0"):
                    preflight.validate_allocated_hardware({"execution": binding})

    def test_both_wrapper_bootstraps_override_inherited_p2p_from_the_manifest(self):
        # Execute only the real shell bootstrap through its environment export;
        # no torch import, job submission, or GPU access occurs in this test.
        export = 'export NCCL_P2P_DISABLE="$CF_NCCL_P2P_DISABLE"'
        for node, expected in (("srv02", "0"), ("srv06", "1")):
            snapshot = self.root / (node + "_transport")
            source = snapshot / "source"
            (source / "scripts").mkdir(parents=True)
            shutil.copy(preflight.__file__, source / "scripts/preflight_corrective_field.py")
            (snapshot / "source-manifest.json").write_text(json.dumps({
                "execution": preflight.execution_binding(node),
            }))
            for name in ("run_corrective_field.sbatch", "validate_corrective_field.sbatch"):
                with self.subTest(node=node, wrapper=name):
                    script = (preflight.ROOT / "scripts/slurm" / name).read_text()
                    prefix, _ = script.split(export, 1)
                    bootstrap = prefix + export + '\nprintf "%s\\n" "$NCCL_P2P_DISABLE"\n'
                    run_root = self.root / (node + "_runs")
                    arguments = ([str(snapshot), "baseline", str(run_root), sys.executable]
                                 if name.startswith("run_") else
                                 [str(snapshot), sys.executable, str(run_root)])
                    env = {**os.environ, "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1",
                           "NCCL_P2P_DISABLE": "0" if expected == "1" else "1"}
                    env.pop("CORRECTIVE_VALIDATION_ROOT", None)
                    result = subprocess.run(["bash", "-c", bootstrap, name, *arguments],
                                            env=env, capture_output=True, text=True, check=True)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_srv06_workdir_and_resume_use_manifest_suite_instead_of_srv02_config(self):
        snapshot = self.root / "srv06_snapshot"
        suite = snapshot / "source/configs/corrective_field_srv06"
        shutil.copytree(self.suite, suite)
        runs = self.root / "srv06_runs"
        binding = preflight.execution_binding("srv06", run_root=runs)
        manifest = {"commit": "a" * 40, "execution": binding}
        (snapshot / "source-manifest.json").write_text(json.dumps(manifest))
        workdir = runtime.prepare_workdir(snapshot, "replay_only", runs, "9401", 0)
        self.assertEqual(workdir, runtime.prepare_workdir(snapshot, "replay_only", runs, "9401", 1))
        self.assertEqual((workdir / "run_metadata/config.yaml").read_bytes(), (suite / "replay_only.yaml").read_bytes())
        with self.assertRaisesRegex(ValueError, "Run root does not match"):
            runtime.prepare_workdir(snapshot, "replay_only", self.root / "wrong_runs", "9401", 1)
        with self.assertRaisesRegex(ValueError, "does not match the immutable execution"):
            preflight.selected_suite_dir(snapshot / "source", manifest, self.suite)


if __name__ == "__main__":
    unittest.main()
