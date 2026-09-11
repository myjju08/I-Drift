"""Required online logging must fail closed without affecting other DDP ranks."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from train_imagenet_gen import Logger


def _wandb(mode="online", *, error=None, missing_run=False):
    module = types.ModuleType("wandb")
    module.run = None if missing_run else types.SimpleNamespace(
        id="test-run", settings=types.SimpleNamespace(mode=mode),
        url="https://wandb.ai/test-entity/DINO/runs/test-run",
    )
    module.Settings = mock.Mock(side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))
    module.init = mock.Mock(side_effect=error, return_value=module.run)
    module.log = mock.Mock()
    module.finish = mock.Mock()
    return module


class RequiredWandbTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="required-wandb-")
        self.addCleanup(directory.cleanup)
        self.workdir = Path(directory.name)
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))

    def logger(self, module, **overrides):
        config = {
            "use_wandb": True, "require_wandb": True, "project": "DINO",
            "entity": "test-entity", "console_log": False,
        }
        rank = overrides.pop("rank", 0)
        config.update(overrides)
        with mock.patch.dict("sys.modules", {"wandb": module}):
            return Logger(str(self.workdir), config, rank)

    def test_required_logging_rejects_disabled_configuration(self):
        module = _wandb()
        with self.assertRaisesRegex(ValueError, "requires use_wandb=true"):
            self.logger(module, use_wandb=False)
        module.init.assert_not_called()

    def test_required_initialization_failure_stops_training(self):
        failure = RuntimeError("authentication unavailable")
        with self.assertRaisesRegex(RuntimeError, "Online W&B logging is required") as raised:
            self.logger(_wandb(error=failure))
        self.assertIs(raised.exception.__cause__, failure)
        self.assertFalse((self.workdir / "wandb_run_id.txt").exists())

    def test_required_logging_rejects_non_online_or_missing_runs(self):
        for mode in ("offline", "dryrun", "disabled", None):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "Online W&B logging is required") as raised:
                    self.logger(_wandb(mode))
                self.assertIn("not online", str(raised.exception.__cause__))
                self.assertFalse((self.workdir / "wandb_run_id.txt").exists())
        with self.assertRaisesRegex(RuntimeError, "Online W&B logging is required") as raised:
            self.logger(_wandb(missing_run=True))
        self.assertIn("no active run", str(raised.exception.__cause__))

    def test_online_run_persists_id_prints_url_and_logs(self):
        module = _wandb()
        logger = self.logger(module)
        self.assertTrue(logger.use_wandb)
        self.assertEqual((self.workdir / "wandb_run_id.txt").read_text(), "test-run")
        self.assertIn(module.run.url, self.output.getvalue())
        self.assertEqual(module.init.call_args.kwargs["project"], "DINO")
        logger.log({"loss": 1.25}, step=3)
        module.log.assert_called_once_with({"loss": 1.25}, step=3, commit=True)

    def test_worker_rank_does_not_initialize_or_require_its_own_run(self):
        module = _wandb(missing_run=True)
        logger = self.logger(module, rank=1)
        self.assertFalse(logger.use_wandb)
        module.init.assert_not_called()
        logger.log({"loss": 1.25}, step=3)
        module.log.assert_not_called()
        self.assertFalse((self.workdir / "wandb_run_id.txt").exists())

    def test_default_preserves_local_fallback_after_initialization_failure(self):
        module = _wandb(error=RuntimeError("authentication unavailable"))
        with mock.patch.dict("sys.modules", {"wandb": module}):
            logger = Logger(str(self.workdir), {"use_wandb": True}, rank=0)
        self.assertFalse(logger.use_wandb)
        logger.log({"loss": 1.25}, step=3)
        self.assertEqual(
            json.loads((self.workdir / "train_log.jsonl").read_text()),
            {"step": 3, "loss": 1.25},
        )
        module.log.assert_not_called()

    def test_optional_logging_still_allows_offline_mode(self):
        logger = self.logger(_wandb("offline"), require_wandb=False)
        self.assertTrue(logger.use_wandb)


if __name__ == "__main__":
    unittest.main()
