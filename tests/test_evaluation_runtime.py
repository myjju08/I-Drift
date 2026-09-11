"""Metric integrity, RNG isolation and collective failures without GPU work."""
import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import train_imagenet_gen as trainer
from train.evaluation_runtime import (
    preserve_evaluation_rng,
    run_rank_zero_evaluation,
    validate_evaluation_metrics,
)


VALID = {"fid": 3.5, "is_mean": 2.25, "is_std": 0.125}


def _rng_snapshot():
    return torch.get_rng_state().clone(), np.random.get_state()


def _assert_rng_equal(expected):
    actual = _rng_snapshot()
    assert torch.equal(actual[0], expected[0])
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]


def _consume_rng():
    torch.rand(5)
    np.random.random(5)


class _Model:
    def __init__(self):
        self.moves = []
        self.train_calls = 0
        self.eval_calls = 0

    def to(self, device):
        self.moves.append(str(device))
        _consume_rng()  # Makes model restore part of the RNG regression test.
        return self

    def eval(self):
        self.eval_calls += 1
        _consume_rng()
        return self

    def train(self):
        self.train_calls += 1
        _consume_rng()
        return self


def _arguments(step=0, **overrides):
    args = dict(
        cfg={"require_eval_metrics": True, "preserve_rng_during_eval": True},
        rank=0, world_size=1, device=torch.device("cpu"), step=step,
        generated_epochs=0.0 if step == 0 else 10.0,
        generator=_Model(), feature_extractor=_Model(),
        ema=SimpleNamespace(shadow=_Model()), pos_banks=[],
        eval_loader=object(), eval_postprocess_fn=lambda x: x,
        cfg_list=[1.0, 2.0, 3.0, 4.0], eval_samples=1024,
        workdir=Path("unused-eval-test"), logger=mock.Mock(),
    )
    args.update(overrides)
    return args


@pytest.fixture(autouse=True)
def _isolate_test_rng():
    # No CUDA allocations: CUDA preservation is exercised with mocked states.
    with preserve_evaluation_rng(True), mock.patch.object(torch.cuda, "is_available", return_value=False):
        yield


@pytest.mark.parametrize("name", ["fid", "is_mean", "is_std"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), "not-a-number"])
def test_required_metrics_reject_missing_or_nonfinite(name, value):
    stats = dict(VALID, **{name: value})
    with pytest.raises(RuntimeError, match=f"step 0, CFG=2.0.*{name}"):
        validate_evaluation_metrics(stats, required=True, step=0, cfg_scale=2.0)


def test_metrics_guard_is_opt_in_and_accepts_zero_std():
    validate_evaluation_metrics({}, step=0, cfg_scale=1.0)
    validate_evaluation_metrics(dict(VALID, is_std=0), required=True, step=0, cfg_scale=1.0)


@pytest.mark.parametrize("step", [0, 1252])
def test_valid_metrics_all_cfg_are_logged_without_protocol_changes(step):
    args = _arguments(step)
    snapshot = _rng_snapshot()
    with mock.patch.object(trainer, "eval_fid_is", return_value=VALID) as evaluate:
        trainer._run_generator_evaluation(**args)
    assert evaluate.call_count == 3
    for call, scale in zip(evaluate.call_args_list, [1.0, 2.0, 3.0]):
        assert call.kwargs == dict(
            cfg_scale=scale, n_samples=1024, workdir=args["workdir"],
            step=step, label=f"CFG{scale}",
        )
    logged = args["logger"].log.call_args
    assert logged.kwargs == ({"step": 0, "commit": False} if step == 0 else {"step": step})
    assert logged.args[0] == {
        "eval/generated_epochs": 0.0 if step == 0 else 10.0,
        **{f"{prefix}/cfg{scale}": VALID[name]
           for scale in [1.0, 2.0, 3.0]
           for prefix, name in [("fid", "fid"), ("is", "is_mean"), ("is_std", "is_std")]},
    }
    assert args["generator"].train_calls == 1
    assert args["feature_extractor"].moves == ["cpu", "cpu"]
    _assert_rng_equal(snapshot)


@pytest.mark.parametrize("step", [0, 1252])
@pytest.mark.parametrize("missing", ["eval_loader", "eval_postprocess_fn"])
def test_required_missing_loader_fails_before_evaluation(step, missing):
    args = _arguments(step, **{missing: None})
    with mock.patch.object(trainer, "eval_fid_is") as evaluate:
        with pytest.raises(RuntimeError, match="Required rank-0 evaluation failed"):
            trainer._run_generator_evaluation(**args)
    evaluate.assert_not_called()
    args["logger"].log.assert_not_called()


def test_missing_cfg_scales_fail_strict():
    with pytest.raises(RuntimeError, match="no CFG scales"):
        trainer._run_generator_evaluation(**_arguments(cfg_list=[]))


@pytest.mark.parametrize("step", [0, 1252])
@pytest.mark.parametrize("failure", ["missing_metric", "exception", "cleanup"])
def test_failure_restores_rng_and_attempts_all_model_and_bank_cleanup(step, failure):
    bank = mock.Mock(spec=trainer.CompressedPixelMemoryBank)
    bank.suspend_codec_workers.side_effect = _consume_rng
    bank.resume_codec_workers.side_effect = _consume_rng
    args = _arguments(step, pos_banks=[bank])
    feature = args["feature_extractor"]
    if failure == "cleanup":
        original_to = feature.to

        def fail_restore(device):
            result = original_to(device)
            if len(feature.moves) == 2:
                raise RuntimeError("mock feature restore failure")
            return result

        feature.to = fail_restore

    def evaluate(*unused_args, **unused_kwargs):
        _consume_rng()
        if failure == "exception":
            raise ValueError("mock metric failure")
        return {"fid": None} if failure == "missing_metric" else VALID

    snapshot = _rng_snapshot()
    with mock.patch.object(trainer, "eval_fid_is", side_effect=evaluate):
        with pytest.raises(RuntimeError, match="Required rank-0 evaluation failed"):
            trainer._run_generator_evaluation(**args)
    _assert_rng_equal(snapshot)
    bank.suspend_codec_workers.assert_called_once_with()
    bank.resume_codec_workers.assert_called_once_with()
    assert args["generator"].train_calls == 1
    assert args["generator"].moves == ["cpu", "cpu"]
    assert feature.moves == ["cpu", "cpu"]
    if failure != "cleanup":
        args["logger"].log.assert_not_called()


def test_guards_default_to_legacy_warn_and_rng_advancement():
    args = _arguments(cfg={})
    before = torch.get_rng_state().clone()
    with mock.patch.object(trainer, "eval_fid_is", side_effect=ValueError("legacy skip")):
        trainer._run_generator_evaluation(**args)
    assert not torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("fail_release", [False, True])
def test_decoder_release_precedes_feature_restore_and_failure_is_strict(fail_release):
    events = []
    args = _arguments()
    feature = args["feature_extractor"]
    original_to = feature.to

    def move_feature(device):
        events.append("feature_to")
        return original_to(device)

    def release():
        events.append("decoder_release")
        _consume_rng()
        if fail_release:
            raise RuntimeError("mock decoder release failure")

    feature.to = move_feature
    args["eval_postprocess_fn"] = SimpleNamespace(release=release)
    before = _rng_snapshot()
    with mock.patch.object(trainer, "eval_fid_is", return_value=VALID):
        if fail_release:
            with pytest.raises(RuntimeError, match="mock decoder release failure"):
                trainer._run_generator_evaluation(**args)
        else:
            trainer._run_generator_evaluation(**args)
    assert events == ["feature_to", "decoder_release", "feature_to"]
    assert args["generator"].train_calls == 1
    _assert_rng_equal(before)


def test_only_rank_local_cuda_rng_is_restored_even_after_exception():
    cuda_state = torch.tensor([1, 2], dtype=torch.uint8)
    device = torch.device("cuda:1")
    with mock.patch.object(torch.cuda, "is_initialized", return_value=True), \
         mock.patch.object(torch.cuda, "get_rng_state", return_value=cuda_state) as get_state, \
         mock.patch.object(torch.cuda, "set_rng_state") as set_state, \
         mock.patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("foreign CUDA device")):
        with pytest.raises(ValueError):
            with preserve_evaluation_rng(True, device=device):
                _consume_rng()
                raise ValueError("evaluation failed")
    get_state.assert_called_once_with(device)
    set_state.assert_called_once_with(cuda_state, device)


def test_cpu_rng_preservation_never_initializes_or_queries_cuda():
    with mock.patch.object(torch.cuda, "is_initialized", side_effect=AssertionError("CUDA query")), \
         mock.patch.object(torch.cuda, "get_rng_state", side_effect=AssertionError("CUDA RNG query")):
        before = _rng_snapshot()
        with preserve_evaluation_rng(True, device=torch.device("cpu")):
            _consume_rng()
        _assert_rng_equal(before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="An allocated CUDA GPU is required")
def test_allocated_cuda_rank_local_stream_is_unchanged_by_evaluation():
    device = torch.device("cuda", torch.cuda.current_device())
    cuda_before = torch.cuda.get_rng_state(device).clone()
    host_before = _rng_snapshot()
    with preserve_evaluation_rng(True, device=device):
        _consume_rng()
        torch.rand(5, device=device)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)
    _assert_rng_equal(host_before)


def _distributed_evaluation_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        for step in [0, 1252]:
            for failure in ["metric", "loader", "cleanup", "success"]:
                args = _arguments(step, rank=rank, world_size=2)
                if failure == "loader":
                    args["eval_loader"] = None
                if failure == "cleanup":
                    feature = args["feature_extractor"]
                    original_to = feature.to

                    def fail_restore(device):
                        result = original_to(device)
                        if len(feature.moves) == 2:
                            raise RuntimeError("DDP mock restoration failure")
                        return result

                    feature.to = fail_restore
                stats = dict(VALID, is_std=float("nan")) if failure == "metric" else VALID
                before = _rng_snapshot()
                raised = False
                with mock.patch.object(torch.cuda, "is_available", return_value=False), \
                     mock.patch.object(trainer, "eval_fid_is", return_value=stats) as evaluate:
                    try:
                        trainer._run_generator_evaluation(**args)
                    except RuntimeError as exc:
                        raised = True
                        assert "Required rank-0 evaluation failed" in str(exc)
                assert raised == (failure != "success")
                assert evaluate.call_count == (3 if rank == 0 and failure in ("success", "cleanup") else
                                                1 if rank == 0 and failure == "metric" else 0)
                # Even rank 0 never moved the DDP generator off its device.
                assert args["generator"].moves == []
                if rank == 0 or failure != "success":
                    _assert_rng_equal(before)
                # Reaching another collective proves both ranks caught the
                # outcome; no rank remained blocked at an eval barrier.
                proof = torch.tensor(int(raised))
                dist.all_reduce(proof)
                assert proof.item() == (0 if failure == "success" else 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_strict_failures_propagate_after_cleanup(tmp_path):
    mp.spawn(_distributed_evaluation_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
