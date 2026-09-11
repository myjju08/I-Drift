"""CPU checks of GPU-gate acceptance criteria and frozen history construction."""
import json

import pytest
import torch
from torch import nn

from scripts.validate_corrective_field import (
    assert_gradients, fixed_generated_history, module_hash,
    validate_step_metrics, write_json_new,
)


def gan_metrics(step):
    return {"loss": 2.0, "drift_loss": 1.0, "g_norm": 3.0,
            "adversarial/d_loss_finite": 1.0, "adversarial/d_grad_finite": 1.0,
            "adversarial/g_loss_finite": 1.0, "adversarial/g_grad_finite": 1.0,
            "adversarial/r1_applied": float(step % 16 == 0), "adversarial/r1_weighted": 0.01}


@pytest.mark.parametrize("step", [0, 1, 15, 16, 17])
def test_gan_acceptance_requires_finite_flags_and_exact_r1_cadence(step):
    metrics = gan_metrics(step)
    validate_step_metrics(metrics, variant="replay_double_gan", step=step)
    for key, bad in (("loss", float("nan")), ("g_norm", float("inf")),
                     ("adversarial/d_grad_finite", 0.0),
                     ("adversarial/r1_applied", 1.0 - metrics["adversarial/r1_applied"])):
        with pytest.raises(AssertionError):
            validate_step_metrics({**metrics, key: bad}, variant="replay_double_gan", step=step)
    del metrics["adversarial/g_loss_finite"]
    with pytest.raises(AssertionError, match="Missing gate metrics"):
        validate_step_metrics(metrics, variant="replay_double_gan", step=step)


def test_baseline_requires_generator_loss_and_gradient_without_gan_flags():
    validate_step_metrics({"loss": 1.0, "drift_loss": 1.0, "g_norm": 2.0}, variant="baseline", step=0)
    with pytest.raises(AssertionError):
        validate_step_metrics({"loss": 1.0, "drift_loss": 1.0}, variant="baseline", step=0)


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, labels, *, cfg_scale, train):
        assert train and torch.equal(cfg_scale, torch.full_like(cfg_scale, 2.0))
        return {"samples": torch.randn(len(labels), 4, 32, 32) * self.weight}


def test_fixed_history_preserves_rng_detaches_and_retains_generated_coordinates():
    generator = TinyGenerator()
    cfg = {"historical_gen_replay_count": 16, "seed": 43, "use_bf16": False,
           "historical_gen_replay_storage_dtype": "float16"}
    labels = torch.arange(8)
    state = torch.get_rng_state()
    first = fixed_generated_history(generator, labels, cfg, torch.device("cpu"))
    assert torch.equal(state, torch.get_rng_state())
    second = fixed_generated_history(generator, labels, cfg, torch.device("cpu"))
    assert torch.equal(first, second)
    assert first.shape == (8, 16, 4, 32, 32) and first.dtype == torch.float16
    assert not first.requires_grad and first.grad_fn is None and generator.weight.grad is None


def test_hash_covers_scalar_and_model_parameters_and_gradient_checks():
    model = TinyGenerator()
    before = module_hash(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert module_hash(model) != before
    with pytest.raises(AssertionError):
        assert_gradients(model)
    model.weight.square().backward()
    assert_gradients(model)
    with pytest.raises(AssertionError):
        assert_gradients(model, frozen=True)
    model.weight.grad = None
    model.requires_grad_(False)
    assert_gradients(model, frozen=True)


def test_success_report_never_overwrites_prior_evidence_or_accepts_nan(tmp_path):
    path = tmp_path / "validation-success.json"
    write_json_new(path, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_json_new(path, {"status": "replacement"})
    assert json.loads(path.read_text()) == {"status": "passed"}
    with pytest.raises(ValueError):
        write_json_new(tmp_path / "invalid.json", {"seconds": float("nan")})
    assert not (tmp_path / "invalid.json").exists()
