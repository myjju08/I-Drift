"""CPU preflight and comparison-contract checks for DINO Double Drift."""
import copy

import pytest
import torch

from experiments.dino.train import (
    ROOT, load_config, preflight, validate_double_drift_config,
)


def config(mode):
    return load_config(config=ROOT / 'configs' / f'double_drift_{mode}.yaml')


@pytest.mark.parametrize('name,mode', [
    ('baseline', 'off'), ('feature', 'feature'), ('sample', 'sample'),
])
def test_cpu_preflight_selects_dino_double_drift_without_cuda(name, mode):
    cfg = config(name)
    report = preflight(cfg, io_backend='standard')
    assert report['passed'] and not report['cuda_initialized']
    assert not torch.cuda.is_initialized()
    assert report['feature_extractor'] == 'dino_resnet50'
    assert report['double_drift']['mode'] == mode
    assert report['generated_per_step'] == 256
    assert report['replay_ratio_before'] == report['replay_ratio_after'] == 0
    assert cfg['adversarial_mode'] == 'none'
    assert not cfg['feature_adapter'] and not cfg['feature_gan']
    assert cfg['activation_kwargs']['active_stages'] == ['stage3', 'stage4']
    assert cfg['total_generated_epochs'] == 40
    if mode != 'off':
        assert report['double_drift']['c0'] == .75
        assert report['double_drift']['c1'] == .25
        assert report['double_drift']['generated_field_evaluations'] == 2
    if mode == 'sample':
        assert report['double_drift']['sample_step_rms'] == .1
        assert report['double_drift']['sample_coordinates'] == 'rgb'


def test_comparison_changes_only_name_and_double_drift_objective():
    baseline = copy.deepcopy(config('baseline')['_raw'])
    for mode in ('feature', 'sample'):
        actual = copy.deepcopy(config(mode)['_raw'])
        for raw in (baseline, actual):
            raw['logging'].pop('name', None)
            for key in ('double_drift_mode', 'double_drift_c0', 'double_drift_c1'):
                raw['train'].pop(key, None)
        assert actual == baseline


@pytest.mark.parametrize('changes,error', [
    ({'drift_matching': 'dual-drift'}, 'rev-drift'),
    ({'drift_matching': 'fwd-drift'}, 'rev-drift'),
    ({'feature_extractor': 'mae'}, 'dino_resnet50'),
    ({'feature_extractor': 'moco_v2_resnet50'}, 'dino_resnet50'),
    ({'adversarial_mode': 'raw_gan'}, 'adapters/GAN'),
    ({'adversarial_mode': 'feature_drift'}, 'adapters/GAN'),
    ({'feature_gan': True}, 'adapters/GAN'),
    ({'feature_adapter': True}, 'adapters/GAN'),
    ({'double_drift_c0': -1}, 'non-negative'),
    ({'double_drift_c0': 0, 'double_drift_c1': 0}, 'positive'),
    ({'double_drift_c1': float('nan')}, 'finite'),
    ({'double_drift_mode': 'dual'}, 'off, feature, or sample'),
    ({'double_drift_sample_step_rms': 0}, 'finite and positive'),
    ({'double_drift_sample_step_rms': float('inf')}, 'finite and positive'),
])
def test_invalid_double_drift_configs_fail_before_training(changes, error):
    cfg = config('sample')
    cfg.update(changes)
    with pytest.raises(ValueError, match=error):
        validate_double_drift_config(cfg)


def test_one_field_sample_control_is_reported_and_existing_presets_stay_supported():
    cfg = config('sample')
    cfg.update(double_drift_c0=1., double_drift_c1=0.)
    assert validate_double_drift_config(cfg)['generated_field_evaluations'] == 1
    # Existing adversarial presets do not enable Double Drift.
    assert validate_double_drift_config(load_config('replay_raw_gan_d32')) == {'mode': 'off'}


@pytest.mark.parametrize('disabled_mode', ['', 'off'])
def test_disabled_adversarial_aliases_match_the_existing_trainer(disabled_mode):
    cfg = config('feature')
    cfg['adversarial_mode'] = disabled_mode
    assert validate_double_drift_config(cfg)['mode'] == 'feature'
