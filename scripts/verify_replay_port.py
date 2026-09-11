"""Verify the replay port against the pinned upstream Git snapshot on CPU.

Example:
    python scripts/verify_replay_port.py --source ../I-Drift-cosmos \
        --report docs/REPLAY_PARITY.json

The source checkout must contain the pinned commit; its current worktree is
not trusted or modified. No checkpoints, datasets, W&B runs, or GPUs are used.
"""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import json

import numpy as np
import torch

SOURCE_COMMIT = '27a0e4fb54c8ed04e2538831a9df86a85c8c25ef'
TARGET_BASE = '4c26125a6acdd1cff488b6904025996bbb6b8dc5'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_text(text, name):
    tree = ast.parse(text)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    return ast.get_source_segment(text, node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Checkout containing the pinned cosmosjhj source commit')
    parser.add_argument('--target', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, help='Optional JSON output path')
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    TARGET = args.target.resolve() / 'memory_bank.py'
    source_text = subprocess.check_output(
        ['git', '-C', str(SOURCE), 'show', SOURCE_COMMIT + ':memory_bank.py'], text=True)
    target_text = TARGET.read_text()

    with tempfile.TemporaryDirectory() as directory:
        pinned_source = Path(directory) / 'memory_bank.py'
        pinned_source.write_text(source_text)
        source = load('source_replay', pinned_source)
    target = load('target_replay', TARGET)
    assert class_text(source_text, 'HistoricalReplayMemoryBank') == class_text(target_text, 'HistoricalReplayMemoryBank')
    previous = subprocess.check_output(['git', '-C', str(TARGET.parent), 'show', TARGET_BASE + ':memory_bank.py'], text=True)
    for name in ['ArrayMemoryBank', 'CompressedPixelMemoryBank']:
        assert class_text(previous, name) == class_text(target_text, name), name
    comparisons = 0


    def compare(first, second, step):
        nonlocal comparisons
        for key in ['bank', 'ptr', 'count', 'use_count', 'insert_step', 'seen_count']:
            np.testing.assert_array_equal(getattr(first, key), getattr(second, key), err_msg=key)
            comparisons += 1
        for key in ['sample_count', 'replacement_count', 'discard_count']:
            assert getattr(first, key) == getattr(second, key), key
            comparisons += 1
        assert first.metrics(step=step) == second.metrics(step=step)
        comparisons += len(first.metrics(step=step))


    cases = 0
    for policy in source.HistoricalReplayMemoryBank.POLICIES:
        for dtype in [np.float16, np.float32]:
            for isolated in [False, True]:
                opts = dict(num_classes=3, max_size=5, dtype=dtype, policy=policy, usage_budget=3)
                first, second = source.HistoricalReplayMemoryBank(**opts), target.HistoricalReplayMemoryBank(**opts)
                samples = np.random.default_rng(619).normal(size=(21, 2, 3)).astype(np.float32)
                labels = np.array([0,1,2] * 7)
                for bank in [first, second]:
                    bank.add(samples, labels, step=10)
                compare(first, second, 10)
                for step in range(11, 41):
                    sampled = []
                    for bank in [first, second]:
                        np.random.seed(312 + step)
                        sampled.append(bank.sample(np.array([0,2,1,0]), 7,
                            rng=np.random.default_rng([823,step]) if isolated else None))
                        bank.update(samples[:9] + step, labels[:9], step=step,
                            rng=np.random.default_rng([472,step]) if isolated else None)
                    assert torch.equal(*sampled)
                    comparisons += 1
                    compare(first, second, step)
                    if step == 25:
                        with tempfile.TemporaryDirectory() as directory:
                            old = [first, second]
                            restored = []
                            for i, module in enumerate([source, target]):
                                path = Path(directory)/f'bank{i}.npz'
                                old[i].save_npz(path)
                                bank = module.HistoricalReplayMemoryBank(**opts)
                                bank.load_npz(path, default_step=10)
                                restored.append(bank)
                            first, second = restored
                            compare(first, second, step)
                cases += 1
    report = dict(cases=cases, comparisons=comparisons, max_absolute_error=0,
                  class_source_identical=True, original_target_bank_classes_unchanged=True,
                  policies=list(source.HistoricalReplayMemoryBank.POLICIES),
                  source_repository='https://github.com/cosmosjhj/I-Drift',
                  source_commit=SOURCE_COMMIT,
                  target_repository='https://github.com/myjju08/I-Drift',
                  target_base_commit=TARGET_BASE,
                  hashes={
                      'source_memory_bank_sha256': hashlib.sha256(source_text.encode()).hexdigest(),
                      'target_memory_bank_sha256': hashlib.sha256(target_text.encode()).hexdigest(),
                      'historical_replay_class_sha256': hashlib.sha256(class_text(source_text, 'HistoricalReplayMemoryBank').encode()).hexdigest(),
                  },
                  scope='CPU state/sample/telemetry parity, four policies, float16/float32 storage, global/isolated RNG, 30 streaming steps with midpoint NPZ resume')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
