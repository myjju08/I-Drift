"""Independent, subprocess-isolated parity checks for Replay, Double Drift, and force balance.

No repository imports occur in the coordinator; source and target run in separate processes.
Run with the same Python environment used for training:
  python scripts/verify_double_drift_port.py --source UPSTREAM_CHECKOUT --target THIS_CHECKOUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def worker(repo, output, scope="all"):
    sys.path.insert(0, str(Path(repo).resolve()))
    import numpy as np
    import torch
    from drifting_core.imagenet_loss import drift_loss_imagenet, reverse_drift_field
    from drifting_core.double_drift import sample_double_drift_loss
    from drifting_core.force_balance import balance_coefficients
    from train_imagenet_gen import compute_drift_loss_from_features

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records = {}
    B, G, P, N, H, S = 2, 3, 3, 3, 5, 7

    def convert(value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise AssertionError(f"Nonfinite tensor: {array}")
            return {"dtype": str(array.dtype), "shape": list(array.shape), "values": array.tolist()}
        if isinstance(value, dict):
            return {key: convert(item) for key, item in sorted(value.items())}
        if isinstance(value, (int, float)):
            if not np.isfinite(value):
                raise AssertionError(f"Nonfinite scalar: {value}")
            return float(value)
        raise TypeError(type(value))

    def weights(dtype, history):
        return dict(
            weight_neg=torch.tensor([[0.5, 1.0, 1.5], [0.4, 0.9, 1.6]], dtype=dtype),
            weight_gen=torch.full((B, G), 0.65 if history else 1.0, dtype=dtype),
            weight_history=torch.full((B, H), 0.35 * G / H, dtype=dtype) if history else None,
        )

    # Static attraction/repulsion changes plus mid/end linear annealing.
    for delta, step, anneal_steps in [(0.0, 0, 0), (0.05, 0, 0), (-0.05, 0, 0),
                                     (-0.1, 5, 10), (0.1, 10, 10)]:
        attraction, repulsion = balance_coefficients(delta, step, anneal_steps)
        for dtype in [torch.float32, torch.float64]:
            for global_stats in [False, True]:
                for history in [False, True]:
                    for c0, c1 in [(1.0, 0.0), (1.0, 1.0), (0.75, 0.25)]:
                        prefix = f'balance={delta},{step},{anneal_steps}/{dtype}/global={global_stats}/history={history}/c={c0},{c1}'
                        torch.manual_seed(7614)
                        values = torch.randn(B, G, S, dtype=dtype)
                        pos = torch.randn(B, P, S, dtype=dtype) + 0.25
                        neg = torch.randn(B, N, S, dtype=dtype) - 0.25
                        hist = torch.randn(B, H, S, dtype=dtype) if history else None
                        args = dict(fixed_pos=pos.float(), fixed_neg=neg.float(),
                                    historical_gen=hist.float() if hist is not None else None,
                                    **weights(torch.float32, history),
                                    attraction_scale=attraction, repulsion_scale=repulsion,
                                    balance_diagnostics=True,
                                    global_scale_stats=global_stats, global_fnorm_stats=global_stats)
                        gen = values.clone().requires_grad_()
                        loss, info = drift_loss_imagenet(gen.float(), **args,
                            double_drift_c0=c0, double_drift_c1=c1)
                        grad = torch.autograd.grad(loss.sum(), gen)[0]
                        field, scale, field_info = reverse_drift_field(gen.float(), **args)
                        records['core/'+prefix] = convert(dict(loss=loss, grad=grad, field=field,
                            scale=scale, info=info, field_info=field_info))
                        if scope == 'core':
                            continue

                        # Two genuinely different differentiable feature maps,
                        # with P=N=G=3 and spatial-token batch folding enabled.
                        torch.manual_seed(27951)
                        sample = torch.randn(B * G, 4, 2, 2, dtype=dtype)
                        pos_sample = torch.randn(B * P, 4, 2, 2, dtype=dtype) + 0.2
                        neg_sample = torch.randn(B * N, 4, 2, 2, dtype=dtype) - 0.2
                        history_sample = torch.randn(B * H, 4, 2, 2, dtype=dtype) if history else None

                        def features(x):
                            # The authentic core supports canonical FP32 features;
                            # preserve an FP64 outer graph when requested.
                            flat = x.flatten(1).float()
                            return {'layer3': torch.sin(flat[:, :12]).reshape(-1, 3, 4),
                                    'layer4': (flat[:, :10].square() * 0.1 + flat[:, :10]).reshape(-1, 2, 5)}

                        fargs = dict(pos_feats=features(pos_sample), neg_feats=features(neg_sample),
                                     B=B, G=G, P=P, N=N, **weights(torch.float32, history),
                                     historical_feats=features(history_sample) if history else None,
                                     historical_count=H if history else 0,
                                     global_scale_stats=global_stats, global_fnorm_stats=global_stats,
                                     feature_loss_weights={'layer3': 0.8, 'layer4': 1.2},
                                     rev_drift_attraction_scale=attraction,
                                     rev_drift_repulsion_scale=repulsion,
                                     rev_drift_balance_diagnostics=True)
                        sample_feature = sample.clone().requires_grad_()
                        loss, info = compute_drift_loss_from_features(
                            gen_feats=features(sample_feature), **fargs,
                            double_drift_c0=c0, double_drift_c1=c1)
                        grad = torch.autograd.grad(loss, sample_feature)[0]
                        records['feature_aggregation/'+prefix] = convert(dict(loss=loss, grad=grad, info=info))

                        sample_actual = sample.clone().requires_grad_()
                        def loss_at(x):
                            return compute_drift_loss_from_features(gen_feats=features(x), **fargs)
                        first_loss, first_info = loss_at(sample_actual)
                        loss, info = sample_double_drift_loss(sample_actual, first_loss, loss_at,
                            c0=c0, c1=c1, step_rms=0.1, global_stats=global_stats)
                        grad = torch.autograd.grad(loss, sample_actual)[0]
                        records['sample/'+prefix] = convert(dict(loss=loss, grad=grad,
                                                                 info=info, first_info=first_info))
    Path(output).write_text(json.dumps(records, sort_keys=True))
    print(f'{repo}: {len(records)} deterministic CPU cases saved to {output}', flush=True)


def compare(expected, actual):
    import numpy as np
    mismatches = []
    extra = []
    comparisons = 0
    max_error = 0.0

    def visit(left, right, path):
        nonlocal comparisons, max_error
        if isinstance(left, dict):
            missing = left.keys() - right.keys()
            additional = right.keys() - left.keys()
            if missing:
                mismatches.append(f'{path}: missing {sorted(missing)}')
            if additional:
                extra.append(f'{path}: additional {sorted(additional)}')
            if 'values' in left and 'dtype' in left:
                if left['shape'] != right['shape'] or left['dtype'] != right['dtype']:
                    mismatches.append(f'{path}: shape/dtype changed')
                    return
                a, b = np.asarray(left['values']), np.asarray(right['values'])
                err = float(np.max(np.abs(a-b))) if a.size else 0.0
                max_error = max(max_error, err)
                comparisons += 1
                if not np.array_equal(a, b):
                    mismatches.append(f'{path}: max_abs_error={err}')
                return
            for key in left.keys() & right.keys():
                visit(left[key], right[key], f'{path}/{key}')
        else:
            comparisons += 1
            err = abs(left-right)
            max_error = max(max_error, err)
            if left != right:
                mismatches.append(f'{path}: {left} != {right} (abs={err})')

    visit(expected, actual, '')
    return dict(cases=len(expected), exact_comparisons=comparisons,
                max_absolute_error=max_error, mismatches=mismatches, extra_fields=extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Checkout of the upstream source commit')
    parser.add_argument('--target', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parents[1] / 'runs/double-drift-parity'))
    parser.add_argument('--worker', nargs=2, metavar=('REPO', 'OUTPUT'))
    parser.add_argument('--scope', choices=['all', 'core'], default='all')
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker, scope=args.scope)
        return
    if not args.source:
        parser.error('--source is required for a source/target comparison')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, repo in [('source', args.source), ('target', args.target)]:
        subprocess.run([sys.executable, __file__, '--worker', repo, str(output/(name+'.json')), '--scope', args.scope],
                       cwd=repo, check=True)
    report = compare(json.loads((output/'source.json').read_text()),
                     json.loads((output/'target.json').read_text()))
    report['scope'] = args.scope
    report['balance_cases'] = ['delta=0', 'delta=+0.05', 'delta=-0.05', 'delta=-0.1 annealed halfway', 'delta=+0.1 annealed to zero']
    report['feature_dtype'] = 'float32 (canonical upstream core; outer sample graph float32 and float64)'
    report['source_commit'] = subprocess.check_output(['git', '-C', args.source, 'rev-parse', 'HEAD'], text=True).strip()
    report['target_base_commit'] = subprocess.check_output(['git', '-C', args.target, 'rev-parse', 'HEAD'], text=True).strip()
    import hashlib
    report['validated_files_sha256'] = {name: hashlib.sha256((Path(args.target) / name).read_bytes()).hexdigest()
        for name in ['drifting_core/double_drift.py', 'drifting_core/force_balance.py',
                     'drifting_core/imagenet_loss.py', 'train_imagenet_gen.py']}
    report['command'] = 'python scripts/verify_double_drift_port.py --source /path/to/cosmosjhj-I-Drift-at-27a0e4f'
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report['mismatches']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
