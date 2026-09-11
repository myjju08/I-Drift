"""Compare a pinned upstream Git tree with this checkout's source files.

This inventories bytes and Python definitions; numerical parity is checked by
verify_double_drift_port.py and verify_replay_port.py separately.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SELF_REPORTS = {"docs/UPSTREAM_PARITY.json", "docs/UPSTREAM_PARITY.md"}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def definitions(data):
    tree = ast.parse(data)
    result = {}

    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                result[name] = ast.dump(node, include_attributes=False)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + ".")
    visit(tree.body)
    return result


def category(name, status):
    if status == "source_only":
        return "upstream_workflow_or_test_not_copied" if name.startswith(("scripts/", "tests/")) else "upstream_only"
    if name.startswith(("experiments/dino/", "experiments/mae/")) or any(token in name.lower() for token in ("dino", "gan", "adversarial", "adapter", "ssl_", "moco")):
        return "dino_gan_or_encoder_adapter_extension"
    if name in {"drifting_core/imagenet_loss.py", "memory_bank.py", "train_imagenet_gen.py"}:
        return "integrated_shared_science_and_target_runtime"
    if name.startswith(("models/", "train/")) or name == "vae_imagenet.py":
        return "shared_model_data_runtime"
    if name.startswith("tests/"):
        return "regression_test"
    if name.startswith("configs/"):
        return "experiment_configuration"
    if name.startswith("scripts/"):
        return "launcher_validation_or_evaluation"
    return "documentation_dependencies_or_repository_metadata"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--target", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/UPSTREAM_PARITY.json")
    args = parser.parse_args()
    commit = git(args.source, "rev-parse", args.source_ref).decode().strip()
    source_names = set(git(args.source, "ls-tree", "-r", "--name-only", commit).decode().splitlines())
    target_names = set(git(args.target, "ls-files", "--cached", "--others", "--exclude-standard").decode().splitlines())
    rows = []
    for name in sorted((source_names | target_names) - SELF_REPORTS):
        left = git(args.source, "show", f"{commit}:{name}") if name in source_names else None
        right = (args.target / name).read_bytes() if name in target_names else None
        status = "target_only" if left is None else "source_only" if right is None else "identical" if left == right else "different"
        row = {"path": name, "status": status, "classification": category(name, status),
               "source_sha256": hashlib.sha256(left).hexdigest() if left is not None else None,
               "target_sha256": hashlib.sha256(right).hexdigest() if right is not None else None}
        if name.endswith(".py") and left is not None and right is not None and status == "different":
            a, b = definitions(left), definitions(right)
            row["python_definitions"] = {
                "identical": sorted(k for k in a.keys() & b.keys() if a[k] == b[k]),
                "different": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
                "source_only": sorted(a.keys() - b.keys()), "target_only": sorted(b.keys() - a.keys())}
        rows.append(row)
    report = {"source_repository": "https://github.com/cosmosjhj/I-Drift", "source_commit": commit,
              "target_repository": "https://github.com/myjju08/I-Drift",
              "target_base_commit": git(args.target, "rev-parse", "HEAD").decode().strip(),
              "target_content": "working-tree source bytes recorded by SHA256, including pending commit files",
              "self_reports_excluded": sorted(SELF_REPORTS),
              "counts": dict(Counter(row["status"] for row in rows)), "files": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
