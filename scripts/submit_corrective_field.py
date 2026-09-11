"""Snapshot committed source and submit the three matched srv02 experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight_corrective_field import VARIANTS, sha256, validate_suite


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def make_snapshot(root, destination, suite_id):
    root = Path(root).resolve(strict=True)
    if git("status", "--porcelain", "--untracked-files=all", root=root):
        raise ValueError("Commit all source/config changes before submitting the suite")
    commit = git("rev-parse", "HEAD", root=root)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit],
                       stdout=handle, check=True)
    source = destination / "source"
    source.mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
    validate_suite(source / "configs/corrective_field")
    files = {str(path.relative_to(source)): sha256(path)
             for path in sorted(source.rglob("*")) if path.is_file()}
    manifest = {
        "kind": "corrective_field_source_snapshot", "suite_id": suite_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": git("remote", "get-url", "origin", root=root),
        "commit": commit, "source_archive_sha256": sha256(archive),
        "files_sha256": files,
    }
    manifest_path = destination / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Runs import this shared immutable snapshot, including when Slurm queues them.
    for path in source.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    for path in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    source.chmod(0o555)
    archive.chmod(0o444)
    manifest_path.chmod(0o444)
    return manifest


def submission_command(snapshot, variant, run_root, python):
    source = snapshot / "source"
    return [
        "sbatch", "--parsable", f"--job-name=CF-{variant}",
        f"--chdir={source}", f"--output={snapshot / 'slurm-%x-%j.out'}",
        str(source / "scripts/slurm/run_corrective_field.sbatch"),
        str(snapshot), variant, str(run_root), str(python),
    ]


def validation_command(snapshot, python):
    source = snapshot / "source"
    return [
        "sbatch", "--parsable", "--job-name=CF-validate",
        f"--chdir={source}", f"--output={snapshot / 'slurm-%x-%j.out'}",
        str(source / "scripts/slurm/validate_corrective_field.sbatch"),
        str(snapshot), str(python),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Snapshot committed source, queue validation and three dependent jobs; otherwise only show the plan.")
    parser.add_argument("--snapshot-root", type=Path, default=ROOT / "runs/corrective-field-submissions")
    parser.add_argument("--run-root", type=Path, default=Path("/home/juhyeong/corrective-field-runs"))
    parser.add_argument("--python", type=Path, default=Path("/home/juhyeong/.venvs/replay-drift/bin/python"))
    args = parser.parse_args()
    validate_suite(ROOT / "configs/corrective_field")
    suite_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    snapshot = args.snapshot_root.resolve() / suite_id
    commands = {name: submission_command(snapshot, name, args.run_root, args.python) for name in VARIANTS}
    validate_command = validation_command(snapshot, args.python)
    print("Corrective Field: 3 fresh runs; each srv02 / 2 RTX3090 / 6 CPUs / 80 GB / 3 days.")
    if not args.submit:
        print(f"validation: {shlex.join(validate_command)}")
        for name, command in commands.items():
            print(f"{name}: {shlex.join([*command[:2], '--dependency=afterok:VALIDATION_JOB_ID', *command[2:]])}")
        print("Plan only. Commit the final source, then add --submit to snapshot and queue all three arms.")
        return
    manifest = make_snapshot(ROOT, snapshot, suite_id)
    submission = {
        "suite_id": suite_id, "commit": manifest["commit"],
        "snapshot": str(snapshot), "run_root": str(args.run_root),
        "validation": None, "jobs": {}, "status": "submitting",
    }
    submission_path = snapshot / "submission.json"

    def record():
        temporary = submission_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(submission, indent=2) + "\n")
        os.replace(temporary, submission_path)

    record()
    try:
        response = subprocess.check_output(validate_command, text=True).strip()
        validation_id = response.split(";", 1)[0]
        if not validation_id.isdigit():
            raise ValueError(f"Unexpected validation sbatch response: {response!r}")
        submission["validation"] = {"job_id": validation_id, "command": validate_command}
        record()
        print(f"Submitted validation: {validation_id}", flush=True)
        for name, command in commands.items():
            command = [*command[:2], f"--dependency=afterok:{validation_id}", *command[2:]]
            response = subprocess.check_output(command, text=True).strip()
            job_id = response.split(";", 1)[0]
            if not job_id.isdigit():
                raise ValueError(f"Unexpected sbatch response: {response!r}")
            submission["jobs"][name] = {"job_id": job_id, "command": command}
            record()
            print(f"Submitted {name}: {job_id}", flush=True)
    except Exception as exc:
        submission["status"] = "submission_failed"
        submission["error"] = str(exc)
        record()
        print(f"Submission stopped; inspect already-submitted job IDs in {submission_path}", file=sys.stderr)
        raise
    submission["status"] = "submitted"
    record()
    print(f"Commit: {manifest['commit']}\nSubmission record: {submission_path}")


if __name__ == "__main__":
    main()
