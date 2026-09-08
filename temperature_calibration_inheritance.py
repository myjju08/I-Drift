"""Authenticate a weights-only DINO ablation with unchanged baseline taus.

This is provenance inheritance, not a calibration of the tuned encoder. The
trainer must still run every original data, geometry, feature and numeric tau
check. Only its calibrated-checkpoint identity check has this explicit alternate
path; ordinary calibrated runs retain their original strict identity check.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping


MANIFEST_KEY = "temperature_calibration_inheritance_manifest"
MANIFEST_SHA_KEY = MANIFEST_KEY + "_sha256"
_ALLOWED_DIFFERENCES = {"name", "feature_checkpoint", MANIFEST_KEY, MANIFEST_SHA_KEY}
_RAW_ALLOWED_DIFFERENCES = {
    "logging": {"name"},
    "feature": {"feature_checkpoint", MANIFEST_KEY, MANIFEST_SHA_KEY},
    "train": {MANIFEST_KEY, MANIFEST_SHA_KEY},
}
_TRAINABLE_NAMES = {
    f"layer{stage}.{block}.conv{conv}.weight"
    for stage, block in ((3, 5), (4, 2)) for conv in (1, 2, 3)
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("DINO temperature inheritance: " + message)


def _checked_file(entry: Mapping, label: str, *, check_size: bool = False) -> Path:
    _require(isinstance(entry, Mapping), f"missing {label} identity")
    path_value = entry.get("path")
    _require(isinstance(path_value, str) and bool(path_value), f"missing {label} path")
    path = Path(path_value).resolve()
    _require(path.is_file(), f"{label} file missing: {path}")
    expected = str(entry.get("sha256", "")).lower()
    _require(len(expected) == 64 and sha256_file(path) == expected, f"{label} SHA256 mismatch")
    if check_size:
        _require(entry.get("bytes") == path.stat().st_size, f"{label} byte count mismatch")
    return path


def _canonical_training_config(cfg: Mapping, *, io_cap_override: bool = False) -> str:
    """Include nested raw model settings so flat equality cannot hide a change."""
    value = copy.deepcopy(dict(cfg))
    for key in _ALLOWED_DIFFERENCES:
        value.pop(key, None)
    if io_cap_override:
        value.pop("raw_image_io_concurrency_per_rank", None)
    raw = value.get("_raw")
    if isinstance(raw, dict):
        for section, allowed_keys in _RAW_ALLOWED_DIFFERENCES.items():
            if isinstance(raw.get(section), dict):
                for key in allowed_keys:
                    raw[section].pop(key, None)
        if io_cap_override and isinstance(raw.get("dataset"), dict):
            raw["dataset"].pop("raw_image_io_concurrency_per_rank", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_dino_temperature_inheritance(
    cfg: Mapping, artifact: Mapping, checkpoint: str | Path,
) -> dict:
    """Validate the complete pinned identity chain, without changing ``cfg``.

    The sidecar contains schema_version=1 or 2, kind=weights_only_dino_ablation,
    numeric_temperature_policy=inherit_original_without_refit, the original
    calibration_artifact_sha256, selected_tuning_step, and four file identities:
    baseline_config, original_teacher, tuned_encoder, tuning_summary. Each file
    identity has path and sha256; both encoder identities also require bytes.
    baseline_config is the original trainer's flattened configuration JSON,
    including its _raw dictionary, before any training-derived fields are added.
    Version 2 additionally requires exactly runtime_overrides={
    "raw_image_io_concurrency_per_rank": 1}. This controls only the shared
    physical JPEG open/decode semaphore; workers, sampler, transforms, output
    order, batch geometry, bank representation and all learning settings remain
    subject to the original exact comparison. Version 1 has no runtime exemption.
    """
    manifest_path = _checked_file(
        {"path": cfg.get(MANIFEST_KEY), "sha256": cfg.get(MANIFEST_SHA_KEY)},
        "inheritance manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    _require(
        type(schema_version) is int and schema_version in {1, 2}
        and manifest.get("kind") == "weights_only_dino_ablation"
        and manifest.get("numeric_temperature_policy") == "inherit_original_without_refit",
        "invalid sidecar protocol; inherited taus must not be called recalibrated",
    )
    io_cap_override = schema_version == 2
    cap_key = "raw_image_io_concurrency_per_rank"
    if io_cap_override:
        overrides = manifest.get("runtime_overrides")
        _require(isinstance(overrides, dict) and set(overrides) == {cap_key}
                 and type(overrides[cap_key]) is int and overrides[cap_key] == 1,
                 "version 2 permits only the declared physical I/O cap of integer 1")
    else:
        _require("runtime_overrides" not in manifest,
                 "version 1 cannot declare runtime overrides")
    _require(
        cfg.get("temperature_calibration_status") == "ready"
        and cfg.get("temperature_calibration_role") == "dino"
        and cfg.get("feature_extractor") == "dino_resnet50"
        and cfg.get("require_raw_temperature_calibration") is True,
        "only production DINO with the normal calibration guard can inherit taus",
    )
    _require(
        str(cfg.get("adversarial_mode", "none")).lower() in {"", "none", "off"}
        and not cfg.get("feature_adapter", False)
        and not cfg.get("feature_gan", False),
        "weights-only ablation cannot add GAN losses or feature adapters",
    )
    artifact_sha = str(manifest.get("calibration_artifact_sha256", "")).lower()
    _require(
        artifact_sha == str(cfg.get("temperature_calibration_artifact_sha256", "")).lower(),
        "sidecar does not identify the original calibration artifact",
    )
    artifact_path = _checked_file(
        {"path": cfg.get("temperature_calibration_artifact"), "sha256": artifact_sha},
        "original calibration artifact",
    )
    _require(json.loads(artifact_path.read_text(encoding="utf-8")) == artifact,
             "supplied calibration artifact differs from the pinned original")
    baseline_path = _checked_file(manifest.get("baseline_config"), "baseline config")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    _require(isinstance(baseline, dict), "baseline config must be a flattened JSON object")
    _require(not baseline.get(MANIFEST_KEY) and not baseline.get(MANIFEST_SHA_KEY),
             "baseline itself must use original calibration identity")
    if io_cap_override:
        active_dataset = (cfg.get("_raw") or {}).get("dataset")
        baseline_dataset = (baseline.get("_raw") or {}).get("dataset")
        _require(isinstance(active_dataset, dict) and isinstance(baseline_dataset, dict),
                 "physical I/O override requires original and active raw dataset sections")
        _require(cfg.get("use_latent") is False and cfg.get("use_cache") is False,
                 "physical I/O override applies only to direct raw image reads")
        for description, config in (("active flattened", cfg), ("active dataset", active_dataset)):
            _require(type(config.get(cap_key)) is int and config[cap_key] == 1,
                     f"{description} physical I/O cap must match declared integer 1")
        for description, config in (("baseline flattened", baseline), ("baseline dataset", baseline_dataset)):
            _require(cap_key not in config or (type(config[cap_key]) is int and config[cap_key] == 0),
                     f"{description} physical I/O cap must be absent or integer 0")
    _require(
        _canonical_training_config(cfg, io_cap_override=io_cap_override)
        == _canonical_training_config(baseline, io_cap_override=io_cap_override),
        "training config differs from baseline beyond encoder weights, name, and provenance",
    )
    teacher_entry = manifest.get("original_teacher")
    teacher_path = _checked_file(teacher_entry, "original teacher", check_size=True)
    teacher_sha = str(teacher_entry["sha256"]).lower()
    provenance = (artifact.get("encoder_provenance") or {}).get("dino") or {}
    _require(
        provenance.get("feature_extractor") == "dino_resnet50"
        and provenance.get("checkpoint_bytes") == teacher_path.stat().st_size
        and str(provenance.get("checkpoint_sha256", "")).lower() == teacher_sha,
        "original teacher does not match original calibration provenance",
    )
    baseline_teacher = _checked_file(
        {"path": baseline.get("feature_checkpoint"), "sha256": teacher_sha},
        "baseline feature checkpoint",
    )
    _require(baseline_teacher.stat().st_size == teacher_path.stat().st_size,
             "baseline feature checkpoint byte count mismatch")
    tuned_entry = manifest.get("tuned_encoder")
    tuned_path = _checked_file(tuned_entry, "tuned encoder", check_size=True)
    tuned_sha = str(tuned_entry["sha256"]).lower()
    checkpoint = Path(checkpoint).resolve()
    _require(checkpoint == Path(str(cfg.get("feature_checkpoint", ""))).resolve()
             and checkpoint == tuned_path, "active encoder path is not the pinned tuned encoder")
    _require(not tuned_path.samefile(teacher_path) and tuned_sha != teacher_sha,
             "tuned encoder must be a separate changed checkpoint; preserve the teacher")
    summary_path = _checked_file(manifest.get("tuning_summary"), "tuning summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_step = manifest.get("selected_tuning_step")
    _require(isinstance(selected_step, int) and not isinstance(selected_step, bool)
             and selected_step > 0 and summary.get("best_step") == selected_step,
             "tuned checkpoint selection step mismatch")
    summary_metadata = summary.get("metadata") or {}
    _require(
        summary.get("export_sha256") == tuned_sha
        and summary_metadata.get("teacher_sha256") == teacher_sha
        and summary.get("original_teacher_unchanged") is True
        and summary.get("in_memory_teacher_unchanged") is True,
        "tuning summary does not authenticate the preserved teacher and tuned export",
    )
    audit = summary.get("last_student_freeze_audit") or {}
    _require(
        summary_metadata.get("trainable_blocks") == ["layer3.5", "layer4.2"]
        and set(audit.get("trainable_parameter_names", ())) == _TRAINABLE_NAMES
        and audit.get("all_frozen_tensors_unchanged") is True
        and audit.get("bn_parameters_and_buffers_unchanged") is True
        and audit.get("changed_frozen_keys") == [],
        "tuning summary lacks the expected frozen-weight and BN integrity audit",
    )
    return {
        "kind": manifest["kind"],
        "numeric_temperature_policy": manifest["numeric_temperature_policy"],
        "original_teacher_sha256": teacher_sha,
        "tuned_encoder_sha256": tuned_sha,
        "selected_tuning_step": selected_step,
        "calibration_artifact_sha256": artifact_sha,
        "inheritance_manifest_sha256": str(cfg[MANIFEST_SHA_KEY]).lower(),
        **({"runtime_overrides": {cap_key: 1}} if io_cap_override else {}),
    }
