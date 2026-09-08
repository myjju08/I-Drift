"""Run the existing offline evaluator with a verified NPZ read-back.

The original evaluator is left unchanged. All reported sample statistics are
recomputed from the closed uint8 image archive, and features are retained for
independent metric recalculation without repeating image generation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import eval_official_imagenet256 as evaluator


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = evaluator.parse_args()
    out = Path(args.out)
    feature_path = out.with_suffix(".features.npz")
    archive = Path(args.sample_npz) if args.sample_npz else out.with_suffix(".samples.npz")
    if any(path.exists() for path in (out, feature_path, archive)):
        raise FileExistsError("Refusing to overwrite an existing result, feature cache, or archive")
    provenance = {
        "seed": args.seed,
        "generation_batch_size": args.batch_size,
        "metrics_batch_size": args.metrics_batch_size,
        "pr_nhood": args.pr_nhood,
        "evaluator": "torch-fidelity inception-v3-compat; existing repository offline protocol",
        "torch_version": torch.__version__,
        "checkpoint_sha256": sha256_file(args.ckpt),
        "config_sha256": sha256_file(args.config),
        "evaluator_sha256": sha256_file(evaluator.__file__),
        "fid_reference_sha256": sha256_file(args.fid_ref_npz or args.eval_ref_npz),
        "pr_reference_sha256": sha256_file(args.pr_ref_npz or args.eval_ref_npz),
    }
    original_sample = evaluator._sample_and_extract

    @torch.no_grad()
    def sample_then_read(**kwargs):
        online_pool, online_logits, _ = original_sample(**kwargs)
        pool_parts, logits_parts = [], []
        moments = evaluator.MomentAccumulator(2048)
        pixel_hash = hashlib.sha256()
        max_pool_diff, max_logits_diff, count = 0.0, 0.0, 0
        print("[roundtrip] Reading the completed NPZ and re-extracting all sample features", flush=True)
        with evaluator.NpzArrayReader(str(kwargs["sample_npz_path"])) as reader:
            if reader.shape != (args.n_samples, 256, 256, 3) or reader.dtype != np.uint8:
                raise ValueError(f"Unexpected NPZ array: {reader.shape}, {reader.dtype}")
            for pixels in tqdm(reader.iter_batches(args.batch_size), desc="npz-readback", unit="batch"):
                pixel_hash.update(pixels.tobytes())
                images = torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous().to(kwargs["device"])
                pool_t, logits_t = kwargs["feature_model"](images)
                pool, logits = pool_t.cpu().numpy(), logits_t.cpu().numpy()
                end = count + len(pixels)
                np.testing.assert_allclose(pool, online_pool[count:end], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(logits, online_logits[count:end], rtol=1e-5, atol=1e-5)
                max_pool_diff = max(max_pool_diff, float(np.max(np.abs(pool - online_pool[count:end]))))
                max_logits_diff = max(max_logits_diff, float(np.max(np.abs(logits - online_logits[count:end]))))
                pool_parts.append(pool.copy())
                logits_parts.append(logits.copy())
                moments.update(pool)
                count = end
                del images, pool_t, logits_t
        if count != args.n_samples:
            raise ValueError(f"Read only {count} of {args.n_samples} requested samples")
        pool, logits = np.concatenate(pool_parts), np.concatenate(logits_parts)
        mu, sigma = moments.mean_cov()
        np.savez(feature_path, pool3=pool, logits=logits, mu=mu, sigma=sigma, labels=kwargs["labels_np"])
        provenance.update({
            "metrics_source": "features re-extracted from completed sample NPZ",
            "npz_readback_samples": count,
            "npz_pixels_sha256": pixel_hash.hexdigest(),
            "npz_readback_pool3_max_abs_difference": max_pool_diff,
            "npz_readback_logits_max_abs_difference": max_logits_diff,
            "sample_features_npz": str(feature_path),
        })
        print(f"[roundtrip] Verified {count} samples; max differences pool3={max_pool_diff}, logits={max_logits_diff}", flush=True)
        return pool, logits, (mu, sigma)

    evaluator._sample_and_extract = sample_then_read
    evaluator.main()
    with out.open() as handle:
        result = json.load(handle)
    result.update(provenance)
    with out.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
