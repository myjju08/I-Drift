# Drift ImageNet Runbook (Servers 2/3/4/5)

This is the historical multi-node cluster runbook. Its host paths, world size
and batch settings do not describe the DINO/MAE adversarial experiments ported
on 2026-09-10. For their two-rank configurations and local runtime commands,
use [the current README](README.md), [DINO guide](DINO_EXPERIMENTS.md), and
[MAE guide](experiments/mae/README.md).

Assumptions:
- 4 nodes total: server2, server3, server4, server5
- each node has 8 GPUs
- same repo path on all nodes: `/home/juhyeong/drift-model-imagenet`
- same data paths on all nodes (shared storage or mirrored):
  - ImageNet root: `/data/imagenet/ILSVRC2012`
  - latent cache: `/data/imagenet/latent_cache_256`

Default cluster presets in this runbook are conservative GPU-safe values:
- MAE: `batch_size=32` per GPU
- Generator: `batch_size=8`, `gen_per_label=32` per GPU

Node rank mapping:
- server2: `NODE_RANK=0` (master)
- server3: `NODE_RANK=1`
- server4: `NODE_RANK=2`
- server5: `NODE_RANK=3`

## 1) One-time setup per node

```bash
cd /home/juhyeong/drift-model-imagenet
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user diffusers transformers accelerate
wandb login
```

## 2) Prepare ImageNet folder layout (run once)

This also downloads the 2.5MB devkit file and organizes `val/` into class folders.

```bash
cd /home/juhyeong/drift-model-imagenet
export IMAGENET_PATH=/data/imagenet/ILSVRC2012
export RAW_DIR=/data/imagenet/raw
bash scripts/cluster/prepare_imagenet.sh
```

## 2.5) Optional: fetch the official MAE-640 checkpoint instead of training MAE

This downloads the official JAX artifact from `Goodeat/drifting` and converts it
to the PyTorch checkpoint format expected by `train_imagenet_gen.py`.

```bash
cd /home/juhyeong/drift-model-imagenet
python3 scripts/convert_mae_hf_jax_to_torch.py \
  --model-id mae_latent_640 \
  --output /data/imagenet/mae_latent_640/ckpt_latest.pt
```

## 3) Build latent cache (run once, usually on one node)

```bash
cd /home/juhyeong/drift-model-imagenet
export IMAGENET_PATH=/data/imagenet/ILSVRC2012
export IMAGENET_CACHE_PATH=/data/imagenet/latent_cache_256
bash scripts/cluster/run_latent_cache.sh
```

## 4) Start MAE training (run the matching command on each node)

Common env (all nodes):

```bash
cd /home/juhyeong/drift-model-imagenet
export MASTER_ADDR=server2
export MASTER_PORT=29500
export NNODES=4
export NPROC_PER_NODE=8
export IMAGENET_PATH=/data/imagenet/ILSVRC2012
export IMAGENET_CACHE_PATH=/data/imagenet/latent_cache_256
export WORKDIR=/data/runs/mae_latent_640_wandb
```

Server-specific:

```bash
# server2
export NODE_RANK=0
bash scripts/cluster/run_mae_node.sh
```

```bash
# server3
export NODE_RANK=1
bash scripts/cluster/run_mae_node.sh
```

```bash
# server4
export NODE_RANK=2
bash scripts/cluster/run_mae_node.sh
```

```bash
# server5
export NODE_RANK=3
bash scripts/cluster/run_mae_node.sh
```

MAE checkpoint (for next stage):
- `/data/runs/mae_latent_640_wandb/checkpoints/ckpt_latest.pt`

## 5) Start Generator training (run the matching command on each node)

Common env (all nodes):

```bash
cd /home/juhyeong/drift-model-imagenet
export MASTER_ADDR=server2
export MASTER_PORT=29500
export NNODES=4
export NPROC_PER_NODE=8
export IMAGENET_PATH=/data/imagenet/ILSVRC2012
export IMAGENET_CACHE_PATH=/data/imagenet/latent_cache_256
export MAE_CHECKPOINT=/data/runs/mae_latent_640_wandb/checkpoints/ckpt_latest.pt
export WORKDIR=/data/runs/gen_latent_B_wandb
```

Server-specific:

```bash
# server2
export NODE_RANK=0
bash scripts/cluster/run_gen_node.sh
```

```bash
# server3
export NODE_RANK=1
bash scripts/cluster/run_gen_node.sh
```

```bash
# server4
export NODE_RANK=2
bash scripts/cluster/run_gen_node.sh
```

```bash
# server5
export NODE_RANK=3
bash scripts/cluster/run_gen_node.sh
```

## 6) Metrics logging

- MAE and Generator metrics are logged to:
  - W&B (when `wandb login` done)
  - `train_log.jsonl` inside each `WORKDIR`
- Generator mid-training eval now logs:
  - `fid/cfg*`
  - `is/cfg*`
  - `is_std/cfg*`

## 7) Quick monitoring

```bash
tail -f /data/runs/mae_latent_640_wandb/train_log.jsonl
tail -f /data/runs/gen_latent_B_wandb/train_log.jsonl
```

If OOM happens:
- reduce `dataset.batch_size` and/or `train.gen_per_label` in `configs/gen/latent_sota_B_wandb.yaml`.
