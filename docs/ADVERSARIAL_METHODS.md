# Adversarial objectives and implementation

This describes the implementation imported on 2026-09-10. The preset is part
of the method: the older DINO EMA/structure experiments and the currently
running hard-copy experiments use different settings.

## Generator objectives

Let `L_base` be the existing reverse-drift objective in frozen DINO or MAE
features, `D_target` the discriminator snapshot fixed throughout a generator
step, and `L_Dfeatures` the existing drift objective evaluated in that
snapshot's intermediate features.

| `adversarial_mode` | Generator loss |
| --- | --- |
| `none` | `L_base` |
| `raw_gan` | `L_base + adversarial_loss_weight * mean(softplus(-D_target(G(z), y)))` |
| `feature_drift` | `L_base + adversarial_drift_weight * L_Dfeatures` |
| `mixed` | `L_base` plus both weighted terms above |

`feature_drift` does not use a discriminator logit in the generator loss.
The discriminator still learns to distinguish real from generated inputs.
`mixed` shares one target forward for generator logits and features. Positive,
negative and replay particles are detached and freshly encoded by the same
snapshot; generated queries retain their input gradients. The base encoder
parameters remain frozen while gradients pass through it to the generator.

The existing `feature_gan: true` implementation in `models/feature_gan.py` is
a separate experiment: projection heads on frozen encoder maps, hinge losses,
and a calibrated gradient ratio. It updates its head before the generator
backward. The current raw-input adversarial modes instead use
`models/adversarial_drift.py`, logistic losses, fixed loss weights, and the
update sequence below. Presets set `feature_gan: false` and
`feature_adapter: false`.

## Discriminator loss and update sequence

The discriminator minimizes

```text
mean(softplus(-D_online(real, y)))
+ mean(softplus(D_online(stop_gradient(fake), y)))
+ lazy R1
+ adversarial_structure_weight * real_structure_loss  [when enabled]
```

R1 uses the squared input-gradient norm on real inputs in float32, multiplied
by `gamma * interval / 2` on steps divisible by the interval (including step
zero). It is included in one Adam update, not a second optimizer update.
Current presets use gamma 1, interval 16, discriminator learning rate 1e-4,
Adam betas `(0, 0.99)`, and no discriminator weight decay.

Each step follows this order:

1. Compute base and auxiliary generator objectives with a fixed target D.
2. Backpropagate their sum, clip the generator gradient, and update G.
3. Update online D using detached real/fake particles from that same step.
4. Copy online D to target D when decay is 0, or apply the configured EMA.

D gradients are explicitly averaged across distributed ranks, including the
lazy second-order R1 path. Non-finite adversarial loss/gradient checks are
collective. Target D has no parameter gradients but preserves gradients with
respect to generated inputs. Discriminator initialization preserves generator
RNG state.

## DINO RGB discriminator

`ConditionalMultiScaleDiscriminator` operates on RGB in training space
`[-1, 1]`. A full-resolution 3×3 stem and average pooling precede four
convolution blocks of widths `d, 2d, 4d, 8d`. The last three stages have
unconditional linear plus class-projection heads; logits are averaged before
the logistic loss. There is no batch normalization, dropout, or spectral
normalization. D32 has 1,624,899 parameters; D128 is a width ablation.

Each learned stage is pooled to 4×4 tokens, normalized along channels, and
scaled by the square root of its channel count. Learned-coordinate drift
uses the existing drift normalization and temperatures `R_list`, with no
DINO calibration multiplier. The frozen DINO branch retains the supplied
stage3/stage4 calibration artifact and the `norm_x` weighting.

Optional real structure preservation compares cosine distances of distinct
same-class real image pairs at matching spatial positions. CNN stages 2/3
match DINO stage3 and CNN stage4 matches DINO stage4. Its stage, pair and
position averages have equal weight. Current hard-copy presets disable this
penalty (`adversarial_structure_weight: 0`); the older EMA presets enable it.
The trainer requests DINO teacher maps only when this penalty is active.

## MAE latent discriminator

The current MAE experiments use `experiments/mae/` and 4×32×32 generator
latents. The frozen MAE also consumes these latents. The VAE is used for
decoded samples and evaluation, not in either adversarial gradient path.

`latent_spatial_844` removes the RGB stem's average pooling and changes the
fourth block's first convolution to stride 1. Stages 2/3/4 therefore expose
native 8×8, 4×4 and 4×4 maps. The drift interface then pools each map to
4×4 (16 tokens at each scale), with channel RMS normalization. Adaptive
upsampling of a coarse 1×1 feature is not used in this current architecture.
R1 is with respect to real latents. DINO teacher
structure loss is disabled, and target D is copied after every update.

The spatial architecture has its own checkpoint input identity. The earlier
direct-latent 4×4/2×2/1×1 checkpoints must not be resumed into it even though
the parameter tensor shapes happen to match.

## Historical replay and current settings

The frozen historical generator snapshot, replay start epoch, count, ratio
and storage dtype remain controlled by the existing trainer. Replay targets
are detached. With `G` current and `H` historical samples, repulsive weights
are `1-rho` and `rho*G/H` respectively.

`adversarial_apply_replay` selects whether the learned discriminator feature
branch also receives history **and** the corresponding current/history
weights. False sends no history and no replay weights to that branch; it does
not disable replay in the frozen base encoder. Omitting this field preserves
the original behavior (true).

The added `adversarial/replay_enabled` metric reports that branch eligibility;
`adversarial/history_count` reports actual history use. A MAE preset with global
replay disabled can therefore log eligibility 1 and history count 0. These two
diagnostic keys are additions to the old MAE runtime's logs, not loss changes.

| Experiment | GAN weight | Learned drift weight | D target decay | Structure | Replay |
| --- | ---: | ---: | ---: | ---: | --- |
| Active DINO-only replay + CNN drift | — | 1 | 0 | 0 | DINO only, rho .35, H16 after epoch 10 |
| Earlier DINO replay + raw GAN D32/D128 | .1 | — | 0 | 0 | DINO, rho .35, H16 after epoch 10 |
| Older DINO feature drift | — | 1 | .99 | 1 | Off |
| Older DINO mixed D32/D128 | .1 | 1 | .99 | 1 | Off |
| Active MAE control | — | — | — | — | Off |
| Active MAE spatial raw GAN | .15 | — | 0 | 0 | Off |
| Active MAE spatial feature drift | — | .1 | 0 | 0 | Off |

All four active experiments use seed 43, S/4 with 64 spatial generator
tokens (DINO: 256px/patch32; MAE: 32 latent/patch4), 4 conditioning labels per
rank, 32 generated/64 positive/32 negative particles per label, and two ranks.
That is 256 generated particles per optimizer step. The 40-generated-epoch
schedule is 200,183 steps; the DINO epoch-10 snapshot threshold is 50,046.
MAE evaluation CFG is `[1, 2, 3]`. See the per-experiment guides for exact
configs, assets, launch commands, and source manifests.

## Checkpoints and interpretation

Adversarial checkpoints include online/target discriminator parameters,
optimizer, update count, mode and configuration fields. Resume validates
compatibility. Generator, generator EMA, optimizer and historical replay state
retain their existing handling. Numbered checkpoints and atomic latest hard
links retain the source implementation's behavior.

Matching code, losses, gradients, parameters and configuration does not imply
bitwise training trajectory resumption: the original trainer does not save
every RNG and raw memory-bank state. CPU parity tests validate the implemented
operations; they do not establish equal final FID or replace a full-duration
GPU reproduction. Periodic 1,024-sample evaluation is a proxy and should not
be described as a full 50K evaluation.
