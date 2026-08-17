# Parihaka amplitude Barlow Twins pretraining

This experiment is survey-specific, transductive self-supervised pretraining on
the full unlabeled Parihaka amplitude volume. It is Salt3DNet-inspired Barlow
Twins encoder pretraining, not a reproduction of the full Salt3DNet segmentation
architecture. It does not include Salt3DNet's DenseNet, SKB, reconstruction
decoder, or salt decoder.

Each preprocessed physical crop produces two views. Inline and crossline flips
are sampled independently for each view with probability 0.5; depth/time is
never flipped. The objective uses no explicit negatives and no EMA teacher. The
model shares the existing `AmplitudeMAE3D` patch projection and encoder with MAE
pretraining. Its projection head is used only by the Barlow Twins loss and is
discarded downstream. Checkpoints expose the encoder in the same bare model
state used by the frozen-encoder and end-to-end runners; the serialized decoder
exists for model compatibility and is not claimed as pretrained.

The full run inherits the MAE comparison's manifest and path list, 128 x 128 x
128 crop, normalization, clipping, AGC, finite checks, zero masking, patch and
encoder geometry, batch size, samples per epoch, 100 epochs, workers, AMP
policy, device, seed, gradient clipping, learning rate, and weight decay. It
replaces masking, reconstruction loss, and MAE visualization with the two-view
flip policy and Barlow Twins objective. The smoke run uses batch size 2, the
minimum supported by cross-correlation normalization, and four samples so its
two full batches produce exactly two optimizer steps.

Run from the repository root after preparing the Parihaka manifest and path
list:

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export PRETRAIN=experiments/parihaka/facies_benchmark_v1/20_pretrain/amp_barlow_twins_flipxy_l0005_v1

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN/01_smoke_2step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN/02_full_100ep.yaml" --dry-run

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN/01_smoke_2step.yaml"
```

The smoke and full output roots are separate from one another and from the MAE
experiment. A completed run writes `latest.pt`, training-loss-selected
`best.pt`, `history.json`, and `resolved_config.json` under its configured
output root. No embedding extraction or downstream benchmark is run by this
experiment definition.
