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
flip policy and Barlow Twins objective.

The tracked feasibility run is a one-step CUDA capacity check, not a CPU smoke
test. It uses the full crop and encoder geometry, the full-run batch size of 4,
two views, and AMP. This deliberately exercises full self-attention over all
4096 unmasked tokens before a 100-epoch run is started. Its epoch log and
`history.json` report `step_time_seconds` and `peak_cuda_memory_mib`; the latter
is PyTorch peak allocated memory for the training invocation. A successful run
shows that one full forward/backward/optimizer step fits on the target GPU, but
does not establish steady-state throughput or convergence.

The lightweight synthetic CPU smoke remains the focused integration test. It
uses a 4 x 4 x 4 volume and encoder dimension 4 to check the data path,
backward pass, checkpoint contract, and epoch-boundary resume without pretending
to measure full-geometry feasibility.

Run from the repository root after preparing the Parihaka manifest and path
list:

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export PRETRAIN=experiments/parihaka/facies_benchmark_v1/20_pretrain/amp_barlow_twins_flipxy_l0005_v1

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN/02_full_100ep.yaml" --dry-run

# Portable synthetic CPU smoke (two one-step epochs with resume):
pytest -q \
  tests/seis_ssl_cluster/test_barlow_twins_training_contract.py::test_checkpoint_contract_round_trip_and_epoch_resume

# Run this on the intended training GPU before starting the full config:
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$PRETRAIN/01_gpu_feasibility_1step.yaml"
```

The feasibility and full output roots are separate from one another and from
the MAE experiment. A completed run writes `latest.pt`,
training-loss-selected `best.pt`, `history.json`, and `resolved_config.json`
under its configured output root. No embedding extraction or downstream
benchmark is run by this experiment definition.
