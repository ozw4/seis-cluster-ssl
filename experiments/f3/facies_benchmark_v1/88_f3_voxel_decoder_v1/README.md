# F3 original-split V1 frozen-embedding decoder

The complete M3-V workflow, identity/failure policy, storage estimate, and
release checks are in
[`docs/f3_voxel_lithology_benchmark.md`](../../../../docs/f3_voxel_lithology_benchmark.md).

V1 trains the same lightweight decoder independently on the precomputed,
frozen `overlap_x16` embeddings from MAE, M1, and M2-A. It is not encoder
fine-tuning and has no raw-amplitude, coordinate, augmentation, boundary-loss,
or early-stopping input. The full jobs share the same voxel supervision,
architecture, tiles, balanced class weights, AdamW settings, 50 epochs, and
seed 42. M2-A's primary comparison is M1.

The exact decoder spec is
`frozen_embedding_decoder_nearest_voxel_ln_v1`: nearest-neighbor upsampling
followed by voxelwise LayerNorm. Trilinear upsampling plus GroupNorm is not the
benchmark implementation. Voxelwise LayerNorm avoids dependence on each
tile's spatial statistics, preserving whole-grid/tiled consistency. The
directory name `88_f3_voxel_decoder_v1` is a milestone number, not the decoder
artifact identity.

For each model, run the smoke config on CPU for two steps before the full job:

```bash
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/01_train_mae_smoke.yaml --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/01_train_mae_smoke.yaml --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/02_train_mae_full.yaml --dry-run
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/02_train_mae_full.yaml --device auto
python proc/seis_ssl_cluster/predict_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/03_predict_mae_voxels.yaml --dry-run
python proc/seis_ssl_cluster/predict_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/03_predict_mae_voxels.yaml --device auto
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/04_evaluate_mae_voxels.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/05_report_mae_voxels.yaml
```

Repeat in numeric order for M1 (`06`–`10`) and M2-A (`11`–`15`). Add
`--dry-run` to evaluation or report commands to validate without writing. Full
training consumes embeddings, not an MAE `mae_latest.pt` or pretext `best.pt`;
decoder inference alone consumes the decoder run's `best.pt`.

Each model root contains
`voxel_decoders/frozen_embedding_decoder_nearest_voxel_ln_v1`,
`voxel_predictions/frozen_embedding_decoder_nearest_voxel_ln_v1`,
`voxel_evaluations/frozen_embedding_decoder_nearest_voxel_ln_v1`, and
`voxel_reports/frozen_embedding_decoder_nearest_voxel_ln_v1`. Smoke checkpoints
use the distinct `frozen_embedding_decoder_nearest_voxel_ln_v1_smoke` directory
and cannot collide with full scientific outputs.

The old `frozen_embedding_decoder_v1` checkpoint/artifact name is not eligible
for resume or evaluation because it does not identify upsampling and
normalization. Do not reuse an old artifact merely by pointing a current config
at it. Regenerate from the smoke stage under the new spec and use checkpoint
schema 5 or later; the paths shown here are recommended examples, not a
validator-enforced hierarchy.

Training writes `latest.pt` for exact continuation and `best.pt` for inference.
Resume an interrupted full run with:

```bash
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config <FULL_CONFIG> --device auto --resume <MODEL_ROOT>/voxel_decoders/frozen_embedding_decoder_nearest_voxel_ln_v1/latest.pt
```

Do not resume from `best.pt`. Prediction/evaluation/report stages refuse existing
outputs by default; retain complete stages and restart at the first missing one.
