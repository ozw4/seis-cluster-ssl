# F3 K=6/8/10 multi-head pretraining

This stage trains two otherwise identical ordered-prototype encoders using the
same F3 inputs, MAE initialization, and K=6/8/10 target manifest:

- `strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1` uses
  `consistency_weight: 0.0`.
- `strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1` uses
  `consistency_weight: 0.1`.

The current-code single-head control,
`strat_hmm_pretext_m1_current_k6_topblock1_distill_v1`, remains the primary
baseline. This stage does not run voxel-decoder evaluation or tune the
consistency weight.

Set the manifest digest after the K=6/8/10 target bundle has passed the
migration and current-control gates. `01_build_multi_head_targets.yaml` is a
runner input for `build_strat_hmm_multi_head_targets.py`; use its paths as the
corresponding command-line arguments.

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json \
  | awk '{print $1}')"
export EXP=experiments/f3/facies_benchmark_v1/94_strat_hmm_multi_head_k6810_v1

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/02_train_nocons_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/02_train_nocons_smoke.yaml" --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/03_train_cons010_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/03_train_cons010_smoke.yaml" --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/04_train_nocons_full.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/04_train_nocons_full.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/05_train_cons010_full.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/05_train_cons010_full.yaml"

python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/06_extract_nocons_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/06_extract_nocons_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/07_extract_cons010_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/07_extract_cons010_embeddings.yaml" --skip-existing
```

Smoke output roots are intentionally separate and must never be resumed by a
full run. A full run may resume only from its own `latest.pt`; the checkpoint
identity rejects cross-variant resumes. The four scientific differences between
the full configs are `loss.consistency_weight`, `identity.model_tag`,
`identity.scientific_identity.variant`, and `paths.output_root`.

## Execution status

The 2026-07-20 preflight is blocked before initialization: the required
multi-head target manifest is absent, its K=8/K=10 and K=6 replay sources are
unavailable, and the available historical K=6 target does not share the source
embedding valid-token mask. No F3 smoke, full run, checkpoint, or embedding was
created. The lightweight status records and handoff are in
`results/f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_v1/`; do not treat
either configured model tag as a completed model until those records are
replaced by PASS validation artifacts.
