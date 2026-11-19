# Parihaka HMM v2: zero/zero decoder seed repeat

Requested on 2026-11-17. Repeat every MAE, asymmetric Barlow Twins and Random
arm from [experiment 53](../../../hmm_v2/v2_dis0_con0_v1/README.md): both medium and large label
budgets, five fixed layouts, **30 new decoder cells**. Change decoder seed
42000 to 42001. The pretrained encoder and embeddings remain fixed, with
encoder seed 42, K=[6,8,10], feature distillation weight 0 and inter-head
consistency weight 0. This is a decoder-seed repeat, not an encoder-seed repeat.

[execution.yaml](execution.yaml) binds all three original downstream and
training definitions, encoder checkpoints, the layout definition and all
30 original metrics by SHA-256. The runner checks the complete benchmark
identity after normalizing only decoder seed and its initial parameter digest.
Supervision, split support, encoder features, geometry and optimization must
remain identical. The canonical decoder parser still requires seed 42000;
only this isolated repeat substitutes 42001 after loading the baseline.

Use the existing repository Python environment and set
`SEIS_SSL_CLUSTER_WORKSPACE`, `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `PYTHONPATH=src`
and `CUDA_VISIBLE_DEVICES` to the chosen free GPU. Actual machine paths and
launch details belong only in ignored operations records.

```bash
python proc/seis_ssl_cluster/run_parihaka_hmm_zero_seed_repeat.py \
  --config experiments/parihaka/facies_benchmark_v1/54_hmm_v2_dis0_con0_decoder_seed42001_v1/execution.yaml \
  --dry-run
python proc/seis_ssl_cluster/run_parihaka_hmm_zero_seed_repeat.py \
  --config experiments/parihaka/facies_benchmark_v1/54_hmm_v2_dis0_con0_decoder_seed42001_v1/execution.yaml \
  --execute --device cuda
```

Run serially on one GPU, requiring at least 20 GiB free before each admission.
Train each decoder for the original 50 epochs, batch size 1, float32,
learning rate 0.001, balanced class weights and all training tiles once per
epoch. Select the first maximum validation Channel IoU, then evaluate test.
All thirty cells are fixed before execution; retain improvements and declines.
Existing seed-42000 results remain separate and unchanged.

An exclusive experiment lock prevents two repeat writers. Restart using the
same command: completed matching cells are audited and reused; partial cells
resume only their own `latest.pt`. Errors stop later admissions and remain
recorded. The runner never stops unrelated GPU processes.

New runs and the final summary are under
`surveys/parihaka/facies_benchmark_v1/benchmark/hmm_v2_dis0_con0_decoder_seed42001_v1/`
within the artifact root. Operations, `status.json`, logs, launch identities and
exit evidence are under
`operations/parihaka/hmm_v2_dis0_con0_decoder_seed42001_v1/`.

The summary is published only after all thirty results pass the existing
50-epoch, checkpoint, history and validation-selection audit. Its exact owned
set is **`cells.csv`, `comparison.csv`, `provenance.json`**. `cells.csv` preserves
all five test metrics for both seeds and their signed differences.
`comparison.csv` reports six source/budget groups, each with five-layout means,
sample SD and paired differences. All checkpoints and metrics remain in
artifacts; reports never become pipeline inputs. The full experiment catalog
will discover each saved decoder result, including its seed, on the next
refresh. A running cell is not a completed result.
