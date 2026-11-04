# F3 LocalBT v3: decoder seed 42001 repeat

Requested on 2026-10-21: **freeze the encoder and change only the downstream
decoder seed from 42000 to 42001**. Scope is LocalBT v3, mask-token LR=3e-5,
masked:visible=50:50, all five original layouts and small/medium/large.
The original encoder, embeddings and seed-42000 results remain unchanged.
The execution does not automatically select a seed from test scores. The
later explicit user designation below is recorded separately from that rule.

## Final result designation: 2026-10-21

After the seed-42000 and seed-42001 test results were both reported, the user
explicitly designated **seed 42001 as the final F3 / LocalBT / v3 / mask-token
LR=3e-5 / masked:visible=50:50 result**, for all three budgets and five layouts.
[selection.yaml](selection.yaml) binds the decision to the completed 15 cells,
summary, execution definition, fixed encoder/test mask and each selected
decoder checkpoint by SHA-256. Artifact paths are artifact-root-relative.

Future comparisons use these 15 cells for this exact condition. Preserve
seed 42000 as a historical result. MAE, random, other learning rates or branch
weights, HMMなし/v1/v2, Parihaka and Volve remain unchanged. Existing immutable
execution receipts and aggregates are not rewritten; the final-result
designation is an explicit override record, not a change to training defaults.

| Budget | Test mean IoU (%) | Test macro-F1 (%) |
|---|---:|---:|
| small | 38.55 ± 4.81 | 52.17 ± 5.52 |
| medium | 44.69 ± 3.59 | 58.29 ± 4.41 |
| large | 52.73 ± 1.45 | 66.80 ± 1.38 |

Values are five-layout means ± sample SD. Completed test summary SHA-256:
`4924af38fb491ebc4681d6e8186e30f362edb76fbc570976620ff54ec90bc929`.
This is a **post-report user designation**, not a pre-specified seed-selection
protocol or evidence that changing the method improves performance. When
compared with seed-42000 HMMなし/v1/v2 references, disclose the decoder-seed
difference. Both seed results remain available.

## Fixed scientific contract

- Encoder: asymmetric LocalBT3 followed by the existing 25-epoch masked-v3
  continuation. K=[6,8,10], distillation=0.2, consistency=0.1, trace-column
  mask=10%, encoder/head LR=1e-5, mask-token LR=3e-5, branch weights=50:50.
  This run performs **no pretraining or embedding extraction**.
- Source encoder SHA-256:
  `fff49d6f4a240167a84bb68e8bd88a0e98c7585dc971f0653b6a0eee643f4ddf`.
- Each decoder: seed=42001, 50 epochs, 440 steps/epoch, 22,000 steps,
  batch=1, LR=0.001, weight decay=0.0001, balanced class weights, AMP,
  gradient clipping=1, unchanged decoder and tile/context geometry.
- All original layout/budget supervision grids remain unchanged. Only decoder
  initialization and seed-dependent training tile sampling/order change.
- Best checkpoint: validation macro-F1, then validation mean IoU, then lower
  weighted CE, using the existing deterministic tie rule. The training/evaluation
  runner never uses test to select epochs or settings, or automatically ranks
  seeds. The later explicit final-result designation is documented above.
- Test is exactly the agreed **113,979,624-voxel** common mask used in
  [136](../136_lithology_common_test_v3_v1/README.md), SHA-256
  `2bac34f4ebf8612d120673ae3dbda1a8720624e27a1701491059abf1c60cfe1d`.
  All 15 train/validation masks are checked for zero overlap before training.
- This is same-survey transductive downstream-label holdout with full-survey
  unlabelled pretraining/context, not buffered spatial holdout. Two decoder
  seeds on one frozen encoder are not two encoder-pretraining seeds.

## Definition and output ownership

[execution.yaml](execution.yaml) pins the original candidate/canonical
definitions, completed baseline selection receipt, old test summary and fixed
test mask. Source checkpoint and embedding hashes are checked against that
receipt. Every new resolved decoder configuration is compared with the
original configuration; only seed and output location differ.

The canonical YAML resolver still supplies seed 42000. The isolated runner
explicitly creates seed-42001 jobs; it does not change the canonical constant,
existing experiment YAMLs, or old completion contracts.

- Training/inference/validation artifacts:
  `surveys/f3/facies_benchmark_v2/benchmark/local_bt_v3_masklr3em5_decoder_seed42001_v1/runs/`.
- Test artifacts:
  `surveys/f3/facies_benchmark_v2/evaluation/local_bt_v3_masklr3em5_decoder_seed42001_v1/`.
- Driver status/log/lock:
  `operations/f3/local_bt_v3_masklr3em5_decoder_seed42001_v1/`.

The test root owns the pre-training selection receipt, exactly two files per
test cell (`metrics.json`, `evidence.json`), and a six-file completed summary:
**comparison.csv, per_layout.csv, class_metrics.csv, confusion_matrices.csv,
summary.json, summary.md**. The summary reports five-layout arithmetic means
and sample SD (ddof=1), with all 15 individual values and all six-class
confusion matrices retained. Decoder seed is explicit in the condition ID,
selection receipt, decoder configurations and run status. Reports are never
pipeline inputs. The existing seed-42000 summary is not overwritten.

## Run and monitor

Set `SEIS_SSL_CLUSTER_WORKSPACE`, `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` and the
existing non-secret `F3_ROOT` data location. From the checkout:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python proc/seis_ssl_cluster/run_f3_decoder_seed_repeat.py \
  --config experiments/f3/facies_benchmark_v2/137_local_bt_v3_masklr3em5_decoder_seed42001_v1/execution.yaml \
  --dry-run

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u proc/seis_ssl_cluster/run_f3_decoder_seed_repeat.py \
  --config experiments/f3/facies_benchmark_v2/137_local_bt_v3_masklr3em5_decoder_seed42001_v1/execution.yaml \
  --execute
```

One cell runs at a time. GPU admission requires 20 GiB free and otherwise
waits without changing the experiment or stopping unrelated processes.
There is no CPU-training fallback. Re-invoking the same command resumes only
the new job's own identity-matched latest.pt, never a seed-42000 decoder.
Completed training/prediction/validation stages are audited and reused; test
cells are identity-checked rather than overwritten. Conflicting inputs or
experiment identities fail closed. Status reports current layout/size and
completed test cells; decoder history.csv provides epoch/step progress.

Validation before launch: all-15-cell read-only preflight, related regression
tests, Ruff lint/format and package-isolation check. Regression coverage
includes seed-only changes, original defaults, source/output protection,
own-seed resume, serial scope, exact summary file set, failure status, and
absence of a silent CPU fallback.

## Launch record: 2026-10-21

- Read-only preflight: 15 paired cells; run identity
  `c946f2fa891b2cbb0ecfe4e21ec688ceaa5c720eaf46ce47377b4767b277eb69`.
- Related regression validation: **151 passed**, Ruff lint/format and package
  isolation passed. After correcting the runtime device selector to the
  shared runner's supported `cuda` value, **75 affected tests passed** again.
- The first launch rejected `cuda:0` before training or writing a decoder
  checkpoint. Its `driver.log` is preserved. The corrected invocation uses
  `CUDA_VISIBLE_DEVICES=1` and `device=cuda`, with `driver.retry1.log` retained
  in the same operations namespace. No scientific setting changed.
- The corrected driver was started detached, serially on GPU 1, beginning
  with layout_000/small. This is a launch record, not a completion claim;
  `status.json`, per-cell `decoder/history.csv` and the eventual completed
  test summary are authoritative for current progress.
