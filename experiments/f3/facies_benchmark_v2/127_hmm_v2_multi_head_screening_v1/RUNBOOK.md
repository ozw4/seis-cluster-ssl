# F3 HMM_v2 screening execution

[README.md](README.md) defines the experiment, fixed scientific conditions,
artifact locations and completion criteria. Run the commands below from the
repository root. The Bash driver also supports invocation from another
working directory because it locates the checkout from its own path.

## Environment and modes

Use the repository's Python environment and provide these variables:

| Variable | Value |
| --- | --- |
| `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` | Absolute path to the existing artifact store. |
| `F3_ROOT` | Absolute path to the existing F3 input root. |
| `SEIS_SSL_CLUSTER_WORKSPACE` | This checkout's absolute path; the driver sets it and rejects a conflicting value. |
| `CUDA_VISIBLE_DEVICES` | Selected GPU; required by the driver in execute mode. |

Bash, `flock`, `mktemp` and `tee` must be available. GPU commands run
sequentially. Historical MAE100 embeddings/checkpoint, immutable K6 targets,
and canonical benchmark inputs must already exist.

```bash
screening=experiments/f3/facies_benchmark_v2/127_hmm_v2_multi_head_screening_v1
export SEIS_SSL_CLUSTER_WORKSPACE="$(pwd)"
bash "$screening/run_all.sh" --plan
bash "$screening/run_all.sh" --dry-run --from cluster-targets --to cluster-replay
```

- `--plan` is the default: print commands without Python, artifacts, logs or
  locks. Manifest hashes are deferred until execution.
- `--dry-run` delegates to CLI checks. Selected stages require their existing
  prerequisite artifacts; earlier dry-runs do not create later inputs. The
  manifest builder may use a temporary validation directory. Completed-source
  audit dry-run only lists recipes; summary dry-run performs the full live
  audit. Neither creates driver logs or locks.
- `--execute` performs live work and source audits, with per-command logs.

## Frozen control receipt

The versioned receipt is mandatory for summary. To reproduce and compare it
against the authoritative freeze records without writing files:

```bash
python tools/build_f3_hmm_v1_control_receipt.py \
  --freeze-root reports/hmm_v1 \
  --output "$screening/60_summary/hmm_v1_condition_2b_receipt.json" --check
```

The receipt's one-time generation command is the same command without
`--check`; creation refuses an existing output. This is an explicit maintenance
step, outside the execution driver. It validates the frozen report hashes and
405-cell completeness, then derives the Condition 2b checkpoint hash and
canonical digest of its 15 F3 cells. Do not edit the hashes manually or
regenerate the receipt to make a changed live control pass.

Runtime summary reads the receipt and live metrics only. A receipt mismatch
requires restoring or locating the frozen control artifacts before retrying;
retraining K6 or rerunning a historical cell cannot redefine the control.

## Stage order

`--from` and `--to` select an inclusive range:

| Stage | Work and required inputs |
| --- | --- |
| `cluster-targets` | K=4/8/10 clustering from the existing HMM_v1 MAE100 embeddings. |
| `cluster-replay` | K=6 clustering at the separate replay root. |
| `export` | Hard-target export for K=4,8,10 and parity-only K6 replay. |
| `manifests` | Four manifests using shared targets, historical K6 and exact replay. |
| `smoke` | Four independent one-step feasibility runs from MAE100. |
| `full` | Four independent 25-epoch continuations from MAE100. |
| `audit` | Read-only audit of all four completed full checkpoints. |
| `embeddings` | Four extractions with `--device cuda --skip-existing`. |
| `downstream` | Four candidates × five layouts × three sizes = 60 cells. |
| `summary` | All-candidate paired screening against the 15 frozen K6 cells. |

Before each smoke/full command, the driver computes the selected manifest's
current hash and supplies it only to that process. Full runs initialize from
MAE100 independently of smoke. Entry to embeddings or downstream always audits
all four completed sources first, including when only one candidate is selected.
Summary independently audits all candidates, the historical source and freeze
parity.

For a fresh scheduled run with prerequisites and resources available:

```bash
bash "$screening/run_all.sh" --execute
```

`--candidate <candidate ID>` selects one candidate for a single `manifests`,
`smoke`, `full`, `embeddings` or `downstream` stage. `--layout` and `--size`
select cells within a downstream-only invocation. Audit and summary always
require all four candidates.

## Resume and recovery

The driver stops on the first failure. Select remaining work using the failed
step in its log; it does not automatically resume training or skip occupied
cells. Keep audited full checkpoints immutable during extraction and downstream.

| Existing state | Restart boundary |
| --- | --- |
| Clustering root exists | Inspect the output, then select the next unfinished stage. The driver refuses to rewrite it and offers no clustering resume. |
| Export interrupted | Use `proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py` for only missing K values, taking the exact roots and flags from `10_targets/03_export_pseudo_targets.sh`; restart at `manifests` after all exports exist. Historical K6 is never exported here. |
| Manifest exists | `--only-missing` revalidates arrays, alignment, hashes and K6 parity. Invalid manifests fail. |
| Smoke/full checkpoint is incomplete | Select one stage and candidate with `--resume`; only that recipe's own `latest.pt` is passed. |
| Smoke/full is complete | Proceed to the next unfinished candidate or stage without resuming the completed run. All full runs must pass audit before extraction. |
| Decoder training is incomplete | Select one candidate/layout/size with `--resume`. |
| Decoder complete; prediction/evaluation unfinished | Select that cell without `--resume`; the runner validates and reuses the completed decoder. |
| Cell has completed metrics | Select unfinished cells. The final summary audits every completed cell. |
| Embeddings exist | `--skip-existing` validates metadata. A partial or incompatible output is not replaced automatically. |
| Paired summary exists | Use summary `--check-only` to re-audit live inputs. Publication refuses an existing summary root; the check does not certify previously published summary bytes. |

Resume one interrupted full run:

```bash
candidate=mae100_hmm_v2_mh_k46_distill020
bash "$screening/run_all.sh" --dry-run --from full --to full \
  --candidate "$candidate" --resume
bash "$screening/run_all.sh" --execute --from full --to full \
  --candidate "$candidate" --resume
```

Resume an interrupted decoder, omitting `--resume` if training has already
completed and only prediction/evaluation remains:

```bash
bash "$screening/run_all.sh" --execute --from downstream --to downstream \
  --candidate "$candidate" --layout layout_002 --size medium --resume
```

If a run stopped before creating a checkpoint, inspect its partial state and
use the existing stage-specific recovery procedure. The driver never deletes
outputs. Do not run manual stage commands concurrently with the driver.

## Final audit and summary

After all full runs, embeddings and candidate cells are complete, run the
read-only source audit and complete paired evidence check, then publish:

```bash
python proc/seis_ssl_cluster/audit_strat_hmm_multi_head_sources.py \
  --config "$screening/30_pretraining/03_audit_completed_sources.yaml"
python proc/seis_ssl_cluster/summarize_f3_multi_head_screening.py \
  --config "$screening/60_summary/01_paired_screening.yaml" --check-only
python proc/seis_ssl_cluster/summarize_f3_multi_head_screening.py \
  --config "$screening/60_summary/01_paired_screening.yaml"
```

Source audit prints JSON without writing artifacts. Summary `--dry-run` is
an alias for `--check-only`; both require all live evidence, including frozen
control parity, and write nothing. Publication creates only `comparison.csv`,
`summary.json` and `summary.md` under the paired summary root in README.
A successful selected stage or driver status message is not a substitute for
these complete evidence checks.

## Logs

Execute-mode logs are under
`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/logs/`.
A nonblocking `driver.lock` prevents concurrent drivers using the same store.
Each invocation has a unique timestamped directory, per-command logs containing
command/stdout/stderr, and `events.log` with UTC start/completion and final exit
status/step. Failed command status propagates through `tee`; later commands do
not start. Earlier logs remain available for recovery.
