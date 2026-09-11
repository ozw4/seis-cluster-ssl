# F3 HMM_v2 screening execution

Run `run_all.sh` with Bash. It locates this checkout from its own location,
so invocation from another working directory is supported. No command starts
training by default. Task 07 prepares execution; it does not generate live
artifacts. The driver uses the Task 01–06 recipes and existing entrypoints.

## Environment and modes

For CLI checks or scheduled execution, export absolute
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT` and `F3_ROOT` pointing to the existing stores.
The driver sets `SEIS_SSL_CLUSTER_WORKSPACE` to its checkout; an already-set
value must identify that same checkout. For execution, also set
`CUDA_VISIBLE_DEVICES` to the selected GPU. Install the repository's existing
Python dependencies and use that environment's `python`. Bash, `flock`,
`mktemp`, and `tee` must be available. GPU commands run sequentially.

```bash
screening=experiments/f3/facies_benchmark_v2/127_hmm_v2_multi_head_screening_v1
bash "$screening/run_all.sh" --plan
bash "$screening/run_all.sh" --dry-run --from cluster-targets --to cluster-replay
```

- `--plan` is the default. It prints the command sequence without invoking
  Python, reading artifacts, creating logs, or taking a lock. Manifest digests
  are explicitly deferred, not represented as verified hashes.
- `--dry-run` calls existing CLIs with `--dry-run`. It creates no driver logs
  or locks. Prerequisite artifacts must already exist for the selected stages.
  This is not a virtual pipeline: dry-running an earlier stage does not create
  the inputs required by a later stage. The manifest CLI may use its existing
  temporary validation directory. Task 04's dry-run only plans the audit;
  it does not certify completed checkpoints.
- `--execute` performs live work. All source audits are live in this mode.
  No overwrite, quarantine, automatic repair, budget override, or target
  replacement flags are passed.

## Order and prerequisites

`--from` and `--to` select an inclusive range in the following order:

| Stage | Work and required inputs |
| --- | --- |
| `cluster-targets` | Generate K=4/8/10 from existing HMM_v1 MAE100 embeddings. |
| `cluster-replay` | Generate K=6 at the separate replay root. |
| `export` | Task 01 wrapper exports K=4,8,10,6 hard targets from clustering outputs. |
| `manifests` | Four Task 02 manifests; requires shared targets, historical K6 and exact replay. |
| `smoke` | Four one-step feasibility runs initialized from MAE100. |
| `full` | Four independent 25-epoch runs initialized from MAE100. |
| `audit` | Task 04 live audit of all four completed full checkpoints. |
| `embeddings` | Four extractions with `--device cuda --skip-existing`. |
| `downstream` | Four candidates × five layouts × three sizes = 60 cells. |
| `summary` | Task 06 complete paired screening against historical K6's 15 cells. |

Every smoke/full command computes the digest of that candidate's current
manifest immediately before invocation and supplies it only to that process.
A previously exported hash variable is not reused. Full runs never resume
from smoke. Embedding or downstream entry, including starting directly there,
always invokes the all-four Task 04 audit first. Summary independently audits
all four sources and the historical source through Task 06.

For a fresh, scheduled run after prerequisites and resource availability have
been checked:

```bash
bash "$screening/run_all.sh" --execute
```

`--candidate <full candidate ID>` selects one candidate for a single
`manifests`, `smoke`, `full`, `embeddings`, or `downstream` stage. It cannot
narrow audit or summary completeness. `--layout` and `--size` narrow a
single downstream stage. These filters are rejected for multi-stage ranges
to avoid presenting a partial candidate run as complete screening.

## Restart boundaries

The driver stops at the first failed command. It does not infer completion
from a file's existence, skip occupied cells, or automatically resume training.
Use the failed step recorded in the invocation log to select the remaining
work. Successful earlier stages are not replayed automatically.

| Existing state | Restart action |
| --- | --- |
| Clustering root exists | Driver refuses to rewrite it. Inspect the existing output before proceeding to the next stage. Clustering has no driver resume. |
| Export interrupted | The wrapper refuses existing target outputs. Use Task 01's existing export CLI for only the missing K values, preserving its documented parameters; then restart at `manifests`. Never re-export historical K6. |
| Manifest exists | `--only-missing` revalidates current arrays, inputs, and K6 parity. Invalid manifests fail closed. |
| Smoke/full checkpoint is incomplete | Select that single stage and candidate with `--resume`. It passes only that recipe's own `latest.pt`. |
| Smoke/full is complete | Do not resume it. Proceed to the next unfinished candidate or stage. Full completion is certified by Task 04 before extraction. |
| Decoder training is incomplete | Select exactly one candidate/layout/size and use `--resume`. |
| Decoder finished; prediction/evaluation unfinished | Select the cell without `--resume`. The generic runner validates and reuses its completed decoder. |
| Cell has completed metrics | Do not rerun that cell. Select unfinished cells. Task 06 audits all completed evidence before any summary is published. |
| Embeddings exist | `--skip-existing` uses the extractor's existing metadata checks. An incompatible/partial output is not authorized for replacement. |
| Paired summary exists | Use `--dry-run --from summary --to summary` to re-audit live inputs. Writing again is refused; this check does not certify the bytes of a previously written summary. |

Example: resume only the interrupted K=[4,6] full run, then later extract
all four candidates after every full run is complete:

```bash
candidate=mae100_hmm_v2_mh_k46_distill020
bash "$screening/run_all.sh" --dry-run --from full --to full \
  --candidate "$candidate" --resume
bash "$screening/run_all.sh" --execute --from full --to full \
  --candidate "$candidate" --resume
bash "$screening/run_all.sh" --execute --from embeddings --to embeddings
```

Example: resume a decoder in one cell. If its decoder is already complete,
omit `--resume` to continue prediction/evaluation:

```bash
bash "$screening/run_all.sh" --execute --from downstream --to downstream \
  --candidate "$candidate" --layout layout_002 --size medium --resume
```

A run interrupted before it produced a checkpoint is not resumed by deleting
outputs. Inspect the partial state and use the existing stage-specific recovery
procedure in a separately reviewed execution. The driver never deletes data.
Keep audited full checkpoints immutable throughout embedding and downstream.
Do not run manual stage commands concurrently with the driver.

## Logs and final evidence

Execution logs are Git-excluded artifacts under
`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/logs/`.
A nonblocking `driver.lock` prevents concurrent drivers using this store.
Each invocation gets a unique timestamped directory, append-only per-command
logs (command, stdout, stderr), and `events.log` with UTC start/completion and
final exit status/step. Earlier invocation logs are preserved. A failed command's
exit status propagates through `tee`; later commands are not launched.

`finished mode=plan` or `finished mode=dry-run` is not scientific completion.
Even `--execute` with a selected stage certifies only that command range's
successful exit. Screening evidence comes from Task 06's all-candidate audit
and its exact `comparison.csv`, `summary.json`, `summary.md` output set.
No driver status marker replaces those audits. HMM_v1 assets and
`reports/hmm_v1/` remain unchanged; Volve and Parihaka are outside this driver.
