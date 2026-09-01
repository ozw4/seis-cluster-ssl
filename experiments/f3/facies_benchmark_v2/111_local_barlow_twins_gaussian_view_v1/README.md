# F3 Local Barlow Twins Gaussian-view search v1

This experiment tests one narrow explanation for the weak F3 Local Barlow
Twins result: the current two views contain no view-specific amplitude change.
The dataset uses `require_distinct=True`, so every pair differs by at least one
X/Y flip. The inverse token-index mapping nevertheless exposes exactly the same
physical amplitudes to the loss through a lossless spatial symmetry. That can
make matching the same physical token too easy for seismic amplitudes.

Two selectable candidates retain the flip and add independent zero-mean
Gaussian noise after the flip, only on valid voxels, at
`gaussian_noise_std: 0.05` and `0.10`. A third selectable candidate fixes both
views in the same orientation and adds independent valid-voxel Gaussian noise
at `0.10`. It directly tests whether forced X/Y invariance, rather than noise
strength alone, removes useful horizontal spatial information. Fixed positions
can themselves become a shortcut, so the identity arm is a narrow geometry
ablation rather than a generally preferred augmentation. A matched-duration
legacy policy keeps the original flip-only construction. It cannot influence
the three-way view lock, but it is both an attribution control and a
duration-only Barlow Twins arm.

The preserved actual-data diagnostic predates the identity arm and covers only
legacy forced flips plus the two forced-flip Gaussian policies. It used the
exact v1 preprocessing and 16 deterministic
epoch-0 crops (seed 42). After inverse-aligning each flip, it measured all
valid physical voxels and confirmed that masks, crop coordinates, and the 128
sampled physical-token pairs were unchanged:

| View policy | Aligned A/B correlation | A/B RMS difference | Per-view RMS vs base |
|---|---:|---:|---:|
| Legacy forced-distinct flips | 1.000000 | 0 | 0 |
| Gaussian noise 0.05 | 0.997322 | 0.070698 | 0.049994 |
| Gaussian noise 0.10 | 0.989375 | 0.141396 | 0.099988 |

The diagnostic covered 33,046,528 valid voxels and exercised 507,904 invalid
voxels without altering any invalid value. It therefore isolates controlled
view difficulty rather than a crop, mask, or correspondence change for those
forced-flip arms. It is not evidence that the new identity policy avoids a
fixed-position shortcut; that mechanism needs its own actual-data check.

The preserved execution output is
`artifacts/seis_ssl_cluster/diagnostics/f3/local_barlow_twins_gaussian_view_v1/view_diagnostic.json`
(SHA-256
`b4f89206bc49ae1fbf14cd62adccee396d6829adea6ff88243c0acf1f5b88d47`).

Reproduce it from the repository root on CPU. The command prints deterministic
JSON and writes nothing unless `--output` is supplied:

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
export EXP=experiments/f3/facies_benchmark_v2/111_local_barlow_twins_gaussian_view_v1

python "$EXP/diagnose_views.py"
```

The defaults are epoch 0 and indices 0 through 15. `--epoch`,
`--start-index`, and `--count` select another deterministic sample. To retain
the JSON, pass an explicit new path such as
`--output artifacts/seis_ssl_cluster/diagnostics/f3_local_bt_views.json`;
existing files are not overwritten.

## Fixed contract

The control is
`experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1/02_full_100ep.yaml`.
The three selectable screening configs copy its manifests, data preprocessing,
zero-mask contract, model, embedding dimension, Local Barlow Twins loss, batch
size, optimizer settings, learning rate, weight decay, sampling budget per
epoch, worker settings, seed, and gradient clipping. Only the augmentation
mapping, output root, and screening duration differ. The forced-flip mappings
are exactly `{policy: horizontal_flip_gaussian_noise_v1,
horizontal_flip_probability: 0.5, gaussian_noise_std: 0.05|0.10}`. The
same-orientation mapping is exactly `{policy: identity_gaussian_noise_v1,
gaussian_noise_std: 0.10}`. Each currently defined screen is a true 25-epoch
base-pretraining run with no `max_steps`.

The `legacy_flip_25ep` producer is a literal matched-duration copy of that
canonical control: only `paths.output_root` and `train.epochs` differ. Its
augmentation mapping remains exactly
`{horizontal_flip_probability: 0.5}`, and 25 epochs at the unchanged 10,000
samples per epoch and batch size 16 produce exactly 15,625 optimizer steps.

Fair downstream comparison requires the same encoder lineage as the published
v3 Local BT source. That source is not the 100-epoch base checkpoint: it is the
checkpoint after a separate 25-epoch top-1 continuation initialized from that
base. Every arm in this search therefore has two explicit phases:

1. a tunable-duration base run (25 epochs in the currently defined screen);
2. a fixed 25-epoch continuation initialized from that arm's exact base
   `latest.pt`, with `unfreeze_top_blocks: 1`.

The continuation copies the canonical stage-2 contract exactly: batch size 16,
10,000 samples per epoch, AdamW, learning rate `1e-5`, weight decay `0.05`, seed
42, FP32, gradient clipping 1.0, and 15,625 fresh continuation steps. It uses
the same view mapping as its base arm. Only that augmentation mapping, the
output root, and `continuation.init_checkpoint` may differ from the canonical
top-1 config; for legacy, even the augmentation mapping is canonical. This is
a fresh continuation phase, not `--resume`: its final metadata is epoch 25 and
global step 15,625 even though its lineage contains both phases. Every final
checkpoint must also record `continuation_lineage` schema 1 with the absolute
base path, the SHA-256 read before initialization, and `resume_count: 0`. The
pre-extraction audit requires that recorded SHA to equal the live base SHA.

Training deliberately reuses the v1 manifest and path list byte-for-byte so
the pretraining dataset and split do not drift. Extraction alone uses the v2
prepared amplitude manifest and the canonical `window_size: [128, 128, 128]`,
`overlap: [64, 64, 64]` (`overlap_x64`) contract used by the v3 downstream
benchmark.

Extraction consumes only each arm's final continuation checkpoint. The
experiment-local validation runner loads the section layouts and downstream
settings directly from
`../110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml`; it does not change
the dataset, splits, layout seeds, decoder seed, fine-tuning configuration,
evaluation procedure, or metrics. It independently audits base parity and
final-continuation parity because the canonical five-way source audit owns the
published comparison and requires the canonical 100-epoch base.

## Preregistered 25-epoch definitions

This table defines the immutable arms in the current protocol; it is not an
outcome ledger and is never updated with observed values. Checkpoint SHAs,
validation metrics, paired deltas, failures, and decisions belong only in the
exclusive artifact locks/results and the authoritative tracked report under
`reports/`.

| Candidate | View policy | Noise std | Base epochs | Fixed continuation | Protocol role |
|---|---|---:|---:|---:|---|
| `local_barlow_twins_gaussian_noise_std005` | `horizontal_flip_gaussian_noise_v1` | 0.05 | 25 | 25 epochs, top block 1 | selectable view |
| `local_barlow_twins_gaussian_noise_std010` | `horizontal_flip_gaussian_noise_v1` | 0.10 | 25 | 25 epochs, top block 1 | selectable view and forced-flip geometry arm |
| `local_barlow_twins_identity_gaussian_noise_std010` | `identity_gaussian_noise_v1` | 0.10 | 25 | 25 epochs, top block 1 | selectable view and identity geometry arm |
| `local_barlow_twins_legacy_flip_25ep` | legacy forced-distinct horizontal flips | — | 25 | 25 epochs, top block 1 | non-selectable attribution/duration control |

The legacy row is mandatory at 25 base epochs plus the fixed continuation.
Never rank it with the three selectable candidates or use it to change the
locked view mapping. Always
record the fixed-strength identity-0.10 minus forced-flip-0.10 contrast. A
reached 5- or 1-epoch branch gets its own preregistered definition document,
runner/config, artifact locks, and report directory; it never appends observed
outcomes to this protocol table.

The random comparison is not retrained or rerun. Reuse the existing per-job
validation metrics at
`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/f3_lithology_benchmark/mae_local_bt_five_way_v3/runs/model=random/layout=<layout>/size=<size>/evaluation/metrics.json`.
All candidate outcomes are recorded only after their validation jobs are
complete and audited under the same v3 conditions.

The existing audited validation reference is:

| Size | Random mean macro-F1 | Existing Local BT mean macro-F1 | Local BT minus random | Local BT layout wins |
|---|---:|---:|---:|---:|
| small | 0.466787 | 0.405799 | -0.060987 | 0/5 |
| medium | 0.512144 | 0.460533 | -0.051611 | 0/5 |
| large | 0.578663 | 0.532079 | -0.046585 | 0/5 |

The unchanged canonical source audit and 75-job result inspection both pass.
The exact consumed canonical v3 config at
`experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml`
has SHA-256
`285b0233ff82fe83808f82e929b611f570a67f01fa983ef191dda23d1858061b`.
The reused random checkpoint SHA-256 is
`6548d52446e7d6b9b57acd2bd39a8389a76bc5df55b52a9eda0472eb182a438c`;
the pinned canonical stage-1 Local BT reference at
`pretraining/f3/facies_benchmark_v1/local_barlow_twins_v1/full_100ep/latest.pt`
has SHA-256
`84550ed658166e8e6a40cd664e2e9ffbeab0c12d6917006abb417cd25e228ac0`.
The pinned canonical final Local BT reference at
`pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/stage2/local_bt100/local_bt_continue/full_25ep/latest.pt`
has SHA-256
`1c5312244f290dbfdcf2688ffa9fa8b5c64452ade162d5335be1bb8a0e256291`.
The audited canonical `summary/comparison.csv` used for final paired baselines
has SHA-256
`b135122a7db2b6b359817096ac546f99d4e4fac1ee003a99ce7289c0445cf913`.
The v1 pretraining manifest and path list are also pinned, respectively, to
`c5dbc3a66a5c2eed0ec5df8745f8bf5a461b1e2e66156700091f1a751bdc0ef5`
and
`b52fd5e0c57edb2d2158be12b94046b554b5e6e13ba17008321bcdbe0ae2acb1`.
The runner derives the random checkpoint and comparison paths from the
canonical v3 config and the pretraining input paths from the canonical artifact
root/reference checkpoint; these five live protocol inputs must match their
configured
digests before any CLI mode proceeds.

## Validation-only selection rule

Use only `macro_f1` on unique validation voxels. The pre-registered base
duration order is **25, then 5, then 1 epoch**. Every base duration is followed
by the same fixed 25-epoch continuation. This shorter-duration contingency is
motivated by the canonical F3 stage-1 training history: at epochs 1, 5, 25,
and 100, training loss is 2.6794, 0.3947, 0.3703, and 0.3585, while mean
cross-correlation diagonal is 0.9844, 0.9965, 0.9974, and 0.9974. Most of the
objective change therefore occurs by epoch 5, whereas the existing full
lineage loses all five paired layouts to random at every validation size. The
contingency tests early stopping; it does not add a longer-duration branch.

Apply this rule without looking at any test result:

1. At 25 base epochs plus fixed continuation, first run the same five `medium`
   v3 layouts for exactly the
   three selectable view candidates. Lock the view using only their unrounded
   mean medium macro-F1. Exact ties use this fixed priority: forced-flip 0.05,
   forced-flip 0.10, then identity 0.10. The lock reads exactly 15 metrics and
   their 15 job-specific candidate audits, records every input path/SHA, and is
   created exclusively without overwrite. Only after it exists may legacy
   medium validation run. The legacy result cannot change this choice, and
   discarded candidates are never revisited except that forced-flip 0.10 is a
   required same-strength geometry control when identity 0.10 is locked.
2. At every reached base duration, the policy pair is the locked view mapping
   and matched-duration legacy flips. A policy clears the medium random gate
   only with a positive paired delta in all five layouts (`5/5`) versus the
   existing random baseline. Gaussian attribution additionally requires a
   positive Gaussian-minus-legacy delta in all five layouts. Record all three
   causal contrasts independently: Gaussian-minus-random,
   legacy-minus-random, and Gaussian-minus-legacy.
3. Small/large execution remains closed until all five locked-view and all five
   legacy medium cells exist, their job identities validate, and either arm
   clears the strict `5/5` random gate. If neither does, record the attempt and
   proceed to the next base duration. If either does, run the locked view and
   legacy
   on the same five `small` and five `large` layouts. When identity 0.10 is
   locked, also run forced-flip 0.10 on those ten cells and record the
   identity-minus-flip geometry contrast across all 15 cells. The geometry
   control cannot open the medium gate. Small, medium, and large are all
   validation tuning signals; none is an untouched holdout or post-selection
   confirmation.
4. A policy is a passing final validation arm only when it has a positive
   paired delta over random in all 15 layout/size cells (`15/15`). Gaussian
   attribution is separately positive only when Gaussian also wins all 15
   cells over matched legacy. The first base duration in the fixed 25 -> 5 -> 1
   order with a passing final arm wins. If both arms pass, choose Gaussian only
   when its attribution gate also passes; otherwise choose the simpler legacy
   duration arm. If Gaussian alone passes random but fails attribution, it is
   still the winning Barlow Twins arm, but report that the incremental Gaussian
   effect was not consistent rather than claiming that noise caused the win.
5. If every medium-eligible arm at a base duration fails either the small or
   large random gate, preserve all results and proceed to the next base
   duration. If no arm passes at 1 base epoch plus fixed continuation, stop and
   report that this pre-registered view/duration family did not meet the
   success criterion.

The 5- and 1-base-epoch producer, fixed-continuation, extraction, and validation
configs are added only after their branch is reached, never speculatively.
Carry only the locked view mapping plus matched legacy; do not retune noise
strength or carry a discarded geometry control into those durations. Give
every policy-duration arm a distinct, immutable model ID, base output root,
continuation output root, and embedding root. Give every base duration a
distinct validation run root; no later arm or duration may reuse or overwrite
a prior directory.

Every base duration is a fresh seed-42 run from initialization. Its fixed
continuation must initialize only from that arm's completed base `latest.pt`;
never pass the base through `--resume`, and never reuse a base or continuation
checkpoint belonging to another duration. Base training uses a
persistent-worker loader; reconstructing it on resume consumes a new worker
base seed and shifts the later shuffle stream, so a resumed trajectory is not
the same fixed-duration condition as a fresh run.

Current-branch caveat: the four 25-epoch bases were operator-observed fresh
launches after exact output-root absence checks and without `--resume`. The
legacy base checkpoint schema has no resume counter, however, so that freshness
is external execution evidence rather than checkpoint-authenticated evidence.
Any reached 5- or 1-epoch branch must require and audit a base resume counter
when its distinct runner and configs are implemented.

Every result used in every branch is a validation result. Test data and
test metrics must remain untouched during this search. If a separate test
evaluation is later defined, run it once only after all view and duration
choices are locked.

## Run the screens

From the repository root:

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export workspace first}"
: "${F3_ROOT:?export F3 root first}"
export EXP=experiments/f3/facies_benchmark_v2/111_local_barlow_twins_gaussian_view_v1

pytest -q tests/seis_ssl_cluster/test_f3_local_barlow_twins_gaussian_view_configs.py

for config in \
  "$EXP/10_stage1/gaussian_noise_std005/01_screen_25ep.yaml" \
  "$EXP/10_stage1/gaussian_noise_std010/01_screen_25ep.yaml" \
  "$EXP/10_stage1/identity_gaussian_noise_std010/01_screen_25ep.yaml" \
  "$EXP/10_stage1/legacy_flip_25ep/01_matched_25ep.yaml"
do
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config" --dry-run
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config"
done

for candidate in \
  local_barlow_twins_gaussian_noise_std005 \
  local_barlow_twins_gaussian_noise_std010 \
  local_barlow_twins_identity_gaussian_noise_std010 \
  local_barlow_twins_legacy_flip_25ep
do
  python "$EXP/run_validation.py" \
    --config "$EXP/30_validation/01_candidates.yaml" \
    --candidate "$candidate" --audit-base-checkpoint-only
done

python "$EXP/run_validation.py" \
  --config "$EXP/30_validation/01_candidates.yaml" \
  --create-protocol-lock

for config in \
  "$EXP/15_stage2/gaussian_noise_std005/01_continue_25ep.yaml" \
  "$EXP/15_stage2/gaussian_noise_std010/01_continue_25ep.yaml" \
  "$EXP/15_stage2/identity_gaussian_noise_std010/01_continue_25ep.yaml" \
  "$EXP/15_stage2/legacy_flip_25ep/01_continue_25ep.yaml"
do
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config" --dry-run
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config"
done
```

The base-only audit reads no continuation or embedding. It records the live
base SHA and verifies exact epoch/global-step completion, augmentation,
objective, output ownership, and parity to the pinned canonical base. Finalize
all implementation, configs, tests, and this runbook before
`--create-protocol-lock`. That exclusive command re-audits and embeds all four
base payloads/live SHAs, the complete benchmark provenance (including the
canonical v3 YAML path/SHA), and the relevant repository file inventory. It
refuses to run if any continuation-output, candidate-embedding, validation,
selection-lock, or final-result evidence already exists.

The resulting `gaussian25_protocol_lock.json` is the immutable boundary before
stage 2. Every later checkpoint audit, dry run, validation job, selection lock,
post-lock audit, and final result revalidates it and records its path/SHA. Do
not edit implementation or experiment files after creating it. Run each
continuation as a new command; do not pass `--resume`.

Extract each frozen full-volume embedding only after that source's final
stage-2 `full_25ep/latest.pt` passes the full checkpoint audit. That second
read-only audit rechecks the base and also verifies the final live SHA, exact
base path in `continuation.init_checkpoint`, top-1 unfreezing, completed state,
FP32 state, 15,625 continuation steps, and parity to the pinned canonical final
checkpoint. The three selectable sources may be extracted and run through
medium validation while the fresh legacy continuation is still running on
another GPU; the selection lock does not use the legacy result. Extract legacy
only after its own continuation completes:

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export workspace first}"
: "${F3_ROOT:?export F3 root first}"
export EXP=experiments/f3/facies_benchmark_v2/111_local_barlow_twins_gaussian_view_v1
export VALIDATION_CONFIG="$EXP/30_validation/01_candidates.yaml"

for candidate in \
  local_barlow_twins_gaussian_noise_std005 \
  local_barlow_twins_gaussian_noise_std010 \
  local_barlow_twins_identity_gaussian_noise_std010
do
  python "$EXP/run_validation.py" \
    --config "$VALIDATION_CONFIG" \
    --candidate "$candidate" --audit-checkpoint-only
done

for config in \
  "$EXP/20_embeddings/01_extract_gaussian_noise_std005.yaml" \
  "$EXP/20_embeddings/02_extract_gaussian_noise_std010.yaml" \
  "$EXP/20_embeddings/04_extract_identity_gaussian_noise_std010.yaml"
do
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config" --dry-run
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config"
done

python "$EXP/run_validation.py" \
  --config "$VALIDATION_CONFIG" \
  --candidate local_barlow_twins_legacy_flip_25ep \
  --audit-checkpoint-only
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/03_extract_legacy_flip_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/03_extract_legacy_flip_25ep.yaml"
```

## Run the validation screens

The one-job CLI is intentionally separate from the canonical five-way runner.
Before any decoder training it audits both base and final checkpoint paths and
SHAs, the exact continuation-to-base link, the Local Barlow objective and pair
count, the exact named view policy and strength, base parity outside the
allowed augmentation/duration/output fields, and final parity outside only
augmentation/output/init-checkpoint fields. It then verifies the extraction
contract and a valid-token mask identical to the canonical random source. It
also runs the unchanged canonical five-source audit so the reused random
checkpoint and embedding are live and verified, then uses the canonical
runner's downstream config mappings. Each job audit binds the source to its
exact layout, size, metrics path, and protocol-lock path/SHA. Selection-lock
creation additionally verifies the
completed decoder config and initial-state identity, prediction source and
decoder checkpoint, and the evaluator's recorded metrics path/SHA. Every lock
input records both the base and final checkpoint SHA-256.

For `local_barlow_twins_legacy_flip_25ep`, the stricter attribution audit
requires the exact legacy flip mapping in both phases. Its base must record
epoch 25/global step 15,625 and may differ from the canonical base only in
duration and output root. Its final must record a separate epoch 25/global step
15,625 and may differ from the canonical top-1 continuation only in output root
and exact base init path. The audit records `selection_eligible: false`; that
flag excludes legacy only from three-way view selection, not from the duration
success gate.

First run the five medium layouts for exactly the three selectable candidates
(jobs may be distributed over GPUs). The runner forbids legacy medium and every
small/large job until the immutable selection lock exists:

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export workspace first}"
: "${F3_ROOT:?export F3 root first}"
export EXP=experiments/f3/facies_benchmark_v2/111_local_barlow_twins_gaussian_view_v1
export VALIDATION_CONFIG="$EXP/30_validation/01_candidates.yaml"

for candidate in \
  local_barlow_twins_gaussian_noise_std005 \
  local_barlow_twins_gaussian_noise_std010 \
  local_barlow_twins_identity_gaussian_noise_std010
do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    python "$EXP/run_validation.py" \
      --config "$VALIDATION_CONFIG" \
      --candidate "$candidate" --layout "$layout" --size medium --dry-run
    python "$EXP/run_validation.py" \
      --config "$VALIDATION_CONFIG" \
      --candidate "$candidate" --layout "$layout" --size medium
  done
done
```

Create the lock from exactly those 15 medium metrics and 15 candidate audits.
The command refuses to overwrite a lock and rejects any legacy or non-medium
candidate audit/metric that was materialized out of order. It records the
three means, fixed tie rule, selected ID/policy/noise, per-input paths and
metric/audit SHA-256s, both base and final checkpoint SHA-256s, explicit base
and continuation durations, fixed-strength identity-0.10 minus
forced-flip-0.10 contrast, protocol-lock path/SHA, UTC timestamp, and the
protocol Git HEAD:

```bash
python "$EXP/run_validation.py" \
  --config "$VALIDATION_CONFIG" --create-selection-lock
```

The earlier protocol lock, not this data-dependent selection lock, owns the
repository freeze. It records whether the relevant state is dirty, a digest of
the filtered relevant porcelain-status entries, and a path/status/SHA-256
inventory of every experiment file and every dirty source, CLI, or test file
under the relevant repository namespaces. Every operation re-derives that
inventory and fails if it drifts. Tracked human-readable results under
`reports/` are intentionally excluded from the frozen execution state and are
never pipeline inputs; publish them only after the immutable final result.

Only now run legacy on the five medium layouts:

```bash
for layout in layout_000 layout_001 layout_002 layout_003 layout_004
do
  python "$EXP/run_validation.py" \
    --config "$VALIDATION_CONFIG" \
    --candidate local_barlow_twins_legacy_flip_25ep \
    --layout "$layout" --size medium --dry-run
  python "$EXP/run_validation.py" \
    --config "$VALIDATION_CONFIG" \
    --candidate local_barlow_twins_legacy_flip_25ep \
    --layout "$layout" --size medium
done
```

The small/large gate revalidates all five locked-view, legacy, and canonical
random medium cells and opens only when the locked view or legacy has a strict
positive paired delta in all five layouts. Read the selected ID from the lock.
Run the locked view and legacy; when identity 0.10 is selected, also run the
forced-flip 0.10 geometry control:

```bash
export SELECTION_LOCK="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_barlow_twins_gaussian_view_v1/validation/gaussian25_selection_lock.json"
export SELECTED_CANDIDATE_ID="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_candidate_id"])' "$SELECTION_LOCK")"

arms=("$SELECTED_CANDIDATE_ID" local_barlow_twins_legacy_flip_25ep)
if [[ "$SELECTED_CANDIDATE_ID" == \
  local_barlow_twins_identity_gaussian_noise_std010 ]]
then
  arms+=(local_barlow_twins_gaussian_noise_std010)
fi

for candidate in "${arms[@]}"
do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    for size in small large
    do
      python "$EXP/run_validation.py" \
        --config "$VALIDATION_CONFIG" \
        --candidate "$candidate" --layout "$layout" --size "$size" --dry-run
      python "$EXP/run_validation.py" \
        --config "$VALIDATION_CONFIG" \
        --candidate "$candidate" --layout "$layout" --size "$size"
    done
  done
done
```

Record the three paired contrasts:
Gaussian-minus-random, legacy-minus-random, and Gaussian-minus-legacy. The
Gaussian random-success and Gaussian attribution must be reported as separate
gates. If identity 0.10 was locked, also record identity-0.10 minus
forced-flip-0.10 for every layout/size cell; all ten extra small/large geometry
cells are mandatory before declaring that arm complete. These small and large
results are validation tuning signals; do not
describe them as an untouched holdout or post-selection confirmation.

Every job audit binds the protocol-lock path/SHA. Each post-selection job audit
also binds the selection-lock path/SHA. Small and large job
audits also bind the five canonical random medium metric paths and SHA-256s that
opened the gate. After the exact reached cell set is complete, publish the
exclusive 25-epoch branch result:

```bash
python "$EXP/run_validation.py" \
  --config "$VALIDATION_CONFIG" --create-final-result
```

This mode rejects missing, duplicate, drifted, symlinked, or out-of-order cell
evidence. If the medium gate opened, it requires the locked view and legacy on
all 15 cells and, when identity 0.10 was selected, the complete 15-cell
forced-0.10 geometry control. It revalidates every live metric, job audit,
decoder/evaluation identity, candidate checkpoint lineage, the pinned random
baseline, canonical-config SHA, protocol-lock SHA, and selection-lock SHA. The immutable
`gaussian25_final_result.json` records candidate-minus-random,
legacy-minus-random, Gaussian-minus-legacy, and the geometry contrast
separately. A strict `15/15` arm produces `passed: true`; otherwise a complete,
identity-valid branch produces `passed: false` and
`authorizes_next_base_duration: true`. Missing or invalid evidence writes
nothing.

Only after that immutable result exists, build the tracked, report-only
projection. The builder replays the live final audit without writing, checks
every referenced SHA-256 again, refuses overwrite, and writes no pipeline
input:

```bash
python "$EXP/build_report.py" --config "$VALIDATION_CONFIG"
```

It publishes `attempts.csv`, `validation_cells.csv`, `paired_deltas.csv`,
`summary.json`, and `summary.md` under
`reports/f3/facies_benchmark_v2/local_barlow_twins_gaussian_view_v1/`.

Finish the 25-base-epoch branch while its protocol state is still valid. First
create the exclusive `gaussian25_final_result.json`; then create and preserve
the fixed 25-epoch report above. Stop if that final result passes. If it fails,
only after both immutable outputs exist may implementation for the
pre-registered 5-base-epoch contingency begin. Do not precreate any 5-epoch
producer, continuation, extraction, validation, or report config/file.

The reached 5-epoch branch must use only the locked view and matched legacy,
with immutable policy-duration IDs and distinct base, continuation, embedding,
validation, lock, and report roots. Implement a distinct duration-specific
runner/config that pins the parent `gaussian25_final_result.json` path and live
SHA-256 and requires that parent to have `passed: false` and
`authorizes_next_base_duration: true`. Before its validation starts, it must
create new 5-epoch protocol and selection locks; after its exact reached cells,
it must create a new exclusive final-result lock and publish outcomes only in a
new 5-epoch report directory. Apply the same medium-first `5/5` and final
`15/15` rules without retuning the locked view.

If and only if the immutable 5-epoch final result also fails, create its fixed
report while the 5-epoch protocol remains valid. Only then may the
pre-registered 1-epoch files be added. The distinct 1-epoch runner/config must
pin the failed 5-epoch final-result path/SHA and use new duration-specific
protocol, selection, and final locks plus a new report directory. Do not
precreate those files. Start every reached base fresh, retain the fixed
25-epoch continuation, do not resume across durations, and do not change the
locked view mapping.

The 25-epoch branch has now been completed and reported. Its immutable final
result has SHA-256
`33584533226842bac41dc9bac57e9479e4346caef2104eca60c595217997c15b`,
records `passed: false` and `authorizes_next_base_duration: true`, and selected
forced-flip Gaussian noise at `std=0.10`. The reached, duration-specific
base-5 definitions and runbook are therefore now present in
`40_base5ep/README.md`. That branch has also completed and been reported. Its
immutable final result has SHA-256
`5b1e9c8778892d1226179c92e7d9fd788cb13329e05e5663ed3dd8ddb1e18d43`,
records `passed: false`, `authorizes_next_base_duration: true`, and
`authorized_next_base_pretraining_epochs: 1`. The authorized, terminal
base-1 definitions and runbook are therefore now present in
`50_base1ep/README.md`; they inherit the locked Gaussian `std=0.10` view and
matched legacy control without retuning.

The currently defined 25-base-epoch jobs write under
`artifacts/.../local_barlow_twins_gaussian_view_v1/validation/runs/`. Each
reached shorter duration uses its own non-overlapping validation root.
`evaluation/metrics.json` contains validation metrics aggregated over unique
validation voxels; this runner defines no test evaluation or test output.
