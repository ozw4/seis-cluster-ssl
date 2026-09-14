# Implementation validation

Validated base commit: `804cb9ec211add5825f3c1af3d2f830b614222fe`, plus the working-tree implementation.
Date: 2026-09-14 (UTC).

| Check | Result |
| --- | --- |
| `pytest -q -m "not slow and not requires_segy and not requires_cuda"` | PASS: 5,606 tests; 3 deselected; 441.42 seconds |
| `pytest -q tests/seis_ssl_cluster/test_hmm_k6810_receipts_aggregate.py tests/seis_ssl_cluster/test_f3_facies_benchmark_v2_configs.py` | PASS: 91 tests, including the final F3 control-reference fix |
| `python -m compileall -q src proc tests` | PASS |
| `python -m ruff check .` | PASS |
| `python tools/check_seis_ssl_cluster_isolation.py` | PASS |
| `git diff --check` | PASS |
| `bash -n` on every new survey shell script | PASS |
| Each survey `run_all.sh --plan` | PASS: full training 2/3/3; downstream 30/45/45; command totals 47/70/70 |
| Receipt CLI `--check` using the three versioned definition files | PASS |
| All candidate/control summary references through their public parsers | PASS: 18 references |

All eight training/embedding/target configurations resolve with portable
synthetic manifest and checkpoint fixtures. Regression tests cover source
family inheritance, fixed scientific settings, no F3 MAE write commands,
own-latest resume, lock conflicts, failed-command propagation, partial-output
protection, exact receipt cells, metric/source drift, paired completion and
support, 27 aggregate rows, 135 cells and exactly three summary output files.

Independent implementation and review used separate Astra medium agents.
All review findings were resolved. Existing HMM_v1 records and experiment 127
definitions are unchanged; the two pre-existing user edits are preserved
exactly. The final F3 Random control reference was separately checked to retain
its original source and output identities while omitting the unsupported
five-way summary field.

Existing evidence was checked read-only: the F3 reuse receipt matches its
checkpoint metadata, resolved recipe and recorded live hashes; all 135 frozen
matching-control metric cells agree with their receipts. These checks do not
execute or regenerate experiments.

New live clustering, GPU pretraining, embedding extraction and downstream
training were not run. Their execution outputs and complete end-to-end live
runs remain unverified. Dry-run stages that require future outputs must be run
after their predecessors have produced those artifacts.

Experiment-start decision: **GO for implementation readiness**. New live
artifacts and complete live executions remain unverified.

## Source-family preflight

Validated base: `7602c08`, plus the source-family preflight change.

```bash
pytest -q tests/seis_ssl_cluster/test_hmm_v2_multi_source_definition.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_configs.py \
  tests/seis_ssl_cluster/test_hmm_k6810_receipts_aggregate.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_driver.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_adapters.py \
  tests/seis_ssl_cluster/test_f3_facies_benchmark_v2_configs.py
```

PASS: 201 tests. Raw-definition checks require no live artifacts or environment
expansion. Both driver execution and the direct summary CLI reject inheritance
drift before artifact reads/writes. Summary input paths must identify the
validated experiment files. The previous test-only source comparisons now run
through the production validator.

`python -m ruff check --no-fix .`, `python -m compileall -q src proc tests`,
`python tools/check_seis_ssl_cluster_isolation.py`, and `git diff --check`: PASS.
No live experiment stages were executed. Implementation readiness remains GO.

## Aggregate metric identity

Validated base: `b72104b`, plus the aggregate metric identity change.

`pytest -q tests/seis_ssl_cluster/test_hmm_k6810_receipts_aggregate.py`: PASS,
69 tests. Nine regression cases alter the top-level primary metric, primary
direction or secondary metric across all three surveys, update both summary
digests, and verify the aggregate rejects the specific contract mismatch before
publishing any output. Existing exact-output and successful aggregation checks
also pass.

`python -m ruff check --no-fix` and `python -m ruff format --check` on
`src/seis_ssl_cluster/hmm/multi_source_aggregate.py` and
`tests/seis_ssl_cluster/test_hmm_k6810_receipts_aggregate.py`: PASS.
`git diff --check`: PASS. No live experiment stages were executed.

## Frozen historical K6 targets

Validated base: `39b027e`, plus the frozen K6 evidence change.

```bash
pytest -q tests/seis_ssl_cluster/test_strat_hmm_source_audit.py \
  tests/seis_ssl_cluster/test_strat_frozen_k6_reference.py \
  tests/seis_ssl_cluster/test_strat_multi_head_target_manifest.py
```

PASS: 81 tests, including the final frozen-source audit regression. Frozen
manifests require exactly one evidence mode, reject changed target, embedding,
checkpoint, reference-config and receipt hashes, and cannot be replaced through
a replay request. Legacy replay manifests remain supported.

All eight actual receipt freeze commands passed `--dry-run`, then created
`frozen_k6_reference.json` in each new arm's artifact namespace. This only read
existing artifacts and wrote receipts; no clustering, target export, training,
embedding extraction or downstream execution was performed. Independent review
verified all 40 historical K6 files (including boundary weights) and the existing
F3 MAE manifest retained their original hashes. The latter also passes the
current manifest validator with its original replay evidence; its SHA-256 is
`ede820efae53c73c4177023207d1bf1ff24ac7428b10e6aa43d84d40d216e1cd`.

Plans retain 2/3/3 full runs and 30/45/45 downstream cells. Command totals remain
47/70/70: each deleted replay command is replaced by one receipt-freeze command
inside the manifests stage. The eight replay configs and K6 export commands
are removed. Existing F3 MAE has no write commands.

`pytest -q -m "not slow and not requires_segy and not requires_cuda"`: PASS,
5,760 tests; 3 deselected; 11 warnings; 446.55 seconds. The final added
frozen-source audit regression also passed in the 81-test focused run above.
`python -m compileall -q src proc tests`, `python -m ruff check --no-fix .`,
scoped Ruff formatting checks, `python tools/check_seis_ssl_cluster_isolation.py`,
`bash -n` on the survey shell scripts, and `git diff --check`: PASS.
Independent Astra medium review has no unresolved findings. Implementation
readiness remains GO; new live training and end-to-end execution remain unverified.
