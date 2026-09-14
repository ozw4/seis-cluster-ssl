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
