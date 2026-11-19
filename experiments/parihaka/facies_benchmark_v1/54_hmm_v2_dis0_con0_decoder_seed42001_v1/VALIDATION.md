# Validation and launch evidence

Prepared and launched on 2026-11-17. This records preparation and launch,
**not completion of the thirty decoder experiments**.

Validation passed:

```bash
.venv/bin/python -m pytest -q \
  tests/seis_ssl_cluster/test_parihaka_hmm_zero_seed_repeat.py \
  tests/seis_ssl_cluster/test_parihaka_channel_seed_repeat.py
# 34 passed
.venv/bin/python -m ruff check --no-fix \
  src/seis_ssl_cluster/parihaka/hmm_zero_seed_repeat.py \
  proc/seis_ssl_cluster/run_parihaka_hmm_zero_seed_repeat.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_zero_seed_repeat.py
.venv/bin/python -m ruff format --check \
  src/seis_ssl_cluster/parihaka/hmm_zero_seed_repeat.py \
  proc/seis_ssl_cluster/run_parihaka_hmm_zero_seed_repeat.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_zero_seed_repeat.py
.venv/bin/python tools/check_seis_ssl_cluster_isolation.py
git diff --check
```

Tests cover fixed encoder and benchmark settings, canonical seed rejection,
all-source/budget/layout scope, provenance drift, changed supervision, wrong
completed identity, all thirty comparison rows including negative changes,
six five-layout summaries, exact published file membership, partial-summary
rejection, own-checkpoint resumption, completed-cell reuse, and failure admission.

The real-data dry-run verified seed-only benchmark equivalence for the initial
MAE cells before launch. The full read-only dry-run continues independently;
each writing cell also requires its own successful preflight. The detached
serial runner was launched on GPU 1, preserving the unrelated workload on GPU 0.
No original seed-42000 metrics, checkpoints or embeddings were modified.

Ignored operational evidence under the artifact root:

`operations/parihaka/hmm_v2_dis0_con0_decoder_seed42001_v1/`

- `launch.json`: fixed thirty-cell scope, definition/code hashes, GPU, worker PID.
- `validation.json`: preparation check receipt.
- `status.json`: current cell, completed cells, running/failure/completion state.
- `runner.log`: per-cell admission, completion and any traceback.
- `exit.json`: written by the detached supervisor when the runner terminates.

The final summary requires all thirty completed cells and is intentionally
absent while execution is incomplete. Check `status.json`, `exit.json`, and
the complete summary before reporting final performance or completion.
