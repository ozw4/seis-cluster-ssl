# Preparation and execution evidence

Prepared and queued on 2026-11-17. This records implementation checks and
ongoing GPU trials, **not completion of the nine continuations or 135 cells**.

## Scientific checks

- All nine frozen k-means target manifests passed the existing unordered-target
  auditor. Manifest and parent checkpoint hashes are bound in `matrix.yaml`.
- Source feature metadata identifies the original parent encoder for every arm.
  Valid-token support hashes match the corresponding zero/zero HMM manifests.
- The complete read-only preflight passed: nine target configurations and
  eighteen smoke/full training configurations. No targets were regenerated.
- Completed GPU trials require exactly one optimizer step, positive prototype
  loss, exactly zero weighted distillation and consistency losses, and initial
  student/head parameter hashes equal to the corresponding HMM run.
- The detached supervisor owns the remaining smoke and full pipeline. It
  reuses completed trials, and full-stage admission verifies all nine trial
  payloads before starting any continuation. The earlier waiting-only
  preflight process was retired without stopping a training child; its
  completed trial payloads and handover receipt are preserved.

## Validation commands and results

```bash
.venv/bin/python -m pytest -q \
  tests/seis_ssl_cluster/test_kmeans_zero.py
# 18 passed, including CPU zero-weight training and catalog replacement safety
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q \
  -m 'not slow and not requires_segy and not requires_cuda'
# Initial run: 6918 passed, 6 failed, 145 deselected.
# After the corrections below, all six failing tests passed on retry.
.venv/bin/python -m pytest -q --lf \
  -m 'not slow and not requires_segy and not requires_cuda'
# 6 passed; the additional admission/catalog regressions also passed.
.venv/bin/python tools/check_seis_ssl_cluster_isolation.py
shellcheck experiments/kmeans/kmeans_k6810_dis0_con0_v1/run_all.sh
git diff --check
```

The broad-run failures identified missing declarations for the new F3/Volve
experiment namespaces and two issues in the previous stopped Parihaka repeat:
a directory link where a file link was required, and its missing standard
`build_parser` export. Dedicated inventory coverage now includes the new
definitions; both previous CLI/document issues are corrected. The stopped
experiment remains stopped. Full affected config/document suites also passed.
Successful unrelated portable tests were reused rather than repeated.

Changed Python files pass Ruff lint with `--no-fix` and formatting checks.
The new tests also cover altered source hashes, missing sources, changed
loss/output settings, unchanged zero-weight guards, partial publication,
shared F3 mask/voxel counts, Volve error sign, complete file ownership,
execution failure recording, and rejecting full training before smoke checks.

## Operational receipts

Under the artifact root, use `operations/kmeans/kmeans_k6810_dis0_con0_v1/`:

- `launch.json`, `supervisor.log`, `exit.json`: detached full-pipeline identity
  and final exit code; the exit receipt appears only after termination.
- `status.json`: current stage, command, log path and waiting/failure state.
- `smoke_*.json`, `all_smokes_verified.json`: actual trial and barrier checks.
- `smoke_driver.log`, `full_driver.log`, timestamped stage logs: execution.
- `validation.json` and copied validation logs: preparation evidence.

Check the final complete summary and exit receipt before reporting performance
or experiment completion. Later CSV/HTML/ZIP publication preserves every
previously saved evaluation and retains a backup of the old catalog.
