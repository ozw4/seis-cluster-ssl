# Config Validation Split Summary

Config validation is divided into stage-focused modules while
`config/validate.py` continues to provide the public resolver facade.

Active resolvers use paths written explicitly in YAML or supplied by a CLI.
They do not reconstruct or require a dataset/version/model/spec directory
hierarchy. Runtime checks still cover required input/output fields, missing
inputs, source/output collisions, and overwrite protection.

The checked-in YAML path values were retained during this change. Complete
local artifacts, checkpoints, embeddings, and large arrays remain outside the
repository; `results/` remains reserved for lightweight review artifacts.

Relevant verification includes:

```bash
PYTHONPATH=src pytest -q \
  tests/seis_ssl_cluster/test_active_experiment_configs.py \
  tests/seis_ssl_cluster/test_config.py \
  tests/seis_ssl_cluster/test_proc_entrypoints.py
```

For `results/`, each producer owns its explicit lightweight file set. Verify
that set in focused tests and inspect `git diff` during review.
