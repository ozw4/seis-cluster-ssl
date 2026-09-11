# Performance migration validation stage artifacts

Status, decision, and policy fields are recorded in
[`performance_migration_summary.md`](performance_migration_summary.md).

| Stage | Exists | SHA-256 |
| --- | --- | --- |
| preflight | True | `c9a04aeea5c9975e7ff4e0be6f352786b6d748edcb99823cee969cc215cfd167` |
| checkpoint_smoke | True | `0bd4a7ac7fb0284e025ea65dd63067ba46beef1d2e69d81c5fa94766b4cee51d` |
| embedding_parity | True | `dfeaf723a4b661c78d02fca1fbb2a80c3a9947f8bdf7c4e8f56d9411fd0c2139` |
| probe_parity | True | `889846396465d6fa3a70fe61c69c3d58ab9903b97ea3e266aa3ecd20a98bdcdd` |
| historical_hmm_config | True | `389b9b348fd256edab0221b3c02d518edf4e40c7ec3597f90d5f4f14fc933371` |
| hmm_parity | True | `4e5daf195727818b5e0224cc785ef0c9cd8d9cc9f346c0761f0e9d02d8de124a` |
| pseudo_target_parity | True | `d37e00a558e9cea8609c0f7dfe8d02bc9c6415371cb7c0ad0edc2e26787e79b7` |
| benchmark | True | `0b32f5645aae763dbb9a3c87e4bb9fa63fb8f87f76a916a4a6a578fcaf43d346` |

Historical checkpoints, embeddings, HMM artifacts, pseudo-targets, probes, and
M3-V/M3-V-LB outputs were read-only inputs.
