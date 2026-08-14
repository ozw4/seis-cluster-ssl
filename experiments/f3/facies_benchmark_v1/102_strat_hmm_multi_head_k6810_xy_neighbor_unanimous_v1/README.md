# F3 unanimous XY-neighbour hard-label artifact producers

This experiment applies one synchronous source-label correction only where all
valid same-z XY neighbours agree on a different label: 4/4 with four valid
neighbours and 3/3 with three. It produces schema-6 targets, checkpoints,
embeddings, and validation evidence.

Run stages `01` through `06` in order with
`export_strat_hmm_multi_head_xy_neighbor_unanimous_targets.py`,
`audit_f3_xy_neighbor_unanimous_targets.py`,
`train_strat_hmm_pretext.py`, `extract_embeddings.py`, and
`validate_f3_xy_neighbor_unanimous_pretraining.py`.

Complete outputs remain under `artifacts/seis_ssl_cluster/`.
