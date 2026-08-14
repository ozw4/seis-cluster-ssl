# F3 XY-neighbour consensus hard-label artifact producers

This experiment creates one immutable K=6/8/10 hard-target artifact with a
single synchronous XY four-neighbour consensus pass and an ordered-trace safety
guard. Target generation does not read facies labels or downstream metrics.

Run stages `01` through `05` in order with
`export_strat_hmm_multi_head_xy_neighbor_consensus_targets.py`,
`train_strat_hmm_pretext.py`, `extract_embeddings.py`, and
`validate_f3_xy_neighbor_consensus_pretraining.py`. The smoke root is
separate from the full root and must not be used to resume full training.

Complete outputs remain under `artifacts/seis_ssl_cluster/`. The former
low-label screening and tracked review publication are retired.
