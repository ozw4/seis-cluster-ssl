# F3 K=6/8/10 multi-head artifact producers

This directory retains immutable target generation, paired pretraining,
embedding extraction, and producer validation for the K=6/8/10 ordered-prototype
encoders. The two variants differ only in consistency weight:
`mh_nocons` uses `0.0` and `mh_cons010` uses `0.1`.

Run the target replay/export/build stages before training, then run smoke,
full pretraining, checkpoint validation, embedding extraction, and complete
validation in numeric YAML order. The generic entrypoints are
`cluster_embeddings.py`, `export_strat_hmm_multi_head_pseudo_targets.py`,
`build_strat_hmm_multi_head_targets.py`, `train_strat_hmm_pretext.py`,
`extract_embeddings.py`, and `validate_f3_multi_head_pretraining.py`.

Complete outputs remain under `artifacts/seis_ssl_cluster/`. This directory
contains only target production, pretraining, embedding extraction, and
producer validation stages.
