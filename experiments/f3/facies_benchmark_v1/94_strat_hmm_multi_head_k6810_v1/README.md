# F3 K=6/8/10 multi-head artifact producers

This experiment compares ordered-prototype pretraining with and without a
cross-head consistency term. Target lineage and all exact conditions belong to
the YAML and validators.

Run the target replay/export/build stages before training, then run smoke,
full pretraining, checkpoint validation, embedding extraction, and complete
validation in numeric YAML order. The generic entrypoints are
`cluster_embeddings.py`, `export_strat_hmm_multi_head_pseudo_targets.py`,
`build_strat_hmm_multi_head_targets.py`, `train_strat_hmm_pretext.py`,
`extract_embeddings.py`, and `validate_f3_multi_head_pretraining.py`.
