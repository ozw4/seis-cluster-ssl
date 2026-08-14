# F3 M5-LS lateral hard-target artifact producers

M5-LS creates offline hard pseudo-targets using one XY four-neighbour,
source-embedding cosine-RBF lateral mean-field message and one ordered Viterbi
reprojection. Training consumes only hard labels through the existing
multi-head data, collate, and loss route.

The fixed target candidates are `beta010`, `beta025`, and `beta050`.
Run YAML stages `01` through `04` with
`export_strat_hmm_multi_head_lateral_targets.py` and
`calibrate_f3_m5_lateral_targets.py`. Then run stages `05` through `08`
with `train_strat_hmm_pretext.py`, `extract_embeddings.py`, and
`validate_f3_m5_lateral_smoothing_pretraining.py`.

No facies labels or downstream metrics are target inputs. Complete outputs
remain under `artifacts/seis_ssl_cluster/`.
