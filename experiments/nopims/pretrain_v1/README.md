# NOPIMS pretrain v1

Experiment configs for the NOPIMS amplitude-only MVP pretraining pipeline.

Source-of-truth inputs:

- Raw NOPIMS root: `/home/dcuser/data/NOPIMS`
- Artifact root: `/workspace/artifacts/seis_ssl_cluster`
- Training path-list: `/workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1/train_npy_paths.txt`

Each YAML is intentionally standalone and avoids inheritance, anchors, merge
keys, and symlinks.


## amp_mae_m025_mse_g0_patchnorm_v1

`10_pretrain/amp_mae_m025_mse_g0_patchnorm_v1/03_full_100ep.yaml` defines the mask 0.25, MSE, gradient-weight 0 experiment with target-only patch z-score normalization (`eps: 1.0e-6`, `min_std: 0.05`). It changes only the MAE loss target; encoder inputs and dataset targets remain survey-wise normalized amplitudes. This repository change adds the YAML only and does not start training.
