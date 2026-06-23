# NOPIMS pretrain v1

Experiment configs for the NOPIMS amplitude-only MVP pretraining pipeline.

Source-of-truth inputs:

- Raw NOPIMS root: `/home/dcuser/data/NOPIMS`
- Artifact root: `/workspace/artifacts/seis_ssl_cluster`
- Training path-list: `/workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1/train_npy_paths.txt`

Each YAML is intentionally standalone and avoids inheritance, anchors, merge
keys, and symlinks.
