# Multi-head handoff: current-code K=6 control

Control status: `CONTROL_READY_POSITIVE`

Control model tag:

`strat_hmm_pretext_m1_current_k6_topblock1_distill_v1`

Fixed handoff contract:

- Teacher: same MAE latest checkpoint.
- Student initialization: same MAE latest checkpoint.
- K=6 target: same exact K=6 pseudo-target identity.
- Pretraining: same scientific settings.
- Downstream: same cap25/cap50/cap100 datasets, five paired seeds, and voxel decoder.

Final repository provenance:

- HEAD: `371ec7ae4156fdf551a11dd1a9d7ae31ec2ed3ec`
- `git diff --binary HEAD` SHA-256: `cbe069e5b564daf1a4d7e0024cbc502f391c590de3e40cfb3d167c6d02a54af4`
