# Parihaka amplitude Barlow Twins pretraining

This experiment pretrains the shared 3D amplitude encoder with a two-view
Barlow Twins objective. It is inspired by Salt3DNet's representation-learning
idea, not a reproduction of its segmentation architecture. The projection head
belongs to pretraining only; downstream consumers use the encoder.

Study-wide context is in the
[Parihaka benchmark overview](../../README.md). Configurations:

- [GPU feasibility](01_gpu_feasibility_1step.yaml)
- [full pretraining](02_full_100ep.yaml)

Run the feasibility condition before the full configuration. It is a capacity
gate for the intended device, not evidence of throughput or convergence.
Embedding extraction and downstream evaluation are outside this experiment.
