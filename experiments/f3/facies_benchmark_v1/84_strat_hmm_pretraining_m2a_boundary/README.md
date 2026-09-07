# F3 Strat-HMM M2-A boundary-weighted pretraining producers

This experiment changes only pseudo-target boundary weighting from the M1
pretraining condition. It is a fixed treatment rather than a parameter sweep;
the exact weighting belongs to the YAML.

Run stages `01` through `05` in numeric order: parity and boundary-weighted
target export, smoke/full pretraining, then embedding extraction. The stage files
own their commands and settings.
