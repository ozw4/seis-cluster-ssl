# 40_random — Random encoder seed 42（v1 checkpointの明示的再利用）

`random`のfrozen encoderは、v1の次のconfigが生成するseed 42のrandom-init
checkpointをそのまま参照する。v2は2つ目のrandom encoderを作らない。

- producer config:
  `experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1/40_random/01_create_random_checkpoint.yaml`
- checkpoint:
  `pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/random/random_init.pt`

five-way source auditは`metadata.seed == 42`、`epoch == 0`、
`reference_checkpoint`が`mae` checkpointと一致することを検証する。
