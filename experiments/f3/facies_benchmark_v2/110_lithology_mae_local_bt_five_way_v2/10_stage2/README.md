# 10_stage2 — Local BT 100+25 control（v1 checkpointの明示的再利用）

v2はencoderを再学習しない。`local_barlow_twins`のfrozen encoderは、v1の
fixed-budget continuation configが生成する次のcheckpointをそのまま参照する。

- producer config:
  `experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1/10_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml`
- checkpoint:
  `pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/stage2/local_bt100/local_bt_continue/full_25ep/latest.pt`

v2側にはtraining configを置かない。checkpointをv2 directoryへコピー・symlinkして
version表記だけを変える運用も行わない。
