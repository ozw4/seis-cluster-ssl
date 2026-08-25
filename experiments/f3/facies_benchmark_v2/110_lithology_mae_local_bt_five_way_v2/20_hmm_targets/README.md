# 20_hmm_targets — Local BT由来HMM-K6 pseudo target（v1 artifactの明示的再利用）

`local_barlow_twins_hmm_k6`の学習に使うHMM-K6 pseudo targetは、v1の次のstageが
生成するartifactを参照する。v2はtargetを再生成しない。

- producer configs:
  `experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1/20_hmm_targets/local_bt100/01_extract_embeddings.yaml`
  `experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml`
  `experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1/20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh`
- pseudo target root:
  `pseudo_targets/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/local_bt100`

five-way source auditは、`local_barlow_twins_hmm_k6` checkpointが記録する
`pseudo_target_input_dir`が`mae_local_bt_five_way_v1/local_bt100`で終わることと、
trace-drop系targetでないことを検証する。
