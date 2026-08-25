# 30_stage2 — Local BT 100 + HMM-K6 25（v1 checkpointの明示的再利用）

`local_barlow_twins_hmm_k6`のfrozen encoderは、v1の次のconfigが生成する
checkpointをそのまま参照する。v2はStage 2を再学習しない。

- producer config:
  `experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml`
- checkpoint:
  `pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/stage2/local_bt100/hmm/k6/full_25ep/latest.pt`

MAE系2件（`mae`, `mae_hmm_k6`）も同様にv1 `21_ssl_hmm_continuation_v1`のStage 2
checkpointを参照する。

- `mae`: `pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/mae_continue/full_25ep/latest.pt`
- `mae_hmm_k6`: `pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/hmm/k6/full_25ep/latest.pt`
