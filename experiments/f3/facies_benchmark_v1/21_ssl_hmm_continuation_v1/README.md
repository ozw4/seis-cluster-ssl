# F3 SSL / HMM continuation v1

F3 prepared amplitude volume全体を未ラベルデータとして使用し、MAEとBarlow
Twinsを同じ固定予算で比較するtransductive SSL suiteである。NOPIMS checkpoint、
facies label split、および既存F3 HMM targetは使用しない。

## 実験行列

| Stage 1 | target source | Stage 2 objective | 主checkpoint |
| --- | --- | --- | --- |
| MAE 100 epoch | なし | MAE 25 epoch control | `stage2/mae100/mae_continue/full_25ep/latest.pt` |
| MAE 100 epoch | MAE100 K=6 target | single-head HMM-K6 25 epoch | `stage2/mae100/hmm/k6/full_25ep/latest.pt` |
| Barlow Twins 100 epoch | なし | Barlow Twins 25 epoch control | `stage2/bt100/bt_continue/full_25ep/latest.pt` |
| Barlow Twins 100 epoch | BT100 K=6 target | single-head HMM-K6 25 epoch | `stage2/bt100/hmm/k6/full_25ep/latest.pt` |

主比較は次の2組である。

- MAE100 → HMM-K6 25 と MAE100 → MAE25
- BT100 → HMM-K6 25 と BT100 → BT25

## 共通条件

- F3 full-volume manifestと`f3_npy_paths.txt`
- `128 x 128 x 128` crop、`8 x 8 x 8` patch
- encoder dim 384 / depth 8 / heads 6
- decoder dim 256 / depth 4 / heads 4
- batch size 16、10,000 samples/epoch、AdamW、weight decay 0.05
- FP32（`amp: false`）、seed 42、gradient clipping 1.0
- Stage 1は100 epoch、LR `1.0e-4`、62,500 global steps
- Stage 2は25 epoch、LR `1.0e-5`、encoder top-1、15,625 global steps
- HMMはsingle-head K=6、prototype / usage / distillation weight =
  `1.0 / 0.005 / 0.2`

artifact rootは`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}`で指定する。固定予算のsourceと
最終checkpointは次の配下に置く。

```text
pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/
├── stage1/mae/full_100ep/latest.pt
├── stage1/barlow_twins/full_100ep/latest.pt
└── stage2/
    ├── mae100/mae_continue/full_25ep/latest.pt
    ├── mae100/hmm/k6/full_25ep/latest.pt
    ├── bt100/bt_continue/full_25ep/latest.pt
    └── bt100/hmm/k6/full_25ep/latest.pt
```

## 実行entrypoint

主要configは次の4組である。

- `10_stage1/mae/02_full_100ep.yaml` / `10_stage1/barlow_twins/02_full_100ep.yaml`
- `30_stage2/mae100/mae_continue/02_full_25ep.yaml` / `30_stage2/bt100/bt_continue/02_full_25ep.yaml`
- `20_hmm_targets/mae100/01_extract_embeddings.yaml` / `20_hmm_targets/bt100/01_extract_embeddings.yaml`
- `30_stage2/mae100/hmm/k6/02_full_25ep.yaml` / `30_stage2/bt100/hmm/k6/02_full_25ep.yaml`

既存の`proc/seis_ssl_cluster/train_amp_mae.py`、
`proc/seis_ssl_cluster/train_amp_barlow_twins.py`、
`proc/seis_ssl_cluster/extract_embeddings.py`、
`proc/seis_ssl_cluster/cluster_embeddings.py`、
`proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py`、
`proc/seis_ssl_cluster/train_strat_hmm_pretext.py`を使用する。

prepare artifactの確認からtarget生成、学習、live checkpoint監査までの手順は
[RUNBOOK_HMM_K6.md](RUNBOOK_HMM_K6.md)を参照する。
