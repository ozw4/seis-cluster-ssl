# Parihaka Stage 1 SSL pretraining

## 目的

Parihaka固有の3D MAEと3D Barlow Twinsを同じ学習予算で事前学習し、HMM second-stage pretrainingの初期重みを作成する。

## 学習条件

両methodの共通条件は次のとおりである。

- Parihaka amplitude manifestとpath list
- amplitude preprocessing（normalized clip 8.0、trace RMS z-score AGC、window 65）とzero mask
- `128 x 128 x 128` crop、`8 x 8 x 8` patch
- encoder dim 384 / depth 8 / heads 6
- compatibility decoder dim 256 / depth 4 / heads 4
- batch size 16、learning rate `1.0e-4`、weight decay 0.05
- FP32（`amp: false`）、seed 42、gradient clipping 1.0

full runは100 epoch、10,000 samples/epochであり、625 steps/epoch、合計62,500 global stepsとなる。

MAEはspatial mask ratio 0.75、MSE reconstruction、gradient weight 0.0、visible reconstruction weight 0.1、patch z-score target normalizationを使用する。Barlow Twinsはhorizontal flip probability 0.5、projector dim 384、redundancy weight 0.005、normalization epsilon `1.0e-4`を使用する。

artifact output rootは次の配下に分離して保存する。

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/
├── mae/
│   ├── gpu_feasibility_1step/
│   └── full_100ep/
└── barlow_twins/
    ├── gpu_feasibility_1step/
    └── full_100ep/
```

## 実行順

リポジトリrootで環境変数を設定する。

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export STAGE1=experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1/10_stage1
```

1. 専用の契約テストを実行する。

   ```bash
   pytest -q tests/seis_ssl_cluster/test_parihaka_stage1_ssl_configs.py
   ```

2. 4設定をdry-runして解決結果を確認する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/01_gpu_feasibility_1step.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/02_full_100ep.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/01_gpu_feasibility_1step.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/02_full_100ep.yaml" --dry-run
   ```

3. Barlow TwinsのCUDA 1-step feasibilityを実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/01_gpu_feasibility_1step.yaml"
   ```

4. MAEのCUDA 1-step feasibilityを実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/01_gpu_feasibility_1step.yaml"
   ```

5. 両方のfeasibilityが完了条件を満たした場合に限り、100 epoch本学習を実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/02_full_100ep.yaml"
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/02_full_100ep.yaml"
   ```

## 完了判定

各GPU feasibilityでは、CUDA OOMが発生しないこと、lossとgradient normがfiniteであること、`latest.pt`が保存されること、peak CUDA memoryに本学習を継続できる余裕があることを確認する。Barlow Twinsではprojection metricsとcorrelation metricsもfiniteであることを確認する。

各100 epoch runは、epoch 100、global step 62,500に到達し、対応する`full_100ep/latest.pt`が存在することを確認する。resolved configがFP32 / `amp: false`であり、lossとgradient metricsが全期間でfiniteであることも必要とする。

Stage 2の初期値には`best.pt`ではなく、固定学習予算を完了した各methodの`full_100ep/latest.pt`を使用する。
