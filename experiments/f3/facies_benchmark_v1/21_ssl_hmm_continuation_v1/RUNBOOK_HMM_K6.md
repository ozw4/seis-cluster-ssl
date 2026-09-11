# F3 paired SSL / HMM-K6 execution runbook

この runbook は実行順と再開境界だけを定めます。学習条件、入力、出力、lineage は各 config の resolved dry-run を確認してください。以下の各 Python command は、同じ引数に `--dry-run` を付けて検証してから実行します。

## 環境

```bash
set -euo pipefail
export SUITE=experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export STAGE1_CONFIGS="$SUITE/10_stage1"
export TARGET_CONFIGS="$SUITE/20_hmm_targets"
export STAGE2_CONFIGS="$SUITE/30_stage2"
```

## Stage 1

各手法で feasibility を完了してから full run へ進みます。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1_CONFIGS/mae/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1_CONFIGS/mae/02_full_100ep.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1_CONFIGS/barlow_twins/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1_CONFIGS/barlow_twins/02_full_100ep.yaml"
```

## Control branches

Stage 1 の両 checkpoint が揃ってから、同じ手法の固定予算 control を feasibility、full の順で実行します。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE2_CONFIGS/mae100/mae_continue/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE2_CONFIGS/mae100/mae_continue/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE2_CONFIGS/bt100/bt_continue/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE2_CONFIGS/bt100/bt_continue/02_full_25ep.yaml"
```

## HMM target branches

各 Stage 1 source について embedding 抽出、clustering、pseudo-target export の順を守ります。MAE と Barlow Twins の target lineage を混在させないでください。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/mae100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/mae100/k6/02_cluster_hmm_k6.yaml"
bash "$TARGET_CONFIGS/mae100/k6/03_export_pseudo_targets.sh" --dry-run
bash "$TARGET_CONFIGS/mae100/k6/03_export_pseudo_targets.sh"

python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/bt100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/bt100/k6/02_cluster_hmm_k6.yaml"
bash "$TARGET_CONFIGS/bt100/k6/03_export_pseudo_targets.sh" --dry-run
bash "$TARGET_CONFIGS/bt100/k6/03_export_pseudo_targets.sh"
```

export 後に feasibility、full の順で HMM Stage 2 を実行します。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml"
```

## 再開

fresh stage には `--resume` を付けません。中断した run だけを、その run 自身の `latest.pt` から再開します。Stage 2 を Stage 1 checkpoint から `--resume` したり、別手法・別 branch の checkpoint を流用したりしないでください。
