# F3 Local VICReg v1

Local Barlow Twins と入力・model・augmentation・学習予算を揃え、objective だけを Local VICReg に置き換える baseline と、その固定予算 control / HMM-K6 branch を生成します。科学条件と artifact binding はこの directory の YAML を正本とします。

screening と lithology extension は [116_local_vicreg_extension_v1](../116_local_vicreg_extension_v1/README.md) に集約しています。

## Stage 1

feasibility、full training、prepared-volume parity、v2 embedding の順です。以下の各 Python command は、同じ引数に `--dry-run` を付けて検証してから実行します。

```bash
set -euo pipefail
export VICREG_EXP=experiments/f3/facies_benchmark_v2/115_local_vicreg_v1

python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_EXP/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_EXP/02_full_100ep.yaml"
python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py --dry-run
python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$VICREG_EXP/03_extract_v2_embeddings.yaml"
```

## Screening gate

Stage 1 embedding の後、linked extension runbook の screening source audit、screen jobs、screen summary を完了します。gate が pass するまで以降の control / HMM branch と full extension を開始しません。判定の正本は screening summary です。

## Control branch

gate pass 後、feasibility から full continuation へ進みます。

```bash
set -euo pipefail
export VICREG_CONTROL="$VICREG_EXP/10_stage2/vicreg100/vicreg_continue"
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_CONTROL/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_CONTROL/02_full_25ep.yaml"
```

## HMM branch

Stage 1 checkpoint から embedding、clustering、pseudo-target export を完了してから HMM feasibility と full training を実行します。

```bash
set -euo pipefail
export VICREG_TARGET="$VICREG_EXP/20_hmm_targets/vicreg100"
export VICREG_HMM="$VICREG_EXP/30_stage2/vicreg100/hmm/k6"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$VICREG_TARGET/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$VICREG_TARGET/k6/02_cluster_hmm_k6.yaml"
bash "$VICREG_TARGET/k6/03_export_pseudo_targets.sh" --dry-run
bash "$VICREG_TARGET/k6/03_export_pseudo_targets.sh"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$VICREG_HMM/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$VICREG_HMM/02_full_25ep.yaml"
```

## 再開

各 stage の fresh run には `--resume` を付けません。中断した stage だけを、その stage 自身の `latest.pt` から再開します。Stage 1 checkpoint は continuation の初期値であり、Stage 2 の resume checkpoint ではありません。別 branch の checkpoint を流用せず、既存 output を暗黙に上書きしないでください。
