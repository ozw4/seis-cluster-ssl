# Volve five-way distillation 0.10

MAE-HMM と Local-BT-HMM の Stage 2 だけで distillation weight を変更し、親の
Stage 1、pseudo target、plain model、Random baseline を再利用する比較条件である。
正確な差分、入力、出力先はこの directory の YAML を正本とする。

親 suite と shared [Volve benchmark environment](../../README.md) を先に確認する。

下流学習は 1 回の trajectory で 3 種類の checkpoint selection を保存する。
`50_five_way_macro_mae.yaml` が学習を所有し、残る 2 YAML は同じ run から別の
selection view を集計する。成果物名、alias、完了条件は runner と summary の検証を
正本とし、ここでは再掲しない。

```bash
export EXP="$VOLVE_EXP/31_mae_local_bt_hmm_five_way_v1/60_distill010"

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/mae100_hmm_k6_25ep.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/local_bt100_hmm_k6_25ep.yaml"

for config in "$EXP"/40_embeddings/*.yaml; do
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config" --device cuda --skip-existing
done

python proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py \
  --config "$EXP/50_five_way_macro_mae.yaml" \
  --layout-config "$VOLVE_LAYOUTS" --device cuda --continue

for config in "$EXP"/5?_five_way_*.yaml; do
  python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
    --config "$config" --check-only
  python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
    --config "$config"
done
```
