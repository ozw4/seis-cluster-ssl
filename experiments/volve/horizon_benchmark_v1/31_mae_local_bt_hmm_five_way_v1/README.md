# Volve horizon MAE / Local Barlow Twins five-way v1

Volve の 5 ホライゾン推定で、MAE、Local Barlow Twins、それぞれの HMM
継続学習、および Random encoder を同じ下流条件で比較する実験である。Stage 1、
HMM target、Stage 2 はホライゾンラベルを参照せず、ラベルは frozen decoder の
学習と評価でのみ使用する。

共通 supervision の科学的境界と定義の所有関係は
[`docs/volve_horizon_supervision.md`](../../../../docs/volve_horizon_supervision.md)
を参照し、実行順はこの runbook が所有する。

## 実行順

shared [Volve benchmark environment](../README.md) を設定してから、この suite の
root を選ぶ。canonical input と MAE source は
[MAE pretraining runbook](../10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md)
で先に準備する。MAE HMM target の前提となる embedding は
[frozen comparison の MAE extraction](../30_mae_vs_random_frozen_v1/README.md#execution-order)
で先に生成する。

```bash
export EXP="$VOLVE_EXP/31_mae_local_bt_hmm_five_way_v1"
```

1. canonical input を準備する。
2. `10_stage1/` の Local BT を学習する。
3. MAE embedding を抽出し、`20_hmm_targets/` の MAE 系と Local BT 系の target を
   別々に生成する。
4. `30_stage2/` の plain continuation と HMM continuation を学習する。
5. `40_embeddings/` の 5 config で embedding を抽出する。
6. source audit 後に下流行列を実行し、完了後に summary を生成する。

Stage 1 は epoch 数を揃えた比較であり、optimizer update 数は batch size により
異なる。Stage 2 の固定 budget を Stage 1 の epoch 数から推定しない。

書き込み前には、対応 CLI が提供する読み取り専用の事前確認（`--dry-run` または
`--check-only`）を先に実行する。

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_stage1/local_barlow_twins/02_full_100ep.yaml"

python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml"
bash "$EXP/20_hmm_targets/mae100/k6/03_export_pseudo_targets.sh"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/01_extract_embeddings.yaml" \
  --device cuda --skip-existing
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml"
bash "$EXP/20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh"

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$EXP/30_stage2/mae100/mae_continue/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/30_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml"

for config in \
  "$EXP/30_stage2/mae100/hmm/k6/02_full_25ep.yaml" \
  "$EXP/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml"; do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config"
done

for config in "$EXP"/40_embeddings/*.yaml; do
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config" --device cuda --skip-existing
done
```

下流処理の前に source audit を通す。通常の MAE selection 行列は同梱 launcher を
使う。中断した行列だけ明示的に `--continue` で再開する。

```bash
python proc/seis_ssl_cluster/audit_volve_horizon_five_way_sources.py \
  --config "$EXP/50_five_way.yaml"

DRY_RUN=1 bash "$EXP/run_five_way.sh"
DEVICE=cuda bash "$EXP/run_five_way.sh"
DEVICE=cuda bash "$EXP/run_five_way.sh" --continue

python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
  --config "$EXP/50_five_way.yaml" --check-only
python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
  --config "$EXP/50_five_way.yaml"
```

単一 cell の調査・再開には
`proc/seis_ssl_cluster/run_volve_horizon_five_way.py` を使う。利用可能な引数と
完了判定は CLI の `--help` と runner を正本とする。

## Checkpoint-selection variants

`50_five_way.yaml`、`51_five_way_within2.yaml`、`52_five_way_within4.yaml` は、
同じ encoder と embedding を使い、下流 decoder の checkpoint-selection 規則だけを
分離して比較する。既に生成済みの旧 run は選択 epoch の全 weight を保持しないため、
別 selection の checkpoint を履歴から後生成せず、各 YAML の独立した出力先で
decoder を実行する。

```bash
for config in 51_five_way_within2.yaml 52_five_way_within4.yaml; do
  python proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py \
    --config "$EXP/$config" --layout-config "$VOLVE_LAYOUTS" \
    --device cuda --dry-run
  python proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py \
    --config "$EXP/$config" --layout-config "$VOLVE_LAYOUTS" \
    --device cuda --continue
  python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
    --config "$EXP/$config" --check-only
  python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
    --config "$EXP/$config"
done
```

3 規則の完了済み 45-cell subset の結果と解釈は、
[`checkpoint_selection_45cell_comparison.md`](../../../../reports/volve/horizon_benchmark_v1/checkpoint_selection_45cell_comparison.md)
だけに記録する。distillation weight を変更する後続条件は
[`60_distill010/README.md`](60_distill010/README.md)を参照する。
