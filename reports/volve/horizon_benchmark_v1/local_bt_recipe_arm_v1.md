# Volve horizon: Local Barlow Twins が Random encoder を上回った

- 日付: 2026-09-07
- 実験: `experiments/volve/horizon_benchmark_v1/32_local_bt_recipe_arm_v1`
- 主指標: `macro_mae_samples`(common test support、**小さいほど良い**)、
  5 layouts × 3 supervision sizes = 15 セルの対比較
- 結論: **単段 3 epoch + 純回転 + 非対称ガウス雑音 + region positive で、
  Local Barlow Twins が Random encoder を 15 セル中 14〜15 セルで上回った**。
  最良 arm は Random より **2.0264 samples 良く(t(14)=+7.87)**、既存の
  125 epoch Local BT からの改善は **11.0076 samples** である。凍結契約
  (decoder / tile / split / seed / embedding 抽出)は一切変更していない。

## 結果

| model | 15 セル平均 macro MAE (samples) | Random との差 | t(14) | 勝ち |
|---|---:|---:|---:|---|
| `mae`(five-way 参照) | 8.9416 | +6.6953 | — | — |
| **`rot90_asym_g080_3ep`** | **13.6104** | **+2.0264** | **+7.87** | 14/15 |
| `rot90_asym_g060_3ep` | 13.7013 | +1.9356 | +6.21 | 14/15 |
| `rot90_asym_g040_3ep` | 13.7370 | +1.8998 | **+8.31** | **15/15** |
| `random` | 15.6369 | — | — | — |
| `flip_3ep` | 17.5283 | −1.8914 | −5.83 | 2/15 |
| `local_barlow_twins`(既存 125 epoch) | 24.6180 | −8.9811 | −14.72 | 0/15 |

Random と MAE の差 6.6953 のうち、**30.3% を Local BT が埋めた**ことになる。

## 学習 recipe と view の比較

`flip_3ep` は view 契約を既存のまま(`horizontal_flip_probability: 0.5`、
region positive なし)にした単段 3 epoch の対照である。既存 recipe は、
全 encoder を batch size 16・learning rate `1.0e-4` で 100 epoch 学習した後、
top encoder block 1 個だけを batch size 4・learning rate `1.0e-5` で
25 epoch 継続学習する。`flip_3ep` は最初の学習設定で 3 epoch 実行し、
continuation を持たない。この比較には epoch 数と continuation の有無、
後半の batch size・learning rate・更新対象の違いが含まれるため、
epoch 削減だけの寄与は分離できない。

| 段階 | 15 セル平均 | 改善 |
|---|---:|---:|
| 既存 Local BT(Stage1 100ep + Stage2 25ep、flip のみ) | 24.6180 | — |
| **単段 3 epoch**(view 据え置き、continuation なし) | 17.5283 | **+7.0897** |
| **+ view を rot90 × 非対称雑音 × region に変更** | 13.6104 | **+3.9178** |
| (Random) | 15.6369 | |

- **短い単段 recipe への変更で劣位 8.9811 の 79.0% が縮まった**。
  過剰な最適化は候補となる説明だが、epoch 数と continuation の影響を
  切り分ける追加対照が必要である。
- **同じ単段 3 epoch 条件での view・region 変更は、残り 21% を埋めた上で
  Random より 2.0264 samples 良い結果になった**。`flip_3ep` は Random に
  1.8914 届かない(2/15 勝ち)ため、今回の短い学習条件では view・region を
  変更した arm で Random を上回った。

## σ 軸は plateau

σ0.40 を基準にした同一セル対比較。

| arm | σ0.40 との差 | t(14) | 勝ち |
|---|---:|---:|---|
| `rot90_asym_g060_3ep` | +0.0357 | +0.12 | 7/15 |
| `rot90_asym_g080_3ep` | +0.1266 | +0.67 | 7/15 |

**σ 0.40 / 0.60 / 0.80 は統計的に区別できない**。F3 では σ を 0.20 → 0.60 と
上げるほど単調に改善したが、Volve では 0.40 で既に飽和している。実務上は
**σ0.40 を既定にする**のが妥当で、15/15 勝ち・t=+8.31 と最も安定している。

## size 依存性は F3 と逆

| arm | small | medium | large |
|---|---:|---:|---:|
| `rot90_asym_g040_3ep` | +2.5135 (5/5) | +1.8981 (5/5) | +1.2878 (5/5) |
| `rot90_asym_g060_3ep` | +2.6573 (5/5) | +2.3391 (5/5) | +0.8102 (4/5) |
| `rot90_asym_g080_3ep` | +2.6808 (5/5) | +2.2479 (5/5) | +1.1507 (4/5) |

F3 では large ほど利得が大きかったが、**Volve では small ほど大きい**。
教師が少ないほど事前学習表現の寄与が大きいという素直な読み方ができる。
σ を上げると small / medium は伸びるが large は落ちるので、σ0.40 が
large を含めて唯一全勝する。

## 全 15 セル(Random − arm、正なら arm が良い)

`rot90_asym_g040_3ep`(15/15 勝ち):

| layout | small | medium | large |
|---|---:|---:|---:|
| layout_000 | +2.5209 | +1.1072 | +1.0903 |
| layout_001 | +3.4492 | +1.1751 | +0.3161 |
| layout_002 | +2.7483 | +1.6341 | +1.6485 |
| layout_003 | +2.5196 | +3.0626 | +1.1995 |
| layout_004 | +1.3297 | +2.5115 | +2.1847 |

`rot90_asym_g080_3ep`(14/15 勝ち、唯一の負けは large/layout_000):

| layout | small | medium | large |
|---|---:|---:|---:|
| layout_000 | +2.0665 | +2.6245 | −0.1963 |
| layout_001 | +3.6999 | +0.8935 | +0.6633 |
| layout_002 | +2.6621 | +2.1873 | +2.1681 |
| layout_003 | +2.3470 | +3.0300 | +1.7858 |
| layout_004 | +2.6283 | +2.5040 | +1.3325 |

## 処方

```yaml
augmentations:
  policy: xy_rot90_asymmetric_noise_v1   # 4 回転のみ(反射なし)+ view_b のみ雑音
  gaussian_noise_std: 0.4                # Volve では 0.4-0.8 が plateau
barlow_twins:
  method: local_barlow_twins_3d
  local_pairs_per_crop: 128
  projector_dim: 384
  redundancy_weight: 0.005
  normalization_eps: 1.0e-4
  positive_window_tokens: [2, 2, 1]
train:
  batch_size: 16
  samples_per_epoch: 10000
  epochs: 3                              # 単段。continuation を持たない
  lr: 1.0e-4
  weight_decay: 0.05
  seed: 42
```

事前学習側の指標は雑音が強いほど「最適化が進んでいない」方向に動く。

| arm | ep1 | ep2 | ep3 | diag | offdiag RMS |
|---|---:|---:|---:|---:|---:|
| `flip` | 2.5317 | 0.5746 | 0.4419 | 0.9956 | 0.0243 |
| `rot90_asym_g040` | 3.0641 | 0.7904 | 0.5662 | 0.9915 | 0.0270 |
| `rot90_asym_g060` | 3.3262 | 0.8373 | 0.6041 | 0.9900 | 0.0277 |
| `rot90_asym_g080` | 3.6238 | 0.8868 | 0.6418 | 0.9886 | 0.0283 |

下流が最良の arm ほど SSL loss が高く、不変性(diag)が低い。**SSL loss は
下流性能の指標にならない**という F3 の観察が Volve でも再現している。

## 比較契約の検証

新しい runner が既存 five-way と同じ量を測っていることを確認した。

- `model=random` の 15 セルを本 benchmark 内で実行し、
  `mae_local_bt_hmm_five_way_v1` の `model=random` と比較した結果、
  **15 セルすべてで `macro_mae_samples` が完全一致**(最大差 0.000000000000)。
  `best_epoch` と `macro_within_2_samples` も一致する。
  → `VOLVE_RECIPE_ARM_RANDOM_REPRODUCED`
- `split_plan_sha256` = `70b47b05...`、`decoder_initial_state_sha256` =
  `4ed0a3ad...` が five-way と同一。
- decoder / tiles / train block は `50_five_way.yaml` と逐語一致
  (自動テストで固定)。
- 集計は arm と Random が同一セルで同一の tile 集合・split・評価 support を
  持つことを検証したうえで対比較する。

## F3 との対応

| 観点 | F3 facies (分類, macro_f1) | Volve horizon (回帰, macro MAE) |
|---|---|---|
| 既存 BT の劣位 | −0.053 | −8.98 samples |
| 短い学習 recipe の結果 | 3ep が最適、10ep で Random 割れ | 単段 3ep で劣位の 79% を回収(epoch と continuation の寄与は未分離) |
| 純回転 + 非対称雑音 + region | +0.0476(15/15) | +1.90〜+2.03(14〜15/15) |
| σ の最適 | 0.60 まで単調改善 | 0.40〜0.80 で plateau |
| size 依存性 | large ほど有利 | small ほど有利 |
| SSL loss と下流の関係 | 予測力なし | 予測力なし(再現) |

**2 つのサーベイ・2 つのタスク種別(分類と深度回帰)・2 つの指標で、短い学習
recipe と非構造 nuisance + 純回転 + region を組み合わせた arm が Random を
上回った。** Volve で epoch 過多が原因だったかは、continuation の影響を
分離して検証する必要がある。

## 実装

- `src/seis_ssl_cluster/volve/horizon_recipe_arm.py` — 1 本の処方 arm と
  canonical random encoder を、five-way と同一の凍結下流契約で比較する。
  config に宣言した epoch / step / view / region window / 単段性は audit 時に
  checkpoint 本体と突き合わされる。
- `src/seis_ssl_cluster/volve/horizon_recipe_arm_results.py` — 対比較集計。
- `proc/seis_ssl_cluster/run_volve_horizon_recipe_arm.py`、
  `proc/seis_ssl_cluster/summarize_volve_horizon_recipe_arm.py`
- テスト 80 件(runner 35 / 集計 8 / config 37)。既存の
  `31_mae_local_bt_hmm_five_way_v1` のコード・config・artifact は無変更。

## 再現

```bash
export EXP=experiments/volve/horizon_benchmark_v1/32_local_bt_recipe_arm_v1
export LAYOUTS=experiments/volve/horizon_benchmark_v1/20_horizon_supervision/01_layouts.yaml
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/rot90_asym_g040_3ep.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/rot90_asym_g040_3ep.yaml" --device cuda
python proc/seis_ssl_cluster/run_volve_horizon_recipe_arm.py \
  --config "$EXP/30_downstream/rot90_asym_g040_3ep.yaml" \
  --model rot90_asym_g040_3ep --layout layout_000 --size small \
  --layout-config "$LAYOUTS" --device cuda
python proc/seis_ssl_cluster/summarize_volve_horizon_recipe_arm.py \
  --config "$EXP/30_downstream/rot90_asym_g040_3ep.yaml"
```

## 出力

```text
pretraining/volve/horizon_benchmark_v1/local_bt_recipe_arm_v1/<arm>/full_3ep/
embeddings/volve/horizon_benchmark_v1/local_bt_recipe_arm_v1/<arm>/overlap_x64/
horizon/volve/horizon_benchmark_v1/local_bt_recipe_arm_v1/runs/model=<arm|random>/
horizon/volve/horizon_benchmark_v1/local_bt_recipe_arm_v1/summary/<arm>/
```

## 未確定

- **epoch 1 / 2 は未測定**。Volve では 3 epoch が下限であって最適とは限らない。
  F3 では 3 が最適だったが、Volve の単段 recipe でも epoch 数を変えた比較が
  必要である。既存 100+25 epoch recipe との差 7.09 samples には
  continuation の有無も含まれるため、epoch 感度の根拠にはできない。
- region window `[2,2,1]` 以外は未評価。`flip_3ep` は window なしなので、
  window 単独の寄与は view 変更と分離できていない。
- HMM 段(`local_barlow_twins_hmm_k6`)への適用は未実施。既存 HMM 版は
  19.3122 で Random に負けているが、本処方を親にすれば改善する可能性がある。
- MAE(8.9416)には依然として 4.67 samples 及ばない。
