# Volve horizon: Local Barlow Twins が Random encoder を上回った

- 日付: 2026-09-07
- 実験: `experiments/volve/horizon_benchmark_v1/32_local_bt_recipe_arm_v1`
- 主指標: `macro_mae_samples`(common test support、**小さいほど良い**)、
  5 layouts × 3 supervision sizes = 15 セルの対比較(arm vs Random、同一セル)
- 最良 arm(`rot90_asym_g080_3ep`)は Random より 2.0264 samples 良く
  (t(14)=+7.87、統計単位はセル)、既存 125 epoch Local BT からの改善は
  11.0076 samples。凍結契約(decoder / tile / split / seed / embedding 抽出)は
  未変更。

## 結果

「Random との差」は Random − arm(正なら arm が良い)。

| model | 15 セル平均 macro MAE (samples) | Random との差 | t(14) | 勝ち |
|---|---:|---:|---:|---|
| `mae`(five-way 参照) | 8.9416 | +6.6953 | — | — |
| **`rot90_asym_g080_3ep`** | **13.6104** | **+2.0264** | **+7.87** | 14/15 |
| `rot90_asym_g060_3ep` | 13.7013 | +1.9356 | +6.21 | 14/15 |
| `rot90_asym_g040_3ep` | 13.7370 | +1.8998 | **+8.31** | **15/15** |
| `random` | 15.6369 | — | — | — |
| `flip_3ep` | 17.5283 | −1.8914 | −5.83 | 2/15 |
| `local_barlow_twins_hmm_k6`(既存 five-way)\* | 19.3122 | — | — | — |
| `local_barlow_twins`(既存 125 epoch) | 24.6180 | −8.9811 | −14.72 | 0/15 |

\* 既存 five-way の HMM 継続 arm。15 セル平均のみ記録。

## 学習 recipe と view の比較

| 条件 | 学習 | view / region |
|---|---|---|
| 既存 Local BT recipe | 全 encoder を batch size 16・lr `1.0e-4` で 100 epoch、その後 top encoder block 1 個のみを batch size 4・lr `1.0e-5` で 25 epoch 継続学習 | `horizontal_flip_probability: 0.5`、region positive なし |
| `flip_3ep` | 既存 recipe の最初の学習設定で 3 epoch、continuation なし | 既存と同じ(flip のみ、region positive なし) |

注: この比較には epoch 数・continuation の有無・後半の batch size・
learning rate・更新対象の違いが含まれる。

| 段階 | 15 セル平均 | 改善 |
|---|---:|---:|
| 既存 Local BT(Stage1 100ep + Stage2 25ep、flip のみ) | 24.6180 | — |
| **単段 3 epoch**(view 据え置き、continuation なし) | 17.5283 | **+7.0897** |
| **+ view を rot90 × 非対称雑音 × region に変更** | 13.6104 | **+3.9178** |
| (Random) | 15.6369 | |

## σ 軸

σ0.40(`rot90_asym_g040_3ep`)を基準にした同一セル対比較(σ0.40 − arm、
正なら arm が良い、統計単位はセル)。

| arm | σ0.40 との差 | t(14) | 勝ち |
|---|---:|---:|---|
| `rot90_asym_g060_3ep` | +0.0357 | +0.12 | 7/15 |
| `rot90_asym_g080_3ep` | +0.1266 | +0.67 | 7/15 |

## size 依存性

size ごとの 5 layout 平均(Random − arm、正なら arm が良い、括弧は勝ち数)。

| arm | small | medium | large |
|---|---:|---:|---:|
| `rot90_asym_g040_3ep` | +2.5135 (5/5) | +1.8981 (5/5) | +1.2878 (5/5) |
| `rot90_asym_g060_3ep` | +2.6573 (5/5) | +2.3391 (5/5) | +0.8102 (4/5) |
| `rot90_asym_g080_3ep` | +2.6808 (5/5) | +2.2479 (5/5) | +1.1507 (4/5) |

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

事前学習側の指標(ep1〜ep3 は epoch ごとの SSL loss、diag / offdiag RMS は
cross-correlation 行列)。

| arm | ep1 | ep2 | ep3 | diag | offdiag RMS |
|---|---:|---:|---:|---:|---:|
| `flip` | 2.5317 | 0.5746 | 0.4419 | 0.9956 | 0.0243 |
| `rot90_asym_g040` | 3.0641 | 0.7904 | 0.5662 | 0.9915 | 0.0270 |
| `rot90_asym_g060` | 3.3262 | 0.8373 | 0.6041 | 0.9900 | 0.0277 |
| `rot90_asym_g080` | 3.6238 | 0.8868 | 0.6418 | 0.9886 | 0.0283 |

## 比較契約の検証

- `model=random` の 15 セルを本 benchmark 内で実行し、
  `mae_local_bt_hmm_five_way_v1` の `model=random` と比較: 15 セルすべてで
  `macro_mae_samples` が完全一致(最大差 0.000000000000)。`best_epoch` と
  `macro_within_2_samples` も一致。→ `VOLVE_RECIPE_ARM_RANDOM_REPRODUCED`
- `split_plan_sha256` = `70b47b05...`、`decoder_initial_state_sha256` =
  `4ed0a3ad...` が five-way と同一。
- decoder / tiles / train block は `50_five_way.yaml` と逐語一致。
  arm と Random は同一セルで同一の tile 集合・split・評価 support を持つ。

## F3 との対応

| 観点 | F3 facies (分類, macro_f1) | Volve horizon (回帰, macro MAE) |
|---|---|---|
| 既存 BT の劣位 | −0.053 | −8.98 samples |
| 短い学習 recipe の結果 | 3ep が最適、10ep で Random 割れ | 単段 3ep で 24.6180 → 17.5283(epoch と continuation の寄与は未分離) |
| 純回転 + 非対称雑音 + region | +0.0476(15/15) | +1.90〜+2.03(14〜15/15) |
| σ の最適 | 0.60 まで単調改善 | 0.40〜0.80 で plateau |
| size 依存性 | large ほど有利 | small ほど有利 |

## 未実施の条件

- epoch 1 / 2、region window `[2,2,1]` 以外、および本処方を親にした HMM
  継続(`local_barlow_twins_hmm_k6`)は未実施。
