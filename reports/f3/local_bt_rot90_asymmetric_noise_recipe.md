# F3 Local Barlow Twins が Random encoder を全 size で上回った処方

- 日付: 2026-09-05(σ0.60 の 15 セル評価: 2026-09-06)
- 実験: [123 Local BT noise/rotation search](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/README.md)
- 結果: `rot90_asym_g060` は canonical five-way と同じ 15 セルで canonical Random 比
  平均 +0.047615(t(14)=+6.01、統計単位 = セル)、15/15 勝ち。small / medium / large
  それぞれ 5 layout 全勝。凍結契約(encoder / LR / optimizer / 下流 voxel decoder)は
  変更なし。

## 処方

- `rot90_asym_g060` = 純回転(rot90)+ 非対称ガウス雑音 σ0.60(view_a 無傷 / view_b のみ劣化)
  + region window [2,2,1]、3 epochs。
- 正典定義: [`rot90_asym_g060_3ep.yaml`](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/10_pretraining/rot90_asym_g060_3ep.yaml)
- epoch scaling の根拠(3ep 最良、σ0.40 上で測定)は
  [epoch scaling report](local_bt_epoch_scaling_v1.md)、λ・projector 幅の探索は
  [anti-collapse report](local_bt_anticollapse_search_v1_summary.md)を参照。

## 全 15 セル(canonical random 比、macro_f1 on unique validation voxels)

- 指標: macro_f1 on unique validation voxels。値 = 候補 − canonical Random(正が候補優位)。
- セル: canonical five-way の 15 セル(size small / medium / large × layout_000〜004)。

| layout | small | medium | large |
|---|---:|---:|---:|
| layout_000 | +0.013958 | +0.037009 | +0.097048 |
| layout_001 | +0.012158 | +0.012902 | +0.069926 |
| layout_002 | +0.004285 | +0.075104 | +0.068061 |
| layout_003 | +0.083717 | +0.070418 | +0.058703 |
| layout_004 | +0.053149 | +0.009050 | +0.048742 |

| size | mean | median | wins | t(4) |
|---|---:|---:|---:|---:|
| small | +0.033454 | +0.013958 | **5/5** | +2.21 |
| medium | +0.040897 | +0.037009 | **5/5** | +2.95 |
| large | +0.068496 | +0.068061 | **5/5** | +8.48 |
| **全 15 セル** | **+0.047615** | +0.053149 | **15/15** | **+6.01** |

絶対値(15 セルを size 平均した macro_f1)は small 0.5002 / medium 0.5530 /
large 0.6472 で、Random の 0.4668 / 0.5121 / 0.5787 を全 size で上回る。
canonical BT five-way(small −0.061 / medium −0.052 / large −0.047)からの改善は
**+0.094 / +0.093 / +0.115**。

## σ0.40 との対比較(同一 15 セル)

値 = `rot90_asym_g060`(σ0.60)− `rot90_asym_g040`(σ0.40)、同一セル対比較、統計単位 = セル。

| 項目 | 値 |
|---|---:|
| 15 セル平均 | +0.012306 |
| t(14) | +3.23 |
| wins | 13/15 |
| small 平均 | +0.018577 |
| medium 平均 | +0.013534 |
| large 平均 | +0.004806 |

σ0.40 で canonical Random に負けていた 4 セル(σ0.40 − Random)はすべて σ0.60 で正に転じた。

| セル | σ0.40 − Random |
|---|---:|
| small/000 | −0.0287 |
| small/002 | −0.0053 |
| medium/000 | −0.0009 |
| medium/001 | −0.0012 |

## 到達までの経路(15 セル平均)

| 処方 | small | medium | large | 15 セル |
|---|---:|---:|---:|---:|
| canonical BT(100ep, flip) | −0.061 | −0.052 | −0.047 | ≈ −0.053 |
| `gauss010_r221`(flip + 対称雑音 σ0.10 + region) | −0.0198 | +0.0089 | +0.0265 | +0.0052 (t=0.66) |
| `asym_g020`(flip + 非対称 σ0.20) | +0.0059 | +0.0176 | +0.0224 | +0.0153 (t=2.13) |
| `rot90_g010`(純回転 + 対称 σ0.10) | −0.0045 | +0.0174 | +0.0405 | +0.0178 (t=2.31) |
| `rot90_asym_g020`(合成 σ0.20) | +0.0030 | +0.0142 | +0.0489 | +0.0220 (t=2.53) |
| `rot90_asym_g040`(合成 σ0.40) | +0.0149 | +0.0274 | +0.0637 | +0.0353 (t=3.75) |
| **`rot90_asym_g060`(合成 σ0.60)** | **+0.0335** | **+0.0409** | **+0.0685** | **+0.0476 (t=6.01)** |

## 単一 cell screen 値(layout_001)

| arm | 値 |
|---|---:|
| D4(反射を含む) | −0.0288 |
| σ0.80 非対称 | −0.0238 |
| σ0.60 非対称(`rot90_asym_g060`) | +0.0129 |
| window `[2,2,2]` | +0.0013 |

## 補足事実

- 最良処方の effective rank は 13〜17(7 回確認)で、Random(24.0)を下回る。
- 単一 cell(layout_001)screen と 15 セル評価の順位は window `[2,2,2]`、`laplace_g010`、`rot90_g010`、σ0.60(screen では 5 位相当)で逆転した。
- 未実施: σ0.80 の 15 セル評価、window `[2,2,2]` の 15 セル評価、σ0.60 上での epoch 比較。
