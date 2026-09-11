# F3 Local BT view × region search v1 — Phase A/B の記録

- 日付: 2026-09-03〜04(GPU 実行)
- 実験: [121 Local BT view × region search](../../experiments/f3/facies_benchmark_v2/121_local_bt_view_region_search_v1/README.md)
- 範囲: 8 arms の view × region を 3 epochs で比較し、
  canonical five-way v3 の `medium / layout_001` で Random と比較して screen した。
- 符号規約: delta = arm macro_f1 − Random macro_f1(正が改善)。

## Phase A 結果(Random macro_f1 = 0.496541 / eff. rank 24.00)

| arm | view | region | macro_f1 | delta | eff. rank | screen |
|---|---|---|---:|---:|---:|---|
| **`gauss010_r221`** | flip + gaussian σ0.10 | [2,2,1] | 0.484280 | **−0.012261** | 16.77 | FAIL(史上最良) |
| `tdrop02_r221` | flip + trace_drop p0.02 | [2,2,1] | 0.475134 | −0.021407 | 15.32 | FAIL |
| `sgain020_r221` | flip + smooth_gain j0.2(新) | [2,2,1] | 0.446680 | −0.049861 | 15.40 | FAIL |
| `sgain020` | flip + smooth_gain j0.2(新) | なし | 0.446149 | −0.050392 | 19.50 | FAIL |
| `zphase025_r112` | flip + zero_phase w0.25 | [1,1,2] | 0.445995 | −0.050546 | 18.10 | FAIL |
| `tokdrop030_r221` | flip + token_dropout f0.3(新) | [2,2,1] | 0.438861 | −0.057680 | 8.35 | FAIL |
| `shift2_r221` | overlap_subcrop shift[2,2,0] | [2,2,1] | 0.438668 | −0.057873 | 15.32 | FAIL |
| `tokdrop030` | flip + token_dropout f0.3(新) | なし | 0.407468 | −0.089074 | 8.80 | FAIL |

参照(再学習なし): 119 `legacy`(flip のみ)−0.036318 / rank 20.43、
120 `region_w221`(flip × region)−0.045909 / rank 16.75。

- gate: 全 arm `BT_VR_SCREEN_FAIL`(delta ≤ 0)。
  `BT_VR_PHASE_B_ELIGIBLE: ['gauss010_r221']`(delta > −0.02)で Phase B へ。
- checkpoint audit は 8 arm すべて通過した。
- 新規 view `sgain020*` の実装は `horizontal_flip_smooth_gain_v1`(AGC が除去しない横方向ゲイン j0.2)。
- `gauss010_r221` の delta −0.012261 は当時の BT 系最良値(従来 best: 119 `legacy` 3ep の −0.036318)。

### region 効果(twin 比較、region あり − region なし、macro_f1 delta)

| view | region なし arm | region あり arm | region 効果 |
|---|---|---|---:|
| flip + token_dropout f0.3 | `tokdrop030` | `tokdrop030_r221` | +0.031394 |
| flip + smooth_gain j0.2 | `sgain020` | `sgain020_r221` | +0.000531 |
| flip のみ(120) | 119 `legacy` | 120 `region_w221` | −0.009591 |

## Phase B(`gauss010_r221` 10ep resume)

| epochs | macro_f1 | delta | eff. rank |
|---:|---:|---:|---:|
| 3 | 0.484280 | −0.012261 | 16.77 |
| 10 | 0.428237 | **−0.068304** | 19.44 |

- `BT_VR_EPOCH_DELTA_NOT_IMPROVING`(勾配 **−0.008006/ep**)。
- `delta_10 ≤ 0` のため `BT_VR_PHASE_C_CANDIDATE` は出ず、Phase C へは進まない。10ep への延長は不採用。
- 学習損失は 0.4405 → 0.4007 と単調減少。
- eff. rank は 16.77 → 19.44、下流 macro_f1 delta は −0.056 悪化。

## 後続

現在の選択処方と結論は
[noise/rotation search report](local_bt_rot90_asymmetric_noise_recipe.md)、epoch の証拠は
[epoch scaling report](local_bt_epoch_scaling_v1.md)に集約する。
