# F3 Local BT view × region search v1 — Phase A/B の記録

- 日付: 2026-09-03〜04(GPU 実行)
- 実験: [121 Local BT view × region search](../../experiments/f3/facies_benchmark_v2/121_local_bt_view_region_search_v1/README.md)
- 範囲: 119 と 120 の失敗を受けて 8 arms の view × region を 3 epochs で比較し、
  canonical five-way v3 の `medium / layout_001` で screen した。

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

- gate: 全 arm `BT_VR_SCREEN_FAIL`(delta ≤ 0)。ただし
  `BT_VR_PHASE_B_ELIGIBLE: ['gauss010_r221']`(delta > −0.02)で Phase B へ。
- checkpoint audit は 8 arm すべて通過した。

## Phase B(`gauss010_r221` 10ep resume)

| epochs | macro_f1 | delta | eff. rank |
|---:|---:|---:|---:|
| 3 | 0.484280 | −0.012261 | 16.77 |
| 10 | 0.428237 | **−0.068304** | 19.44 |

`BT_VR_EPOCH_DELTA_NOT_IMPROVING`(勾配 **−0.008006/ep**)。
`delta_10 ≤ 0` のため `BT_VR_PHASE_C_CANDIDATE` は出ず、Phase C へは進まない。

含意: 雑音 + region の処方は **3ep 時点の損傷を大きく減らすが、損傷の蓄積
そのものは止められない**。学習損失は 0.4405 → 0.4007 と単調減少しており、
BT 目的の最適化が進むほど下流が悪化するという関係は保たれている。
rank は 16.77 → 19.44 と上昇しながら下流は −0.056 悪化しており、
**rank 非律速の 4 度目の確認**でもある。

## 主要な発見

1. **BT 系史上最良を更新**: `gauss010_r221` の delta −0.0123 は、従来 best
   (119 `legacy` 3ep の −0.0363)に対し **Random との gap を 1/3 に短縮**した。
2. **律速は「どの不変性を要求するか」**(rank ではない): `region_w221`
   (rank 16.75)と `gauss010_r221`(rank 16.77)は **effective rank がほぼ同一**
   なのに下流は +0.034 違う。119 の rank 非律速の知見が、今度は「同一 rank・
   異なる view」という直交した形で再確認された。
   - **構造的不変性**(flip / D4 / subcrop shift / token 遮蔽)= 入力から復元
     可能な幾何・内容を捨てることを要求 → 固定 voxel decoder が使う入力忠実
     構造を破壊する。token 遮蔽が最悪(−0.089、rank 8.80)。
   - **非構造的 nuisance 不変性**(加法ガウス雑音)= 信号に無関係な擾乱のみを
     無視 → 構造を保ったまま redundancy-reduction 項が特徴を整形できる。
3. **region 効果の符号は view の破壊度に依存する**(twin 比較):
   - `tokdrop030`(破壊的)→ region 化で **+0.031394 改善**(平均が遮蔽 member
     を希釈し損傷を緩和)
   - `sgain020`(無害)→ **+0.000531**(ほぼ中立)
   - flip 単独(120)→ **−0.009591 悪化**(平滑化 shortcut を開く)
4. **MSN 型遮蔽は BT 損失単体では逆効果**: 「無傷 view と 30% 遮蔽 view の一致」は
   *内容に鈍感な特徴*で最も安価に充足でき、rank 8.8 まで崩壊した。MSN が機能
   するのは prototype + エントロピー正則化がこの shortcut を塞ぐためであり、
   BT の invariance 項だけでは「遮蔽からの推論」ではなく「遮蔽への無関心」が
   学ばれる。
5. **新規 view の実装知見**: `horizontal_flip_smooth_gain_v1`(AGC が除去しない
   横方向ゲイン)は rank をほぼ保つ(19.50)が下流は −0.050 で、
   「無害だが情報も足さない nuisance」だった。

## 結論と後続

この実験は「invariance 族は全滅」という早期判断を退け、構造的不変性と
非構造 nuisance 不変性を分けて探索する根拠を与えた。Phase B は上記のとおり完了し、
10ep への延長は不採用となった。後続の 122 と 123 も完了しているため、進行中の
作業はない。現在の選択処方と結論は
[noise/rotation search report](local_bt_rot90_asymmetric_noise_recipe.md)、epoch の証拠は
[epoch scaling report](local_bt_epoch_scaling_v1.md)に集約する。
