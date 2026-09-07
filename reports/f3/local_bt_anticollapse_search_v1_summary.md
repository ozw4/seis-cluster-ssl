# F3 Local BT anti-collapse search v1 — Phase A の記録

- 日付: 2026-09-03(GPU 実行同日)
- 実験: [119 Local BT anti-collapse search](../../experiments/f3/facies_benchmark_v2/119_local_bt_anticollapse_search_v1/README.md)
- 範囲: collapse を抑える 5 arms を 3 epochs 学習し、canonical five-way v3
  の `medium / layout_001` で Random encoder と比較した。Phase B への事前登録
  gate は delta > −0.02。

## 結果(Random macro_f1 = 0.496541、effective rank = 24.00 / raw norm = 26.45)

| arm | 介入 | macro_f1 | delta vs Random | eff. rank | raw norm | screen |
|---|---|---:|---:|---:|---:|---|
| `legacy` | epochs: 3 のみ(無介入対照) | 0.460224 | **−0.036318** | 20.43 | 61.2 | FAIL |
| `lambda020` | redundancy_weight 0.005→0.02 | 0.428823 | −0.067718 | 23.25 | 52.1 | FAIL |
| `lambda080` | redundancy_weight 0.005→0.08 | 0.431601 | −0.064940 | 26.07 | 45.8 | FAIL |
| `proj1536_lambda020` | projector 384→1536 + λ0.02 | 0.453970 | −0.042571 | 30.15 | 48.4 | FAIL |
| `d4td02` | view: xy_d4_trace_drop_v1 (refl 0.5 / drop 0.02) | 0.455534 | −0.041007 | 15.15 | 77.2 | FAIL |

- gate 判定: 全 arm `BT_AC_SCREEN_FAIL`。`BT_AC_PHASE_B_ELIGIBLE: []`(delta > −0.02 該当なし)につき、事前登録どおり Phase B(10ep 延長・epoch–delta 勾配測定)は実施しない。
- checkpoint と candidate source の監査は全 arm で通過した。

## 主要な発見

1. **rank 崩壊は下流劣化の律速要因ではない(乖離の実証)。** anti-collapse 介入は表現レベルでは設計どおり機能した(eff. rank: legacy 20.4 → λ0.02 23.3 → λ0.08 26.1 → proj1536 30.2 と単調回復、proj1536 は Random の 24.0 を超過)。しかし下流 macro_f1 は**全介入 arm が無介入 legacy より悪い**。BT の off-diagonal 圧を強めるほど、Random init が持つ入力忠実な振幅・テクスチャ構造が早く破壊される。
2. **この単一セルでは、plain BT の既知点は 3ep 以降で負だった。** legacy 3ep
   raw = −0.036 は、既知の 1ep(+25cont)= +0.0015 と 5ep(+25cont)= −0.044
   の間を埋めた。この観測だけを他の view や複数 layout へ一般化はしない。
3. **`xy_d4_trace_drop_v1` は F3 でも不成立(Parihaka の negative prior を確認)。** 幾何 view は invariance 要求を強めるため崩壊が最も深く(rank 15.1、norm 77)、delta −0.041。
4. 5 arms とも `cross_correlation` 系 objective の学習は正常収束しており(audit 通過、loss 有限)、失敗は最適化ではなく表現の性質による。

## 結論

この実験が示すのは、固定契約下で試した 5 arms が単一の
`medium / layout_001` screen を通過せず、effective rank の回復だけでは下流改善を
説明できなかったことまでである。Local Barlow Twins 全体や invariance objective
全体の不可能性は示していない。後続の view/noise 探索でこの一般化は覆ったため、
現在の選択処方と結論は
[noise/rotation search report](local_bt_rot90_asymmetric_noise_recipe.md)を参照する。

## 当時の判断と後続

当時は reconstruction 系または短い学習を候補とした。その後の 120–123 では
view と noise の組み合わせを追加探索した。時系列上の本レポートは 119 の結果だけを
保持し、後続結果は上記の正典レポートへ集約する。
