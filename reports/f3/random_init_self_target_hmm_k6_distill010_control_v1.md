# F3 random-init self-target HMM-K6 (distillation 0.1) control v1

- 日付: 2026-09-07
- 実験: [125 random-init self-target HMM-K6 distillation 0.10 control](../../experiments/f3/facies_benchmark_v2/125_lithology_random_init_self_target_hmm_k6_distill010_v1/README.md)
- 問い: [experiment 124](random_init_hmm_k6_distill010_control_v1.md) は random init
  から始めた HMM-K6 Stage 2 が macro_f1 で `mae_hmm_k6` と同等に達することを示した
  が、その pseudo-target は MAE100 embedding の HMM clustering から作られていた。
  encoder 重みだけでなく pseudo-target の系譜からも SSL（MAE / Barlow Twins）を外し、
  未学習 random encoder 自身の embedding から作った K=6 target に Stage 2 適応した
  とき、F3 lithology の下流利得はどこまで残るのか。
- 設計: five-way の `random` encoder（seed 42、epoch 0）の v1-manifest embedding を
  canonical MAE100 target と同じ契約（window 128 / overlap 64 / float16 / amp off /
  min token valid fraction 0.5）で抽出し、同じ `stratigraphic_hmm_kmeans` 設定
  （K=6、l2、local_token_position residualization、PCA 64、seed 42）で clustering、
  同じ export 引数（k 6、confidence 1.0、boundary alpha 0.0、tau 1.0、schema 2）で
  pseudo-target を作った。Stage 2 は experiment 124 と同一で、同じ random encoder を
  teacher 兼 student init とし、固定予算（25 epoch / 15,625 steps、encoder top-1
  block + prototype head のみ学習、LR 1e-5、FP32、seed 42）、loss weights
  prototype / usage / distillation = 1.0 / 0.005 / 0.1 で学習した。experiment 124
  との差分は pseudo-target の系譜（MAE100 由来か random encoder 自身由来か）だけで
  ある。下流は canonical v3 の 15 cell（5 layouts × 3 sizes）を candidate path で
  評価し、canonical 5 model と layout 単位で paired 比較する。
- 結論: **random encoder 自身の embedding から作った K=6 target への Stage 2 適応は、
  frozen `random` を macro_f1 で +0.084（t(14)=+9.59、15/15 勝）改善するが、`mae`
  （−0.071、0/15 勝）、`mae_hmm_k6`（−0.095、0/15 勝）、そして MAE100 target を使った
  experiment 124（−0.103、0/15 勝）を全 cell で下回る。** 124 と 125 は初期重み、
  teacher、Stage 2 config、seed が同一で pseudo-target の系譜だけが異なるので、
  random init から `mae_hmm_k6` 同等に達した 124 の利得のほぼ全部（0.10 / 0.19）は
  MAE100 embedding が担う target の品質に由来する。F3 の HMM 系列の利得に SSL は
  必要であり、それは encoder 重みとしてではなく pseudo-target の情報源として効く。

## 結果: macro_f1（unique validation voxels、15 cell）

`control` = `random_init_self_target_hmm_k6_distill010`、`exp124` =
`random_init_hmm_k6_distill010`（同じ random init、MAE100 由来 target）。

| size | layout | control | exp124 | random | mae | mae_hmm | local_bt | local_bt_hmm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | layout_000 | 0.528452 | 0.675474 | 0.499579 | 0.636991 | 0.667558 | 0.410101 | 0.438391 |
| small | layout_001 | 0.470132 | 0.574402 | 0.415459 | 0.547472 | 0.582467 | 0.350421 | 0.379567 |
| small | layout_002 | 0.560218 | 0.633679 | 0.503411 | 0.601376 | 0.662714 | 0.463667 | 0.475996 |
| small | layout_003 | 0.615652 | 0.730552 | 0.480840 | 0.637365 | 0.658629 | 0.410295 | 0.452334 |
| small | layout_004 | 0.488242 | 0.592521 | 0.434644 | 0.559512 | 0.604916 | 0.394512 | 0.420104 |
| medium | layout_000 | 0.600896 | 0.750102 | 0.528652 | 0.710079 | 0.723173 | 0.467389 | 0.496991 |
| medium | layout_001 | 0.547416 | 0.651692 | 0.496541 | 0.570732 | 0.617940 | 0.404042 | 0.437049 |
| medium | layout_002 | 0.599028 | 0.719717 | 0.523814 | 0.657987 | 0.682416 | 0.500710 | 0.532505 |
| medium | layout_003 | 0.675867 | 0.754088 | 0.534262 | 0.698109 | 0.729625 | 0.484384 | 0.512984 |
| medium | layout_004 | 0.556147 | 0.612976 | 0.477452 | 0.645227 | 0.653346 | 0.446140 | 0.456405 |
| large | layout_000 | 0.694460 | 0.794889 | 0.592479 | 0.778528 | 0.799403 | 0.520609 | 0.550759 |
| large | layout_001 | 0.716899 | 0.790802 | 0.599376 | 0.793942 | 0.799564 | 0.545623 | 0.557801 |
| large | layout_002 | 0.683920 | 0.793037 | 0.579903 | 0.752309 | 0.750765 | 0.523798 | 0.558085 |
| large | layout_003 | 0.699180 | 0.793356 | 0.580637 | 0.771056 | 0.771789 | 0.548114 | 0.569796 |
| large | layout_004 | 0.610075 | 0.730786 | 0.540922 | 0.746163 | 0.767108 | 0.522249 | 0.536058 |

### size 平均

| size | control | exp124 | random | mae | mae_hmm | local_bt | local_bt_hmm |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 0.532539 | 0.641326 | 0.466787 | 0.596543 | 0.635257 | 0.405799 | 0.433279 |
| medium | 0.595871 | 0.697715 | 0.512144 | 0.656427 | 0.681300 | 0.460533 | 0.487187 |
| large | 0.680907 | 0.780574 | 0.578663 | 0.768400 | 0.777726 | 0.532079 | 0.554500 |
| all 15 | 0.603106 | 0.706538 | 0.519198 | 0.673790 | 0.698094 | 0.466137 | 0.491655 |

### 同一 cell 対比較: control − reference（macro_f1）

| reference | 15 cell 平均 | t(14) | wins/ties/losses | small | medium | large |
|---|---:|---:|---:|---:|---:|---:|
| random | +0.083908 | +9.59 | 15/0/0 | +0.065753 | +0.083727 | +0.102244 |
| mae | −0.070684 | −8.15 | 0/0/15 | −0.064004 | −0.060556 | −0.087493 |
| mae_hmm_k6 | −0.094988 | −11.61 | 0/0/15 | −0.102717 | −0.085429 | −0.096819 |
| local_barlow_twins | +0.136969 | +14.12 | 15/0/0 | +0.126740 | +0.135338 | +0.148828 |
| local_barlow_twins_hmm_k6 | +0.111451 | +12.65 | 15/0/0 | +0.099261 | +0.108684 | +0.126407 |
| exp124（MAE100 target） | −0.103432 | −15.44 | 0/0/15 | −0.108786 | −0.101844 | −0.099667 |

canonical 5 model との size 別 mean / median / sample std / 符号 count は
[`five_way/summary.md`](facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/five_way/summary.md)
と `summary_by_size.csv`、cell 単位の値は `comparison.csv` / `paired_deltas.csv` にある。
exp124 との対比較は両 experiment の `five_way/comparison.csv` の candidate 行から
layout / size を揃えて計算した。

## 副指標（control − reference、15 cell 平均、t(14)、wins）

| reference | mean_iou | balanced_accuracy | weighted_f1 |
|---|---:|---:|---:|
| random | +0.084027 (+8.21, 15/15) | +0.094141 (+9.37, 15/15) | +0.070432 (+7.57, 15/15) |
| mae | −0.079587 (−7.94, 0/15) | −0.070467 (−6.30, 1/15) | −0.064016 (−7.63, 1/15) |
| mae_hmm_k6 | −0.109076 (−12.17, 0/15) | −0.104520 (−10.49, 0/15) | −0.085148 (−10.17, 0/15) |
| exp124（MAE100 target） | −0.111877 (−16.61, 0/15) | −0.080441 (−8.81, 0/15) | −0.078707 (−13.54, 0/15) |

4 指標とも符号と順位は macro_f1 と同じで、control は `random` より一様に高く、
`mae` / `mae_hmm_k6` / exp124 より一様に低い。

### class 別 F1（15 cell 平均）

| class | control | exp124 | random | mae | mae_hmm_k6 |
|---|---:|---:|---:|---:|---:|
| 0 Upper North Sea | 0.870 | 0.931 | 0.847 | 0.907 | 0.931 |
| 1 Middle North Sea | 0.697 | 0.786 | 0.586 | 0.808 | 0.824 |
| 2 Lower North Sea | 0.833 | 0.905 | 0.764 | 0.893 | 0.915 |
| 3 Rijnland/Chalk | 0.440 | 0.573 | 0.354 | 0.518 | 0.532 |
| 4 Scruff | 0.493 | 0.589 | 0.391 | 0.562 | 0.578 |
| 5 Zechstein | 0.285 | 0.455 | 0.172 | 0.355 | 0.408 |

`random` に対する改善は全 class で正（+0.023 から +0.113）で、Middle North Sea、
Scruff、Zechstein で大きい。exp124 に対しては全 class で負で、少数 class
（Rijnland/Chalk −0.133、Zechstein −0.170）ほど差が開く。124 が `mae_hmm_k6` を
上回っていた少数 class の利得は、MAE100 由来 target があってはじめて現れる。

## 解釈

- **target 系譜が利得のほぼ全部を決める。** 124 と 125 の差分は pseudo-target の
  系譜だけ（同じ `random_init.pt` を init と teacher に、同じ Stage 2 config、同じ
  seed 42）で、macro_f1 の差は −0.103（0/15 勝、size 間でほぼ一定）である。
  `random` → `mae_hmm_k6` の距離 0.179 のうち、random 自身由来 target で埋まるのは
  0.084、残り 0.095 は target を MAE100 embedding から作ることで埋まる。124 が
  「random init でも `mae_hmm_k6` に届く」と示した利得は、MAE100 表現を target に
  蒸留した結果であって、HMM pretext の形式そのものの効果ではない。
- **SSL 抜きの Stage 2 適応にも一定の効果はある。** frozen `random` に対して +0.084
  （15/15 勝）、`local_barlow_twins` 系列には +0.11 から +0.14 で勝つ。未学習
  encoder の embedding でも stratigraphic HMM は depth 順の 6 state を返し、top block と
  prototype head をそこへ適応させるだけで class 別 F1 は全 class で上がる。ただし
  frozen `mae`（Stage 1 のみ）には −0.071 で届かず、「SSL 事前学習は不要」とは
  言えない。
- **F3 五者の読みへの含意。** `mae_hmm_k6 − mae`（+0.024）と 124 の結果だけを見ると
  HMM Stage 2 が MAE 重みを不要にするように見えたが、完全 control では逆に、MAE100
  embedding が target の情報源として必須である。HMM 系列の設計上の要点は「良い
  embedding を stratigraphic 事前分布付きで離散化し、その label に適応する」ことで、
  embedding の質（ここでは MAE100）が上限を決める。
- **Stage 2 の学習は健全。** final loss 0.670（prototype 0.668、distillation 0.017）、
  prototype usage entropy 1.5505 ≒ target usage entropy 1.5506 で崩壊なし。124
  （prototype loss 0.475）より prototype loss が高いのは、random 自身由来 target が
  random encoder の top-block 表現から予測しにくい label であることを示す。

## 注意点

- seed は五者と同じ 42 の 1 本のみ。layout 5 本の paired 分散は示すが、初期重みや
  学習 seed、target clustering seed の分散は含まない。
- distillation weight は 0.1 で、五者 HMM 系列の 0.2 と異なる。random teacher への
  distillation は意味が異なるため、weight の効果比較としては読めない。
  `student.unfreeze_top_blocks > 0` のとき resolver は正の distillation weight を
  要求するので、0.0 にはできない。
- pseudo-target は未学習 encoder の embedding から作っているため、target の構造は
  学習された表現ではなく、random projection 越しの振幅統計と stratigraphic HMM の
  depth 事前分布（`order_by: mean_z`、shallow / deep anchor、reverse 遷移禁止）が
  主に決める。実際の target は depth 順の 6 state（token 単位の mean z が
  5.9 / 7.9 / 13.7 / 19.3 / 21.7 / 24.9）で、label entropy は 1.639（MAE100 target は
  1.734）、state 別 token 数は 12,214–52,071 と MAE100 target より偏る。MAE100
  target との一致率は最良置換で 43.1%（NMI 0.281、ARI 0.203）にとどまり、target の
  中身は MAE100 由来のものとは大きく異なる。
- candidate path で評価しているため、five-way source audit の固定 model set・
  100 epoch Stage 1 系譜・distillation 0.2 の検証は意図的に適用していない。
  extraction 契約（window / overlap / float16 / amp off / min token valid fraction）と
  valid-token mask（SHA 一致）が canonical `random` と同一であることだけを audit する。
- random checkpoint の `config` は参照 MAE checkpoint の resolved config の複製
  （geometry 参照用。重みは random、`metadata.random_encoder_baseline: true`、
  `metadata.pretrained_weights_loaded: false`）なので、target 抽出 embedding の
  metadata と Stage 2 checkpoint の `config.continuation` には MAE 系譜が写る。
  下流 v2 embedding metadata の `stratigraphy_pretext.base_objective`（`amp_mae3d`）
  と `pretraining_objective` も同じ複製 config の `stage: train_amp_mae` から導出
  されるため、control の系譜を表さない（experiment 124 の metadata でも
  `base_objective: amp_mae3d` になっている。candidate source audit の identity key
  には含まれない）。真の系譜は `stratigraphy_config`、`identity.scientific_identity`
  （`experiment_role: random_init_self_target_control`）、
  `control_identity.input_identities`（下記 SHA）、embedding metadata では
  `stratigraphy_pretext.pseudo_target_input_dir` と `model_tag` にある。
- experiment 124 とは同じ seed、同じ Stage 2 config で target だけが異なるが、
  target の変更は loss の landscape を変えるので、−0.103 を target 系譜の純粋な効果
  として読むには seed の追加が必要である。ただし 15 cell すべてで同符号かつ size 間で
  ほぼ一定（−0.100 から −0.109）であり、符号が変わる余地は小さい。

## 次の候補

1. seed を追加し、初期重みと学習 seed、target clustering seed の分散を評価する。
2. local BT 由来 K=6 target を使う random-init 版（`local_barlow_twins_hmm_k6` 対）。
3. distillation 0.2 での再実行と、全 block unfreeze 版。
4. random 自身由来 target と MAE100 由来 target の agreement / entropy の比較
   （target 品質と下流利得の関係）。

## 系譜（SHA-256）

- student init / teacher / target source encoder:
  `mae_local_bt_five_way_v1/random/random_init.pt`
  `6548d52446e7d6b9b57acd2bd39a8389a76bc5df55b52a9eda0472eb182a438c`
- target embeddings（v1 manifest、overlap_x64）:
  `embeddings/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/hmm_targets/random_init/overlap_x64`
  `a58d1fdb22c8b38f35484f6a9a5e873df12547b00d0facc0d93881c84ae62230`、
  valid tokens `3bfeb8db8a47420ae7671db90a7e4d6e5a07fceba27648ec76213df3c2b38fd7`
  （canonical MAE100 target 抽出と byte 一致）。embedding metadata の
  `checkpoint_sha256` は上記 random checkpoint の SHA と一致
- clustering: `clustering/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/hmm_targets/random_init/k6`
- pseudo-target labels:
  `random_init_self_target_hmm_k6_distill010_v1/random_init/k6/f3_facies_benchmark.hmm_labels_token.npy`
  `5eb6b067487c465b4013f1fa65ffb0edb8010c7f24b289d9cffd9f8e9d84da9c`
  （clustering の `cluster_labels_token.npy` と同一 SHA。valid 162,960 token、
  label counts 32,330 / 12,214 / 52,071 / 15,215 / 12,413 / 38,717）
- Stage 2 checkpoint（epoch 25 / global_step 15,625、trainable 1,774,464 /
  frozen 16,007,680 parameters）:
  `pretraining/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/stage2/random_init/hmm/k6/full_25ep/latest.pt`
  `b2e09065f3f4fe65d660a4d93023bee4bcdbc949d7a013b86794dfba53c5136f`
- embeddings（v2 manifest、overlap_x64）:
  `embeddings/f3/facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/random_init_self_target_hmm_k6_distill010/overlap_x64`
  `ac3871235abb19778473c1f3c6a212001fd0e03a5aa50cac527625a5a4fe195a`、
  valid tokens `3bfeb8db8a47420ae7671db90a7e4d6e5a07fceba27648ec76213df3c2b38fd7`
  （canonical `random` と一致）
- canonical checkpoints: mae `a02b34ee1edcd769…`、mae_hmm_k6 `c09e885a3d892f7e…`、
  local_barlow_twins `1c5312244f290dbf…`、local_barlow_twins_hmm_k6 `c068a555b6beff3b…`、
  random `6548d52446e7d6b9…`（全 SHA は `five_way/summary.json` の provenance）
- code: git `f8033a0554df` + experiment 124 / 125 の追加と candidate five-way
  summary の実装（`src/seis_ssl_cluster/f3/lithology/candidate_benchmark.py` の変更、
  `proc/seis_ssl_cluster/summarize_f3_lithology_candidate_five_way.py` と
  `tests/seis_ssl_cluster/test_f3_lithology_candidate_five_way_summary.py` の追加。
  実行時は uncommitted で、その後 branch `experiment/f3-random-init-hmm-k6-distill010`
  に commit した）。
  Stage 2 の `control_identity.runtime_identity` は `git_commit`
  `f8033a0554df1b85c4f1914f5ca834777a69d344` と `git_diff_sha256`
  `46c6dc860f5330da4121459bcbb908fce09d5d4cbc883c09f5a83dc515400d20`
  （tracked file の `git diff --binary HEAD`。untracked file は含まない）を記録し、
  smoke と full の `run_metadata.json` で同一である。

## 実行記録

- pipeline は `/workspace/logs/f3_random_init_self_target_hmm_k6_distill010_v1/driver.sh`
  が target 抽出 → clustering → export → Stage 2（smoke、full）→ v2 embedding 抽出
  → candidate preflight → 15 cell（size ごとに 5 layout 並列、1 プロセス 16 thread）
  → summary の順で実行する。`--dry-run` を持つ stage（target 抽出、clustering、
  export、Stage 2、embedding 抽出、five-way summary）は先に `--dry-run` を通し、
  cell は layout_000 / small の preflight で source audit を通す。
- target 抽出: 2026-09-07 13:48–13:56 UTC（H100 NVL、共有 GPU）。
- clustering: 13:56–13:57 UTC（26 秒）。export: 13:57 UTC。
- Stage 2: 1 step smoke（13:57 UTC）で random teacher / student の受け入れと
  `control_identity` の記録を確認してから、full を 13:57–16:01 UTC（2 時間 04 分）
  で実行。final loss 0.670（prototype 0.668、distillation 0.017、usage 0.0）、
  prototype usage entropy 1.5505 ≒ target usage entropy 1.5506（崩壊なし）、
  valid supervised token fraction 0.865。
- embedding 抽出（v2 manifest）: 16:01–16:09 UTC。
- 15 cell: 16:09–17:28 UTC（small 16:09–16:35、medium 16:35–17:01、large
  17:01–17:28。size ごとに 5 layout 並列、1 プロセス 16 thread、失敗 0）。
  `SEIS_SSL_CLUSTER_WORKSPACE` は v3 dataset を build した repository root
  （`/workspace`）を与えた。preflight（layout_000 / small）の candidate source audit
  で抽出契約と valid-token mask が canonical `random` と一致することを確認した。
- summary: 17:28 UTC。canonical `random` 比（`summarize_f3_lithology_candidate.py`）と
  five-way 5 model 比（`summarize_f3_lithology_candidate_five_way.py`、90 job の
  identity 監査後）。review file は
  `reports/f3/facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/`
  （[`five_way/summary.md`](facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/five_way/summary.md)、
  [`random/summary.md`](facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/random/summary.md)）
  に置く。
