# F3 random-init HMM-K6 (distillation 0.1) control v1

- 日付: 2026-09-07
- 実験: [124 random-init HMM-K6 distillation 0.10 control](../../experiments/f3/facies_benchmark_v2/124_lithology_random_init_hmm_k6_distill010_v1/README.md)
- 問い: canonical five-way v3 で HMM 系列（`mae_hmm_k6`）が示す下流利得は、Stage 1 の
  MAE 事前学習に依存するのか、HMM pseudo-target への Stage 2 適応だけで得られるのか。
- 設計: five-way の `random` encoder（seed 42、epoch 0）を teacher 兼 student init とし、
  `mae_hmm_k6` と同じ MAE100 由来 K=6 pseudo-target、同じ固定予算
  （25 epoch / 15,625 steps、encoder top-1 block + prototype head のみ学習、LR 1e-5、
  FP32、seed 42）で HMM-K6 Stage 2 を行った。差分は初期重みと distillation weight
  （0.2 → 0.1）だけである。下流は canonical v3 の 15 cell（5 layouts × 3 sizes）を
  candidate path で評価し、canonical 5 model と layout 単位で paired 比較した。
- 結論: **random init から始めた HMM-K6 Stage 2 は、macro_f1 で `mae_hmm_k6` と
  統計的に区別できない（15 cell 平均 +0.008、t(14)=+1.02、8/15 勝）。`mae` には
  +0.033（t(14)=+3.78、12/15 勝）、`random` には +0.187（15/15 勝）で上回る。**
  F3 における HMM 系列の利得は、凍結された下位 7 block の MAE 事前学習ではなく、
  MAE100 由来 pseudo-target への top-block 適応が担っている。MAE の情報が入る経路は
  pseudo-target の系譜だけである。

## 結果: macro_f1（unique validation voxels、15 cell）

`control` = `random_init_hmm_k6_distill010`。

| size | layout | control | random | mae | mae_hmm | local_bt | local_bt_hmm |
|---|---|---:|---:|---:|---:|---:|---:|
| small | layout_000 | 0.675474 | 0.499579 | 0.636991 | 0.667558 | 0.410101 | 0.438391 |
| small | layout_001 | 0.574402 | 0.415459 | 0.547472 | 0.582467 | 0.350421 | 0.379567 |
| small | layout_002 | 0.633679 | 0.503411 | 0.601376 | 0.662714 | 0.463667 | 0.475996 |
| small | layout_003 | 0.730552 | 0.480840 | 0.637365 | 0.658629 | 0.410295 | 0.452334 |
| small | layout_004 | 0.592521 | 0.434644 | 0.559512 | 0.604916 | 0.394512 | 0.420104 |
| medium | layout_000 | 0.750102 | 0.528652 | 0.710079 | 0.723173 | 0.467389 | 0.496991 |
| medium | layout_001 | 0.651692 | 0.496541 | 0.570732 | 0.617940 | 0.404042 | 0.437049 |
| medium | layout_002 | 0.719717 | 0.523814 | 0.657987 | 0.682416 | 0.500710 | 0.532505 |
| medium | layout_003 | 0.754088 | 0.534262 | 0.698109 | 0.729625 | 0.484384 | 0.512984 |
| medium | layout_004 | 0.612976 | 0.477452 | 0.645227 | 0.653346 | 0.446140 | 0.456405 |
| large | layout_000 | 0.794889 | 0.592479 | 0.778528 | 0.799403 | 0.520609 | 0.550759 |
| large | layout_001 | 0.790802 | 0.599376 | 0.793942 | 0.799564 | 0.545623 | 0.557801 |
| large | layout_002 | 0.793037 | 0.579903 | 0.752309 | 0.750765 | 0.523798 | 0.558085 |
| large | layout_003 | 0.793356 | 0.580637 | 0.771056 | 0.771789 | 0.548114 | 0.569796 |
| large | layout_004 | 0.730786 | 0.540922 | 0.746163 | 0.767108 | 0.522249 | 0.536058 |

### size 平均

| size | control | random | mae | mae_hmm | local_bt | local_bt_hmm |
|---|---:|---:|---:|---:|---:|---:|
| small | 0.641326 | 0.466787 | 0.596543 | 0.635257 | 0.405799 | 0.433279 |
| medium | 0.697715 | 0.512144 | 0.656427 | 0.681300 | 0.460533 | 0.487187 |
| large | 0.780574 | 0.578663 | 0.768400 | 0.777726 | 0.532079 | 0.554500 |
| all 15 | 0.706538 | 0.519198 | 0.673790 | 0.698094 | 0.466137 | 0.491655 |

### 同一 cell 対比較: control − reference（macro_f1）

| reference | 15 cell 平均 | t(14) | wins/ties/losses | small | medium | large |
|---|---:|---:|---:|---:|---:|---:|
| random | +0.187340 | +21.12 | 15/0/0 | +0.174539 | +0.185571 | +0.201911 |
| mae | +0.032748 | +3.78 | 12/0/3 | +0.044782 | +0.041288 | +0.012174 |
| mae_hmm_k6 | +0.008444 | +1.02 | 8/0/7 | +0.006069 | +0.016415 | +0.002848 |
| local_barlow_twins | +0.240401 | +21.76 | 15/0/0 | +0.235526 | +0.237182 | +0.248495 |
| local_barlow_twins_hmm_k6 | +0.214883 | +22.98 | 15/0/0 | +0.208047 | +0.210528 | +0.226074 |

size 別の mean / median / sample std / 符号 count は
[`five_way/summary.md`](facies_benchmark_v2/random_init_hmm_k6_distill010_v1/five_way/summary.md)
と `summary_by_size.csv`、cell 単位の値は `comparison.csv` / `paired_deltas.csv` にある。

## 副指標（control − reference、15 cell 平均、t(14)、wins）

| reference | mean_iou | balanced_accuracy | weighted_f1 |
|---|---:|---:|---:|
| random | +0.195904 (+17.10, 15/15) | +0.174582 (+14.59, 15/15) | +0.149138 (+17.14, 15/15) |
| mae | +0.032290 (+3.46, 12/15) | +0.009974 (+0.89, 8/15) | +0.014691 (+2.06, 11/15) |
| mae_hmm_k6 | +0.002801 (+0.28, 8/15) | −0.024079 (−2.39, 5/15) | −0.006442 (−0.90, 5/15) |

mean_iou は macro_f1 と同じく `mae_hmm_k6` と同等である。balanced_accuracy だけは
control が `mae_hmm_k6` を −0.024 下回り、`mae` とは同等になる。

### class 別 F1（15 cell 平均）

| class | control | random | mae | mae_hmm_k6 |
|---|---:|---:|---:|---:|
| 0 Upper North Sea | 0.931 | 0.847 | 0.907 | 0.931 |
| 1 Middle North Sea | 0.786 | 0.586 | 0.808 | 0.824 |
| 2 Lower North Sea | 0.905 | 0.764 | 0.893 | 0.915 |
| 3 Rijnland/Chalk | 0.573 | 0.354 | 0.518 | 0.532 |
| 4 Scruff | 0.589 | 0.391 | 0.562 | 0.578 |
| 5 Zechstein | 0.455 | 0.172 | 0.355 | 0.408 |

control は少数 class（Rijnland/Chalk、Zechstein）で `mae_hmm_k6` を上回り、
多数 class（Middle / Lower North Sea）ではわずかに下回る。macro_f1 の同等性は
この配分の結果であり、balanced_accuracy の差もここに由来する。

## 解釈

- 下位 7 block を random のまま凍結し、top block と prototype head だけを MAE100 由来
  K=6 target に 25 epoch 適応させるだけで、MAE100 → HMM-K6 と同じ macro_f1 に達する。
  five-way v3 で観測した `mae_hmm_k6 − mae`（small +0.039 / medium +0.025 /
  large +0.009）は、frozen token embedding として使う限り、Stage 1 の MAE 表現では
  なく Stage 2 の target 適応で説明できる。
- ただし pseudo-target 自体は MAE100 embedding の stratigraphic HMM clustering から
  作られている。MAE の寄与は「target の系譜」として残っており、この control は
  「SSL 事前学習を encoder 重みとして持つ必要があるか」にだけ答える。
- Stage 2 の学習曲線は健全である（final loss 0.477、prototype loss 0.475、
  distillation loss 0.020、prototype usage entropy 1.589 ≒ target entropy 1.588、
  崩壊なし）。distillation は random teacher への「初期値から離れすぎない」正則化
  として働き、weight 0.1 は Parihaka / Volve の screening に合わせた値である。

## 注意点

- seed は五者と同じ 42 の 1 本のみ。layout 5 本の paired 分散は示しているが、
  初期重みや学習 seed の分散は含まない。
- distillation weight は 0.1 で、五者 HMM 系列の 0.2 と異なる。random teacher への
  distillation は意味が異なるため、weight の効果比較としては読めない。
- candidate path で評価しているため、five-way source audit の固定 model set・
  100 epoch Stage 1 系譜・distillation 0.2 の検証は意図的に適用していない。
  extraction 契約（window / overlap / float16 / amp off / min token valid fraction）と
  valid-token mask（SHA 一致）は canonical `random` と同一であることを audit した。
- Stage 2 checkpoint の `config.continuation` は random checkpoint が複製した参照 MAE
  config の写しである。初期重みの真の系譜は `stratigraphy_config` と
  `control_identity.input_identities`（下記 SHA）にある。

## 次の候補

1. pseudo-target を random encoder 自身の embedding から作る（SSL を target 系譜から
   も外した完全な control）。
2. local BT 由来 K=6 target を使う random-init 版（`local_barlow_twins_hmm_k6` 対）。
3. seed を追加し、初期重みと学習 seed の分散を評価する。
4. distillation 0.2 での再実行と、全 block unfreeze 版。

## 系譜（SHA-256）

- student init / teacher: `mae_local_bt_five_way_v1/random/random_init.pt`
  `6548d52446e7d6b9b57acd2bd39a8389a76bc5df55b52a9eda0472eb182a438c`
- pseudo-target labels: `ssl_hmm_continuation_v1/mae100/k6/f3_facies_benchmark.hmm_labels_token.npy`
  `c2ec030a3ca9e63ac06d3a8dee8285752c486435bfe7298235252810abd4ac4b`
- Stage 2 checkpoint（epoch 25 / global_step 15,625、trainable 1,774,464 /
  frozen 16,007,680 parameters）:
  `pretraining/f3/facies_benchmark_v1/random_init_hmm_k6_distill010_v1/stage2/random_init/hmm/k6/full_25ep/latest.pt`
  `bb7f9b805497a66419c35275369c316f9c09826f1a7780f8b56fdf886f1ca5af`
- embeddings（v2 manifest、overlap_x64）:
  `e368cb6ccd2020d679e6ce27825a0356dfb57197eb8e0156b364363a2d3d02b7`、
  valid tokens `3bfeb8db8a47420ae7671db90a7e4d6e5a07fceba27648ec76213df3c2b38fd7`
  （canonical `random` と一致）
- canonical checkpoints: mae `a02b34ee1edcd769…`、mae_hmm_k6 `c09e885a3d892f7e…`、
  local_barlow_twins `1c5312244f290dbf…`、local_barlow_twins_hmm_k6 `c068a555b6beff3b…`、
  random `6548d52446e7d6b9…`（全 SHA は `five_way/summary.json` の provenance）
- code: git `f8033a0554df` + experiment 124 の追加と candidate five-way summary の
  実装（`src/seis_ssl_cluster/f3/lithology/candidate_benchmark.py` の変更、
  `proc/seis_ssl_cluster/summarize_f3_lithology_candidate_five_way.py` と
  `tests/seis_ssl_cluster/test_f3_lithology_candidate_five_way_summary.py` の追加。
  実行時は uncommitted で、その後 branch `experiment/f3-random-init-hmm-k6-distill010`
  に commit した）

## 実行記録

- Stage 2: 2026-09-07 05:14–07:42 UTC、H100 NVL（共有 GPU）、1 step smoke で
  random teacher / student の受け入れと `control_identity` の記録を確認してから実行。
- embedding 抽出: 07:42–08:00 UTC。
- 15 cell: 5 layout を並列実行。最初の small round は evaluation の
  `png_label_inventory` path identity check で停止した（dataset build 時の
  repository root と異なる worktree path を `SEIS_SSL_CLUSTER_WORKSPACE` に渡した
  ため）。decoder と prediction は完了・有効だったので、root を揃えて同じ cell を
  再実行し evaluation から続行した。medium / large は 1 プロセス 16 thread に
  制限して各 round 約 27 分。
- summary: canonical `random` 比（`summarize_f3_lithology_candidate.py`）と
  five-way 5 model 比（`summarize_f3_lithology_candidate_five_way.py`、90 job の
  identity 監査後）。review file は
  [`facies_benchmark_v2/random_init_hmm_k6_distill010_v1/`](facies_benchmark_v2/random_init_hmm_k6_distill010_v1/)
  に置く。
