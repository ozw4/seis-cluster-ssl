# Volve horizon MAE / Local Barlow Twins five-way v1

Volveの5ホライゾン推定を、同じembedding抽出契約、decoder、split、tile、
評価supportで比較する実験である。model順は次のとおりで固定する。

- `mae`
- `mae_hmm_k6`
- `local_barlow_twins`
- `local_barlow_twins_hmm_k6`
- `random`

Local Barlow Twinsは`local_barlow_twins_3d`、128 local pairs per crop、
水平反転のみのview contractを使う。HMMはMAEとLocal BTで別々にfitしたK=6を
使う。4 learned sourcesは100 epochのStage 1から、top encoder block 1個を
62,500 optimizer steps（25 epoch）更新する。Randomは同じencoder geometryの
seed 42 checkpointである。Stage 1、HMM target、Stage 2はホライゾンラベルを
参照しない。

下流行列は5 models × 5 layouts × 3 sizes = **75 jobs**で、各jobは50 epoch、
batch size 1、learning rate `1.0e-3`、seed 42000の既存Volve decoder契約を使う。

## 前提と環境

Python packageと必要なextrasをinstallし、公開Volve rootはread-only、artifact
rootはwritableにする。以下の変数はこのrunbook全体で使う。

```bash
cd "$(git rev-parse --show-toplevel)"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?set the read-only public Volve root}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?set the writable artifact root}"
export EXP=experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1
export CONFIG="$EXP/50_five_way.yaml"
export LAYOUTS=experiments/volve/horizon_benchmark_v1/20_horizon_supervision/01_layouts.yaml
export RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/horizon/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/runs"
export SUMMARY_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/horizon/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/summary"
```

canonical amplitude manifest、path list、input metadataを準備または確認する。

```bash
python proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py --only-missing
test -f "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/data/volve/horizon_benchmark_v1/volve_amplitude_manifest.json"
test -f "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/data/volve/horizon_benchmark_v1/volve_npy_paths.txt"
test -f "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/data/volve/horizon_benchmark_v1/volve_canonical_input_metadata.json"
```

既存のMAE Stage 1 `full_100ep/latest.pt`、Random seed-42 checkpointとembeddingも
入力である。全volume embeddingとすべての学習成果物は`artifacts/`に置き、
`reports/`をpipeline inputにしない。

## Artifact dependency graph

```text
canonical unlabeled Volve amplitude
├── existing MAE 100ep
│   ├── MAE plain 25ep ────────────────────────> mae embedding
│   └── MAE embedding -> independent HMM K6
│       └── MAE HMM 25ep ──────────────────────> mae_hmm_k6 embedding
├── Local BT 100ep
│   ├── Local BT plain 25ep ───────────────────> local_barlow_twins embedding
│   └── Local BT embedding -> independent HMM K6
│       └── Local BT HMM 25ep ─────────────────> local_barlow_twins_hmm_k6 embedding
└── existing Random seed 42 ───────────────────> reused random embedding

five checkpoint/embedding pairs
└── read-only source and common-support preflight
    └── 5 models × 5 layouts × 3 sizes
        └── 75 metrics.json -> completeness audit -> five summary files
```

## 1. Local BT Stage 1（100 epoch）

smoke configをread-onlyでvalidateしてからfull runを開始する。必要なら
`01_smoke_2step.yaml`を`--dry-run`なしで実行できるが、outputはfullと分離される。

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_stage1/local_barlow_twins/01_smoke_2step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_stage1/local_barlow_twins/02_full_100ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_stage1/local_barlow_twins/02_full_100ep.yaml"
```

完了fileは
`pretraining/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/stage1/local_bt/full_100ep/latest.pt`
である。中断時は同じconfigへ、そのoutputの`latest.pt`を`--resume`で明示する。

## 2. MAE plain continuation（25 epoch）

これは既存MAE100 checkpointをweight initializationとして使い、Stage 2の
optimizerとschedulerを新規作成する。

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$EXP/30_stage2/mae100/mae_continue/01_smoke_2step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$EXP/30_stage2/mae100/mae_continue/02_full_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config "$EXP/30_stage2/mae100/mae_continue/02_full_25ep.yaml"
```

完了fileは`stage2/mae100/mae_continue/full_25ep/latest.pt`である。

## 3. Local BT plain continuation（25 epoch）

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/30_stage2/local_bt100/local_bt_continue/01_smoke_2step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/30_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/30_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml"
```

完了fileは`stage2/local_bt100/local_bt_continue/full_25ep/latest.pt`である。
Stage 2のresume checkpointには親path、親SHA-256、再開回数を含む
`continuation_lineage`が必須である。このfield導入前のcontinuation checkpointは
監査可能な親SHAを復元できないためresumeせず、Stage 1 checkpointからStage 2を
再実行する。

## 4. MAE HMM K=6 targetとcontinuation

MAE target fitは既存MAE100 embeddingだけを使う。未作成の場合は既存のVolve
extract configで作成し、compatible metadataがあればskipする。

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/volve/horizon_benchmark_v1/30_mae_vs_random_frozen_v1/01_extract_pretrained_embeddings.yaml \
  --device cuda \
  --skip-existing
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml" \
  --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml"
bash "$EXP/20_hmm_targets/mae100/k6/03_export_pseudo_targets.sh"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/mae100/hmm/k6/01_smoke_2step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/mae100/hmm/k6/02_full_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/mae100/hmm/k6/02_full_25ep.yaml"
```

target rootは`pseudo_targets/.../mae100`、完了checkpointは
`stage2/mae100/hmm/k6/full_25ep/latest.pt`である。

## 5. Local BT HMM K=6 targetとcontinuation

Local BT targetは専用embeddingとtarget rootへ独立に生成する。

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/01_extract_embeddings.yaml" \
  --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/01_extract_embeddings.yaml" \
  --device cuda \
  --skip-existing
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml" \
  --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml"
bash "$EXP/20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/local_bt100/hmm/k6/01_smoke_2step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml"
```

target rootは`pseudo_targets/.../local_bt100`、完了checkpointは
`stage2/local_bt100/hmm/k6/full_25ep/latest.pt`である。

## 6. Final embeddings

5 configsはwindow `[128,128,128]`、overlap `[64,64,64]`、float16、
`min_token_valid_fraction: 1.0`を共有する。Randomは既存artifactが同じidentityなら
`--skip-existing`で再利用される。

```bash
for embedding_config in \
  01_mae.yaml \
  02_mae_hmm_k6.yaml \
  03_local_barlow_twins.yaml \
  04_local_barlow_twins_hmm_k6.yaml \
  05_random.yaml
do
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$EXP/40_embeddings/$embedding_config" \
    --dry-run
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$EXP/40_embeddings/$embedding_config" \
    --device cuda \
    --skip-existing
done
```

各directoryの完了契約は`volve_st10010.embeddings.npy`、
`volve_st10010.valid_tokens.npy`、`volve_st10010.embedding_metadata.json`の
整合した組である。

## 7. Source auditとembedding preflight

最初のcommandはartifactを開かない静的plan、2つ目はcheckpoint lineage、SHA、
embedding geometry、precision、canonical input、valid-token supportをまとめて
read-only監査する。PASSするまでdecoderを開始しない。

```bash
python proc/seis_ssl_cluster/audit_volve_horizon_five_way_sources.py \
  --config "$CONFIG" \
  --dry-run
python proc/seis_ssl_cluster/audit_volve_horizon_five_way_sources.py \
  --config "$CONFIG"
```

## 8. 共通cellの5-model dry-run

`layout_000/small`でsplit SHA、decoder initial-state SHA、support counts、tile countsが
5 model間で一致することを確認する。このblockはfileを書かない。

```bash
for model in \
  mae \
  mae_hmm_k6 \
  local_barlow_twins \
  local_barlow_twins_hmm_k6 \
  random
do
  python proc/seis_ssl_cluster/run_volve_horizon_five_way.py \
    --config "$CONFIG" \
    --model "$model" \
    --layout layout_000 \
    --size small \
    --layout-config "$LAYOUTS" \
    --device cuda \
    --dry-run
done
```

GPU lifecycleも確認する場合だけ、同じ5セルを1 stepで開始できる。

```bash
for model in \
  mae \
  mae_hmm_k6 \
  local_barlow_twins \
  local_barlow_twins_hmm_k6 \
  random
do
  python proc/seis_ssl_cluster/run_volve_horizon_five_way.py \
    --config "$CONFIG" \
    --model "$model" \
    --layout layout_000 \
    --size small \
    --layout-config "$LAYOUTS" \
    --device cuda \
    --max-steps 1
done
```

これはcanonical runs rootへ`latest.pt`を作る実runである。このoptional smokeを
実行した場合は、bulk launcherの明示的な`--continue` modeで各cell自身の
`latest.pt`から再開できる。

## 9. Exact-cell resumeと完了判定

中断したcellは同じmodel/layout/sizeだけを指定し、そのcell自身の`latest.pt`を
明示する。別cellのcheckpoint、異なるprecision、完了済みcellは拒否される。

```bash
export RUN_DIR="$RUNS_ROOT/model=mae/layout=layout_000/size=small"
python proc/seis_ssl_cluster/run_volve_horizon_five_way.py \
  --config "$CONFIG" \
  --model mae \
  --layout layout_000 \
  --size small \
  --layout-config "$LAYOUTS" \
  --device cuda \
  --resume "$RUN_DIR/latest.pt"
```

`latest.pt`はrolling resume state、`best.pt`はvalidation macro horizon MAEが
strictly lowerになったepoch、`metrics.json`はbest checkpointによる固定test評価で
ある。**cellの完了判定fileは`metrics.json`**であり、runnerは暗黙に上書きしない。

## 10. 75 jobs

launcherは`proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py`を1 processだけ
起動し、cell単位のSlurm用途には既存のone-job CLIを残す。

listing modeはsource artifactを開かず、Python plannerが固定順の75 unique cellを
表示する。最初にinventoryを確認する。

```bash
DRY_RUN=1 bash "$EXP/run_five_way.sh"
```

fresh runs rootで、source audit、embedding suite preflight、horizon data loadを各1回
だけ実行した後、同じprocessで75 jobsをfail-fastに逐次実行する。summaryは自動実行
しない。

```bash
DEVICE=cuda bash "$EXP/run_five_way.sh"
```

中断後は`--continue`を明示する。`metrics.json`があるcellはskipし、`latest.pt`だけが
あるcellはそのcell自身のcheckpointからresumeし、どちらもないcellはfresh実行する。
defaultはfresh-onlyで、既存cellを暗黙にskip／resumeしない。

```bash
DEVICE=cuda bash "$EXP/run_five_way.sh" --continue
```

全cellの完了数は次で確認できる。

```bash
test "$(find "$RUNS_ROOT" -type f -name metrics.json | wc -l)" -eq 75
```

## 11. Completeness auditとsummary

`--check-only`は75 metrics、source identity、split/support/decoder identity、有限値、
best checkpointを監査し、fileを書かない。`complete_jobs: 75`を確認後にsummaryを
atomicに作成する。既存summary rootは上書きしない。

```bash
python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
  --config "$CONFIG" \
  --check-only
python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
  --config "$CONFIG"
```

summaryはMAEが小さいほど良く、paired deltaは`left MAE - right MAE`（正ならrightの
誤差が小さい）としてsize別、5 layouts単位で集計する。

## Expected output tree

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/
├── pretraining/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/
│   ├── stage1/local_bt/full_100ep/latest.pt
│   └── stage2/{mae100,local_bt100}/.../full_25ep/latest.pt
├── clustering/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/
├── pseudo_targets/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/
├── embeddings/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/
│   ├── mae/overlap_x64/
│   ├── mae_hmm_k6/overlap_x64/
│   ├── local_barlow_twins/overlap_x64/
│   └── local_barlow_twins_hmm_k6/overlap_x64/
└── horizon/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_v1/
    ├── runs/model=<model>/layout=<layout>/size=<size>/
    │   ├── latest.pt
    │   ├── best.pt
    │   └── metrics.json
    └── summary/
        ├── comparison.csv
        ├── paired_deltas.csv
        ├── summary_by_size.csv
        ├── summary.json
        └── summary.md
```

Random checkpoint/embeddingは既存seed-42 rootを参照するため、この新しいembedding
subtreeには複製しない。

## Definition of Done

- 5 checkpoint/embedding pairsのread-only preflightがPASSする。
- 同じlayout/sizeの5 dry-runでsplit、support、decoder initializationが一致する。
- 固定順で75 unique cellsがあり、各cellに`metrics.json`が1つある。
- completeness auditが`complete_jobs: 75`を返す。
- summary rootに上記5 filesが1回だけ生成される。
- upstream artifactsはcanonical unlabeled amplitudeだけから作られ、ホライゾンラベルは
  frozen decoder段階だけで使われる。

## Validation within-2 checkpoint selection protocol

`51_five_way_within2.yaml`は、validation `macro_within_2_samples`がstrictly higherに
なった最初のepochを`best.pt`として選ぶ探索的protocolである。既存の
`50_five_way.yaml`と、そのMAE選択で完了済みの45件は再現性のため変更しない。この
selection変更は既存test結果を確認した後に定義したため、benchmark ID
`mae_local_bt_hmm_five_way_within2_v1`と次の専用rootへ旧v1とは別結果として保存する。

```bash
export CONFIG="$EXP/51_five_way_within2.yaml"
export RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/horizon/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_within2_v1/runs"
export SUMMARY_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/horizon/volve/horizon_benchmark_v1/mae_local_bt_hmm_five_way_within2_v1/summary"
```

旧45件の`history.json`にはepochごとの指標はあるがepochごとのweightsはなく、残って
いるweightsはMAE基準の`best.pt`とrolling `latest.pt`だけである。このためhistoryから
within-2最良epochを特定できても、そのepochの`best.pt`を後生成できず、旧45件を
再利用できない。encoder checkpointと既存embeddingは同じものを再利用し、下流
decoderだけを新rootへ再学習する。

まず`mae_hmm_k6/layout_000/small`の1セルを実行する。

```bash
python proc/seis_ssl_cluster/run_volve_horizon_five_way.py \
  --config "$EXP/51_five_way_within2.yaml" \
  --model mae_hmm_k6 \
  --layout layout_000 \
  --size small \
  --layout-config "$LAYOUTS" \
  --device cuda
```

完了後、history内の最大within-2が同値なら最初のepochを選び、metricsとbest.ptが
一致することを確認する。

```bash
export RUN_DIR="$RUNS_ROOT/model=mae_hmm_k6/layout=layout_000/size=small"
python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

import torch

run_dir = Path(sys.argv[1])
history = json.loads((run_dir / 'history.json').read_text(encoding='utf-8'))
metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
best = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
best_row = max(history, key=lambda row: row['validation_macro_within_2_samples'])
assert metrics['best_epoch'] == best_row['epoch'] == best['epoch']
assert metrics['validation']['macro_within_2_samples'] == best['best_validation_score']
print(metrics['best_epoch'], best['best_validation_score'])
PY
```

続いてMAE、MAE-HMM、Randomの45セルをplanし、同じmodel subsetを実行する。先の
1セルは`--continue`で安全にskipされる。

```bash
python proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py \
  --config "$EXP/51_five_way_within2.yaml" \
  --layout-config "$LAYOUTS" \
  --models mae mae_hmm_k6 random \
  --device cuda \
  --dry-run
python proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py \
  --config "$EXP/51_five_way_within2.yaml" \
  --layout-config "$LAYOUTS" \
  --models mae mae_hmm_k6 random \
  --device cuda \
  --continue
```

Local BT系30セルは後から同じrootへ追加できる。

```bash
python proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py \
  --config "$EXP/51_five_way_within2.yaml" \
  --layout-config "$LAYOUTS" \
  --models local_barlow_twins local_barlow_twins_hmm_k6 \
  --device cuda \
  --continue
```

正式summaryは75セル完了後だけ生成する。主比較はtest primary common supportの
within-2である。paired improvementはwithin-1/2/4で`right - left`、MAEと
adjacent-order violation rateで`left - right`とし、どのmetricも正ならright modelが
良い。summaryにはwithin-2、MAE、within-1、within-4、adjacent-order violation rate、
best epochをlayout別に含める。

```bash
test "$(find "$RUNS_ROOT" -type f -name metrics.json | wc -l)" -eq 75
python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
  --config "$EXP/51_five_way_within2.yaml" \
  --check-only
python proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py \
  --config "$EXP/51_five_way_within2.yaml"
```
