# F3 Local VICReg v1

F3 の未ラベル振幅から `local_vicreg_3d` を事前学習し、Random encoder との
medium screening、固定予算の VICReg continuation、VICReg100 由来 HMM-K6
continuation へ接続する versioned experiment である。Local Barlow Twins との
stage 1 比較では loss だけを変える。

## 固定する比較契約

- encoder は `AmplitudeMAE3D`、embedding dimension 384、depth 8、heads 6。
- projector は既存 Local Barlow Twins と同じ 3 層、dimension 384。
- crop は `[128, 128, 128]`、patch は `[8, 8, 8]`、local pair は 128。
- view は同一物理 token の forced-distinct horizontal flip。trace drop、D4、
  noise、filter は使わない。
- batch 16、AdamW、stage 1 LR `1.0e-4`、weight decay `0.05`、AMP off、seed 42。
- VICReg weight は invariance 25、variance 25、covariance 1、target std 1、
  epsilon `1.0e-4`。
- stage 1 baseline は 100 epochs、10,000 samples/epoch、62,500 global steps。

stage 1 の学習には、既存 MAE / Local Barlow Twins と同じ
`facies_benchmark_v1` manifest と path list を使う。v1 と v2 の prepared
amplitude は既存 parity gate で byte-identical と確認する。F3 v2 downstream
評価用の baseline embedding だけは `facies_benchmark_v2` manifest から抽出し、
次へ保存する。

```text
embeddings/f3/facies_benchmark_v2/local_vicreg_v1/base100/overlap_x64/
```

projector 出力は保存せず、bare `AmplitudeMAE3D.encode_tokens()` のみを保存する。
別 epoch の試行は別 config と別 output namespace にし、100 epoch artifact を
上書きしない。

## Artifact dependency graph

```text
pretraining/.../local_vicreg_v1/full_100ep/latest.pt
├── v2 full-volume embedding (Random screening source)
├── VICReg objective、top block 1 を +25 epochs
│   └── stage2/vicreg100/vicreg_continue/full_25ep/latest.pt
└── v1 bare-encoder embedding
    └── local-token-position residualization → PCA64 → stratigraphic HMM K=6
        └── schema-2 pseudo-target (confidence 1、boundary alpha 0)
            └── VICReg100 teacher/student、top block 1 を +25 epochs
                └── stage2/vicreg100/hmm/k6/full_25ep/latest.pt
```

VICReg control と HMM はともに 25 epochs、10,000 samples/epoch、batch 16、
15,625 global steps、encoder LR `1.0e-5`、weight decay `0.05`、AMP off、seed 42
の固定予算である。control の fresh continuation は encoder と projector の
weight だけを VICReg100 から読み、optimizer state は継承しない。同じ stage 2
run の明示的 resume だけが optimizer、RNG、dataloader generator を復元する。

## 環境

リポジトリ root で次を設定する。

```bash
set -euo pipefail
export SEIS_SSL_CLUSTER_WORKSPACE="${SEIS_SSL_CLUSTER_WORKSPACE:-/workspace}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${F3_ROOT:?export F3 data root first}"
cd "$SEIS_SSL_CLUSTER_WORKSPACE"
export VICREG_EXP=experiments/f3/facies_benchmark_v2/115_local_vicreg_v1
export VICREG_STAGE1_CHECKPOINT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/f3/facies_benchmark_v1/local_vicreg_v1/full_100ep/latest.pt"
```

## Stage 1 baseline runbook

GPU full training と downstream decoder job は実装セッションでは実行しない。
実環境では次の順を変えない。

### 1. Config tests

```bash
python -m compileall -q src proc tests
pytest -q \
  tests/seis_ssl_cluster/test_f3_local_vicreg_experiment_configs.py \
  tests/seis_ssl_cluster/test_config_strat_hmm_pretext.py \
  tests/seis_ssl_cluster/test_config_strat_hmm_pseudo_targets.py
```

### 2. 1-step dry-run

```bash
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_EXP/01_gpu_feasibility_1step.yaml" \
  --dry-run
```

### 3. 1-step live

```bash
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_EXP/01_gpu_feasibility_1step.yaml"
```

### 4. Full 100 epoch dry-run

```bash
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_EXP/02_full_100ep.yaml" \
  --dry-run
```

### 5. Full 100 epoch live

```bash
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_EXP/02_full_100ep.yaml"
```

### 6. Checkpoint and prepared-volume audit

checkpoint は 100 epochs / 62,500 steps の completed-epoch artifact でなければ
ならない。metrics は全て finite、projector state は bare encoder state と分離する。

```bash
python - <<'PY'
import math
import os

import torch

path = os.environ['VICREG_STAGE1_CHECKPOINT']
payload = torch.load(path, map_location='cpu', weights_only=False)
assert payload['epoch'] == 100
assert payload['global_step'] == 62_500
assert payload['config']['stage'] == 'vicreg_training'
assert payload['pretraining_method'] == 'local_vicreg_3d'
assert payload['checkpoint_kind'] == 'vicreg_pretraining'
assert isinstance(payload['projector_state_dict'], dict)
assert payload['projector_state_dict']
assert isinstance(payload['model_state_dict'], dict)
assert not any(
    key.startswith(('backbone.', 'projector.'))
    for key in payload['model_state_dict']
)
required_metrics = {
    'training_loss',
    'invariance_loss',
    'variance_loss',
    'covariance_loss',
    'projection_std_mean',
    'projection_std_min',
    'covariance_offdiag_rms',
}
assert required_metrics <= payload['metrics'].keys()
assert all(math.isfinite(float(payload['metrics'][key])) for key in required_metrics)
print('VICREG_STAGE1_CHECKPOINT_AUDIT_PASS')
PY

python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py --dry-run
python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py
```

parity が FAIL の場合は v1 checkpoint を v2 amplitude に適用せず停止する。

### 7. F3 v2 embedding dry-run

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$VICREG_EXP/03_extract_v2_embeddings.yaml" \
  --dry-run
```

### 8. F3 v2 embedding live

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$VICREG_EXP/03_extract_v2_embeddings.yaml"
```

抽出条件は window 128、overlap 64、float16、batch 1、AMP off、
`min_token_valid_fraction: 0.5` である。

## Random screening gate

Task 05 の medium screening は 5 layouts を paired unit とし、探索 gate は次の
3条件を全て満たす必要がある。

```text
mean(local_vicreg_100 - random) > 0
median(local_vicreg_100 - random) > 0
wins >= 3 / 5
```

screening summary は次へ保存される。

```text
f3_lithology_benchmark/local_vicreg_screen_v1/summary/summary.json
```

Task 06 の config と dry-run は gate 前でも検証できるが、次が PASS しない限り
control / HMM の full GPU run は開始しない。

```bash
export VICREG_SCREEN_SUMMARY="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_vicreg_screen_v1/summary/summary.json"
python - <<'PY'
import json
import os

with open(os.environ['VICREG_SCREEN_SUMMARY'], encoding='utf-8') as handle:
    summary = json.load(handle)
assert summary['gate_status'] == 'VICREG_BASELINE_GATE_PASS'
print(summary['gate_status'])
PY
```

`VICREG_BASELINE_GATE_FAIL` の結果も削除しない。FAIL 時はここで停止する。

## Fixed-budget VICReg continuation

### Feasibility dry-run and live

```bash
export VICREG_CONTROL="$VICREG_EXP/10_stage2/vicreg100/vicreg_continue"
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_CONTROL/01_gpu_feasibility_1step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_CONTROL/01_gpu_feasibility_1step.yaml"
```

### Full 25 epoch dry-run and live

```bash
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_CONTROL/02_full_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_CONTROL/02_full_25ep.yaml"
```

中断した control decoder ではなく VICReg training 自体を再開する場合だけ、同じ
run の completed-epoch checkpoint を明示する。

```bash
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$VICREG_CONTROL/02_full_25ep.yaml" \
  --resume "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/f3/facies_benchmark_v1/local_vicreg_v1/stage2/vicreg100/vicreg_continue/full_25ep/latest.pt"
```

## VICReg100 HMM-K6 target

HMM target は VICReg125 control ではなく、VICReg100 の bare encoder embedding
から作る。Local BT / MAE K6 と同じ local-token-position residualization、PCA64、
stratigraphic HMM K=6 を使う。

### Embedding extraction

```bash
export VICREG_HMM_TARGET="$VICREG_EXP/20_hmm_targets/vicreg100"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$VICREG_HMM_TARGET/01_extract_embeddings.yaml" \
  --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$VICREG_HMM_TARGET/01_extract_embeddings.yaml"
```

### Clustering

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$VICREG_HMM_TARGET/k6/02_cluster_hmm_k6.yaml" \
  --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$VICREG_HMM_TARGET/k6/02_cluster_hmm_k6.yaml"
```

### Pseudo-target export

export script は
`proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py` へ追加引数を渡すため、
同じ script で dry-run と live を実行できる。

```bash
bash "$VICREG_HMM_TARGET/k6/03_export_pseudo_targets.sh" --dry-run
bash "$VICREG_HMM_TARGET/k6/03_export_pseudo_targets.sh"
```

export 後に survey identity、shape、K、valid mask、schema を検証する。

```bash
python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

from seis_ssl_cluster.embedding.writer import output_paths
from seis_ssl_cluster.stratigraphy import (
    discover_pseudo_target_inputs,
    load_pseudo_target_arrays,
)

root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
embedding_dir = root / 'embeddings/f3/facies_benchmark_v1/local_vicreg_v1/hmm_targets/vicreg100/overlap_x64'
pseudo_root = root / 'pseudo_targets/f3/facies_benchmark_v1/local_vicreg_v1/vicreg100'
inputs = discover_pseudo_target_inputs(pseudo_root, k=6)
assert len(inputs) == 1
target = inputs[0]
assert target.survey_id == 'f3_facies_benchmark'
assert target.k == 6
arrays = load_pseudo_target_arrays(target)
source_valid = np.load(
    output_paths(embedding_dir, target.survey_id).valid_tokens,
    allow_pickle=False,
)
assert arrays.labels.shape == arrays.confidence.shape == arrays.valid_tokens.shape
assert np.array_equal(arrays.valid_tokens, source_valid)
with target.metadata_path.open(encoding='utf-8') as handle:
    metadata = json.load(handle)
assert metadata['survey_id'] == target.survey_id
assert metadata['k'] == 6
assert metadata['schema_version'] == 2
print('VICREG_HMM_K6_TARGET_AUDIT_PASS')
PY
```

## HMM Stage 2

teacher と student は同じ VICReg100 checkpoint から初期化する。K=6、top block 1、
prototype weight 1、usage weight 0.005、distillation weight 0.2 に固定する。

### Feasibility dry-run and live

```bash
export VICREG_HMM_STAGE2="$VICREG_EXP/30_stage2/vicreg100/hmm/k6"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$VICREG_HMM_STAGE2/01_gpu_feasibility_1step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$VICREG_HMM_STAGE2/01_gpu_feasibility_1step.yaml"
```

### Full 25 epoch dry-run and live

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$VICREG_HMM_STAGE2/02_full_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$VICREG_HMM_STAGE2/02_full_25ep.yaml"
```

HMM checkpoint は既存 encoder factory から特別な VICReg-HMM loader なしで再読込
できなければならない。base config は `vicreg_training`、追加された
`stratigraphy_config.stage` は `train_strat_hmm_pretext` である。

```bash
export VICREG_HMM_CHECKPOINT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/f3/facies_benchmark_v1/local_vicreg_v1/stage2/vicreg100/hmm/k6/full_25ep/latest.pt"
python - <<'PY'
import os

import torch

from seis_ssl_cluster.models.amplitude_encoder_factory import (
    build_model_from_checkpoint_payload,
)

payload = torch.load(
    os.environ['VICREG_HMM_CHECKPOINT'],
    map_location='cpu',
    weights_only=False,
)
assert payload['epoch'] == 25
assert payload['global_step'] == 15_625
assert payload['config']['stage'] == 'vicreg_training'
assert payload['stratigraphy_config']['stage'] == 'train_strat_hmm_pretext'
assert payload['stratigraphy_config']['teacher']['checkpoint'] == payload['stratigraphy_config']['student']['init_checkpoint']
build_model_from_checkpoint_payload(payload)
print('VICREG_HMM_CHECKPOINT_AUDIT_PASS')
PY
```

## Validation and non-overwrite policy

実装変更の確認は次で行う。

```bash
python -m compileall -q src proc tests
pytest -q tests/seis_ssl_cluster/test_f3_local_vicreg_experiment_configs.py
python -m ruff check tests/seis_ssl_cluster/test_f3_local_vicreg_experiment_configs.py
git diff --check
```

full GPU run、HMM target live generation、downstream decoder は runbook 利用者が
gate 後に実行する。既存 Local Barlow Twins config、five-way 75 jobs、既存
summary は変更・再実行しない。checkpoint、embedding、pseudo-target、decoder output
は `artifacts/` にのみ保存し、repository へ commit しない。
