# F3 paired SSL / HMM-K6 execution runbook

F3 full-volumeを用いてMAE100とBT100を学習し、同じStage 1 sourceから25 epoch
controlとpaired HMM-K6 target / HMM25を作る。固定学習予算の入力と比較には
`latest.pt`を使用し、`best.pt`、NOPIMS、MAE25 / BT25をHMM sourceにしない。

## 1. 環境変数

リポジトリrootでartifact root、suite、GPUを固定する。

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export SUITE=experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export STAGE1_CONFIGS="$SUITE/10_stage1"
export TARGET_CONFIGS="$SUITE/20_hmm_targets"
export STAGE2_CONFIGS="$SUITE/30_stage2"
export CUDA_VISIBLE_DEVICES=1

export MANIFEST="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/registry/manifests/f3/facies_benchmark_v1/f3_amplitude_manifest.json"
export PATH_LIST="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/registry/splits/f3/facies_benchmark_v1/f3_npy_paths.txt"
export NORMALIZATION_STATS="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/registry/normalization_stats/f3/facies_benchmark_v1/f3_seismic.normalization_stats.json"
export ARTIFACT_SUITE="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1"

export MAE100="$ARTIFACT_SUITE/stage1/mae/full_100ep/latest.pt"
export BT100="$ARTIFACT_SUITE/stage1/barlow_twins/full_100ep/latest.pt"
export MAE_CONTROL="$ARTIFACT_SUITE/stage2/mae100/mae_continue/full_25ep/latest.pt"
export BT_CONTROL="$ARTIFACT_SUITE/stage2/bt100/bt_continue/full_25ep/latest.pt"
export MAE_HMM25="$ARTIFACT_SUITE/stage2/mae100/hmm/k6/full_25ep/latest.pt"
export BT_HMM25="$ARTIFACT_SUITE/stage2/bt100/hmm/k6/full_25ep/latest.pt"
export MAE_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/mae100"
export BT_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/bt100"
```

## 2. prepared F3 artifact確認

manifest、full-volume path list、normalization statsが揃っていることを先に確認する。

```bash
for artifact in "$MANIFEST" "$PATH_LIST" "$NORMALIZATION_STATS"
do
  test -f "$artifact" || { echo "missing prepared F3 artifact: $artifact" >&2; exit 1; }
done
```

不足している場合は既存prepare stageを確認して実行する。新しいprepare処理は作らない。

```bash
python proc/seis_ssl_cluster/prepare_f3_facies_volume.py \
  --config experiments/f3/facies_benchmark_v1/10_prepare/01_prepare_f3_volume.yaml \
  --dry-run
python proc/seis_ssl_cluster/prepare_f3_facies_volume.py \
  --config experiments/f3/facies_benchmark_v1/10_prepare/01_prepare_f3_volume.yaml
```

## 3. targeted tests

suite固有の5 testと関連generic regressionを一度ずつ実行する。これらはlive
artifactやGPUを必須にしない。

```bash
pytest -q \
  tests/seis_ssl_cluster/test_f3_stage1_ssl_configs.py \
  tests/seis_ssl_cluster/test_f3_ssl_continuation_configs.py \
  tests/seis_ssl_cluster/test_f3_hmm_k6_target_configs.py \
  tests/seis_ssl_cluster/test_f3_hmm_k6_configs.py \
  tests/seis_ssl_cluster/test_f3_ssl_hmm_runbook.py \
  tests/seis_ssl_cluster/test_mae_continuation_runner.py \
  tests/seis_ssl_cluster/test_barlow_twins_continuation.py \
  tests/seis_ssl_cluster/test_barlow_twins_training_contract.py \
  tests/seis_ssl_cluster/test_embedding_extractor.py \
  tests/seis_ssl_cluster/test_strat_checkpoint_extraction.py \
  tests/seis_ssl_cluster/test_strat_hmm_pretraining_head_only.py \
  tests/seis_ssl_cluster/test_strat_hmm_barlow_runner_integration.py
```

## 4. Stage 1 MAE100 / BT100

4設定をdry-runし、F3 manifest、FP32、batch 16、100 epoch条件を確認する。

```bash
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1_CONFIGS/mae/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1_CONFIGS/mae/02_full_100ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1_CONFIGS/barlow_twins/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1_CONFIGS/barlow_twins/02_full_100ep.yaml" --dry-run
```

1-step feasibilityがOOMなし、finite loss / gradient、`latest.pt`保存を満たして
からfull runへ進む。

```bash
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1_CONFIGS/mae/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1_CONFIGS/barlow_twins/01_gpu_feasibility_1step.yaml"

# fresh runでは--resumeを付けない
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1_CONFIGS/mae/02_full_100ep.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1_CONFIGS/barlow_twins/02_full_100ep.yaml"
```

中断時だけ、同じStage 1 run自身の`latest.pt`からresumeする。

```bash
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1_CONFIGS/mae/02_full_100ep.yaml" --resume "$MAE100"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1_CONFIGS/barlow_twins/02_full_100ep.yaml" --resume "$BT100"
```

## 5. Stage 1 live監査

次の1本でMAE100 / BT100の固定予算、F3 input、FP32、BT projectorを監査する。

```bash
python - "$MANIFEST" "$PATH_LIST" "$MAE100" "$BT100" <<'PY'
import math
from pathlib import Path
import sys

from seis_ssl_cluster.training import load_checkpoint


manifest_path = Path(sys.argv[1]).resolve()
path_list = Path(sys.argv[2]).resolve()
pairs = (
    ('mae', Path(sys.argv[3]), 'train_amp_mae'),
    ('bt', Path(sys.argv[4]), 'barlow_twins_training'),
)
for method, checkpoint_path, expected_stage in pairs:
    assert checkpoint_path.is_file() and checkpoint_path.name == 'latest.pt'
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    config = payload['config']
    assert payload['epoch'] == 100
    assert payload['global_step'] == 62_500
    assert payload['amp_enabled'] is False
    assert config['stage'] == expected_stage
    assert config['train']['amp'] is False
    assert config['train']['batch_size'] == 16
    assert config['train']['samples_per_epoch'] == 10_000
    assert Path(config['manifests']['train']).resolve() == manifest_path
    assert Path(config['manifests']['train_path_list']).resolve() == path_list
    assert all(math.isfinite(float(value)) for value in payload['metrics'].values())
    if method == 'bt':
        assert payload['pretraining_method'] == 'barlow_twins_3d'
        assert payload['checkpoint_kind'] == 'barlow_twins_pretraining'
        assert payload['projector_state_dict']
    print(f'Stage 1 PASS: {method} {checkpoint_path}')
PY
```

## 6. Stage 2 controls

MAE100 → MAE25とBT100 → BT25はStage 1 `latest.pt`をweights-only初期値に
使う。Stage 1 checkpointを`--resume`へ渡さない。

```bash
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE2_CONFIGS/mae100/mae_continue/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE2_CONFIGS/mae100/mae_continue/02_full_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE2_CONFIGS/bt100/bt_continue/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE2_CONFIGS/bt100/bt_continue/02_full_25ep.yaml" --dry-run

python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE2_CONFIGS/mae100/mae_continue/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE2_CONFIGS/bt100/bt_continue/01_gpu_feasibility_1step.yaml"

# fresh optimizer / counter / RNGで開始し、--resumeを付けない
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE2_CONFIGS/mae100/mae_continue/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE2_CONFIGS/bt100/bt_continue/02_full_25ep.yaml"
```

中断時だけ、対応するStage 2 control自身の`latest.pt`からresumeする。

```bash
python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE2_CONFIGS/mae100/mae_continue/02_full_25ep.yaml" --resume "$MAE_CONTROL"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE2_CONFIGS/bt100/bt_continue/02_full_25ep.yaml" --resume "$BT_CONTROL"
```

## 7. paired K=6 target pipeline

embeddingはKに依存しないため`mae100/`と`bt100/`直下のconfigを使う。MAEと
BTを独立にextract、anchor-only K=6 clustering、exportの順で実行する。

```bash
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/mae100/01_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/mae100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/bt100/01_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/bt100/01_extract_embeddings.yaml"

python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/mae100/k6/02_cluster_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/mae100/k6/02_cluster_hmm_k6.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/bt100/k6/02_cluster_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/bt100/k6/02_cluster_hmm_k6.yaml"

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/mae100/k6" \
  --pseudo-target-root "$MAE_TARGET_ROOT" --k 6 --confidence 1.0 \
  --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 --dry-run
bash "$TARGET_CONFIGS/mae100/k6/03_export_pseudo_targets.sh"

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/bt100/k6" \
  --pseudo-target-root "$BT_TARGET_ROOT" --k 6 --confidence 1.0 \
  --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 --dry-run
bash "$TARGET_CONFIGS/bt100/k6/03_export_pseudo_targets.sh"
```

`--pseudo-target-root`には`k6`を付けない。公開APIが実artifactを
`mae100/k6`と`bt100/k6`へ作成する。

## 8. target live監査

公開pseudo-target APIを使い、survey、shape、valid mask、K=6 occupancy、confidence、
alpha 0のboundary weightを1本で監査する。境界数は診断値として表示し、空state
以外のoccupancy閾値は設けない。

```bash
python - "$MANIFEST" "$MAE_TARGET_ROOT" "$BT_TARGET_ROOT" <<'PY'
from pathlib import Path
import sys

import numpy as np

from seis_ssl_cluster.data import read_manifest_json
from seis_ssl_cluster.stratigraphy import (
    discover_pseudo_target_inputs,
    load_pseudo_target_arrays,
)


manifest_ids = {
    item.survey_id for item in read_manifest_json(Path(sys.argv[1]))
}
roots = {'mae': Path(sys.argv[2]).resolve(), 'bt': Path(sys.argv[3]).resolve()}
assert roots['mae'] != roots['bt']
shapes = {}
valid_masks = {}
for method, root in roots.items():
    inputs = discover_pseudo_target_inputs(root, k=6)
    assert {item.survey_id for item in inputs} == manifest_ids
    occupancy = np.zeros(6, dtype=np.int64)
    boundary_count = 0
    shapes[method] = {}
    valid_masks[method] = {}
    for item in inputs:
        arrays = load_pseudo_target_arrays(item, mmap_mode='r')
        labels = np.asarray(arrays.labels)
        confidence = np.asarray(arrays.confidence)
        valid = np.asarray(arrays.valid_tokens)
        boundary_weight = np.asarray(arrays.boundary_weight)
        assert labels.shape == confidence.shape == valid.shape == boundary_weight.shape
        assert np.all(np.isfinite(confidence))
        assert np.all(np.isfinite(boundary_weight))
        assert np.all((labels[valid] >= 0) & (labels[valid] <= 5))
        assert np.all(labels[~valid] == -1)
        assert np.all((confidence >= 0.0) & (confidence <= 1.0))
        assert np.all(confidence[~valid] == 0.0)
        assert np.all(boundary_weight[valid] == 1.0)
        assert np.all(boundary_weight[~valid] == 0.0)
        occupancy += np.bincount(labels[valid], minlength=6)[:6]
        adjacent = valid[..., 1:] & valid[..., :-1]
        boundary_count += int(np.count_nonzero(
            adjacent & (labels[..., 1:] != labels[..., :-1])
        ))
        shapes[method][item.survey_id] = labels.shape
        valid_masks[method][item.survey_id] = valid.copy()
    assert np.all(occupancy > 0), occupancy
    print(f'{method}: occupancy={occupancy.tolist()} z_boundaries={boundary_count}')

assert shapes['mae'].keys() == shapes['bt'].keys() == manifest_ids
for survey_id in manifest_ids:
    assert shapes['mae'][survey_id] == shapes['bt'][survey_id]
    np.testing.assert_array_equal(
        valid_masks['mae'][survey_id], valid_masks['bt'][survey_id]
    )
PY
```

## 9. HMM-K6 25 epoch

4設定をdry-runし、K=6、paired target、Stage 1 teacher / student、top-1、loss、
head / encoder LR、予算を確認する。

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml" --dry-run

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml"

# fresh full run: Stage 1 bindingを使い、--resumeを付けない
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml"
```

MAE25 / BT25 controlをteacher、student、resumeのどこにも使用しない。中断時だけ
同じHMM25 run自身の`latest.pt`からresumeする。

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml" --resume "$MAE_HMM25"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$STAGE2_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml" --resume "$BT_HMM25"
```

## 10. final checkpoint監査

control 2本とHMM25 2本を1本で監査する。HMMではpaired binding、K=6 head、
optimizer、frozen / trainable encoder範囲、bare encoder consumer、base stageも
確認する。head初期値は再構築しない。

```bash
python - \
  "$MAE100" "$MAE_CONTROL" "$MAE_TARGET_ROOT" "$MAE_HMM25" \
  "$BT100" "$BT_CONTROL" "$BT_TARGET_ROOT" "$BT_HMM25" <<'PY'
import math
from pathlib import Path
import sys

import torch

from seis_ssl_cluster.models.amplitude_encoder_factory import (
    build_model_from_checkpoint_payload,
)
from seis_ssl_cluster.training import load_checkpoint


def assert_finite_metrics(payload):
    metrics = payload['metrics']
    assert metrics
    assert all(math.isfinite(float(value)) for value in metrics.values())


def audit_control(source_path, checkpoint_path, expected_stage):
    source_path = source_path.resolve()
    assert checkpoint_path.is_file() and checkpoint_path.name == 'latest.pt'
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    config = payload['config']
    assert payload['epoch'] == 25
    assert payload['global_step'] == 15_625
    assert payload['amp_enabled'] is False
    assert config['stage'] == expected_stage
    assert config['train']['amp'] is False
    assert config['train']['batch_size'] == 16
    assert config['train']['samples_per_epoch'] == 10_000
    assert Path(config['continuation']['init_checkpoint']).resolve() == source_path
    assert config['continuation']['unfreeze_top_blocks'] == 1
    assert_finite_metrics(payload)
    print(f'control PASS: {checkpoint_path}')


def audit_hmm(
    source_path,
    target_root,
    checkpoint_path,
    expected_stage,
):
    source_path = source_path.resolve()
    target_root = target_root.resolve()
    assert source_path.is_file() and source_path.name == 'latest.pt'
    assert checkpoint_path.is_file() and checkpoint_path.name == 'latest.pt'
    source = load_checkpoint(source_path, map_location='cpu')
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    config = payload['config']
    stratigraphy = payload['stratigraphy_config']
    assert payload['epoch'] == 25
    assert payload['global_step'] == 15_625
    assert payload['amp_enabled'] is False
    assert payload['training_state']['stage'] == 'train_strat_hmm_pretext'
    assert config['stage'] == expected_stage
    assert stratigraphy['train']['amp'] is False
    assert stratigraphy['pseudo_targets']['k'] == 6
    assert stratigraphy['head']['num_prototypes'] == 6
    assert stratigraphy['student']['unfreeze_top_blocks'] == 1
    assert stratigraphy['loss'] == {
        'prototype_weight': 1.0,
        'usage_weight': 0.005,
        'entropy_floor': None,
        'distillation_weight': 0.2,
    }
    assert Path(stratigraphy['teacher']['checkpoint']).resolve() == source_path
    assert Path(stratigraphy['student']['init_checkpoint']).resolve() == source_path
    assert Path(stratigraphy['pseudo_targets']['input_dir']).resolve() == target_root
    assert_finite_metrics(payload)

    source_state = source['model_state_dict']
    student_state = payload['model_state_dict']
    assert source_state.keys() == student_state.keys()
    patch_keys = [key for key in source_state if key.startswith('patch_projection.')]
    lower_keys = [
        key
        for key in source_state
        if any(key.startswith(f'encoder.layers.{index}.') for index in range(7))
    ]
    decoder_keys = [
        key
        for key in source_state
        if (
            key == 'mask_token'
            or key.startswith('encoder_to_decoder.')
            or key.startswith('decoder.')
            or key.startswith('prediction_head.')
        )
    ]
    top_keys = [
        key for key in source_state if key.startswith('encoder.layers.7.')
    ]
    assert patch_keys and lower_keys and top_keys
    assert decoder_keys
    assert all(torch.equal(source_state[key], student_state[key]) for key in patch_keys)
    assert all(torch.equal(source_state[key], student_state[key]) for key in lower_keys)
    assert all(
        torch.equal(source_state[key], student_state[key])
        for key in decoder_keys
    )
    assert any(not torch.equal(source_state[key], student_state[key]) for key in top_keys)

    optimizer = payload['optimizer_state_dict']
    groups = {group['name']: group for group in optimizer['param_groups']}
    assert set(groups) == {'head', 'encoder'}
    for group in groups.values():
        assert group['lr'] == 1.0e-5
        assert group['params']
        for parameter_id in group['params']:
            step = optimizer['state'][parameter_id]['step']
            step = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
            assert step == 15_625

    head_state = payload['stratigraphy_state_dict']
    assert head_state
    assert all(torch.isfinite(tensor).all() for tensor in head_state.values())
    assert set(head_state).isdisjoint(student_state)
    assert 'projector_state_dict' not in payload
    encoder = build_model_from_checkpoint_payload(payload)
    assert encoder.state_dict().keys() == student_state.keys()
    assert all(
        torch.equal(encoder.state_dict()[key], student_state[key])
        for key in student_state
    )
    print(f'HMM PASS: {checkpoint_path} base_stage={expected_stage}')


mae_source = Path(sys.argv[1])
audit_control(mae_source, Path(sys.argv[2]), 'train_amp_mae')
audit_hmm(
    mae_source,
    Path(sys.argv[3]),
    Path(sys.argv[4]),
    'train_amp_mae',
)
bt_source = Path(sys.argv[5])
audit_control(bt_source, Path(sys.argv[6]), 'barlow_twins_training')
audit_hmm(
    bt_source,
    Path(sys.argv[7]),
    Path(sys.argv[8]),
    'barlow_twins_training',
)
PY
```

監査を通過した4つの`full_25ep/latest.pt`だけをpaired comparisonへ渡す。
