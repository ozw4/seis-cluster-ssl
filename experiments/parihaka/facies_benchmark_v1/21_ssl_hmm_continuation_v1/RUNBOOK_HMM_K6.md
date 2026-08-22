# Parihaka paired HMM-K6 execution runbook

MAE100、BT100、local BT100から独立したK=6 pseudo targetを作り、single-head HMM pretextを25 epoch実行する。target、teacher、student初期値は各baseのStage 1 `latest.pt`へ結び付ける。MAE25 / BT25 / local BT25 controlはHMM sourceやresumeに使わない。固定25 epoch比較にはHMM `full_25ep/latest.pt`を使い、`best.pt`は診断専用とする。

## 1. 環境とtargeted tests

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export SUITE=experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export TARGET_CONFIGS="$SUITE/20_hmm_targets"
export HMM_CONFIGS="$SUITE/30_stage2"
export ARTIFACT_SUITE="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1"
export MANIFEST="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/data/parihaka/facies_benchmark_v1/parihaka_amplitude_manifest.json"
export MAE100="$ARTIFACT_SUITE/stage1/mae/full_100ep/latest.pt"
export BT100="$ARTIFACT_SUITE/stage1/barlow_twins/full_100ep/latest.pt"
export LOCAL_BT100="$ARTIFACT_SUITE/stage1/local_barlow_twins_v1/full_100ep/latest.pt"
export MAE_EMBEDDINGS="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/mae100/overlap_x64"
export BT_EMBEDDINGS="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/bt100/overlap_x64"
export LOCAL_BT_EMBEDDINGS="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/local_bt100/overlap_x64"
export MAE_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/mae100"
export BT_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/bt100"
export LOCAL_BT_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/local_bt100"
export LOCAL_BT_CONT="$HMM_CONFIGS/local_bt100/bt_continue"
export CUDA_VISIBLE_DEVICES=1

pytest -q \
  tests/seis_ssl_cluster/test_embedding_extractor.py \
  tests/seis_ssl_cluster/test_strat_checkpoint_extraction.py \
  tests/seis_ssl_cluster/test_strat_hmm_pretraining_head_only.py \
  tests/seis_ssl_cluster/test_strat_hmm_barlow_runner_integration.py \
  tests/seis_ssl_cluster/test_parihaka_barlow_continuation_configs.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_k6_target_configs.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_k6_configs.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_k6_runbook.py
```

## 2. embedding、clustering、export

full-volume embeddingはKに依存しない。各configをdry-runしてからMAE、BT、local BTの順で実行する。

```bash
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/mae100/01_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/mae100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/bt100/01_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/bt100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/local_bt100/01_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$TARGET_CONFIGS/local_bt100/01_extract_embeddings.yaml"

python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/mae100/k6/02_cluster_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/mae100/k6/02_cluster_hmm_k6.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/bt100/k6/02_cluster_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/bt100/k6/02_cluster_hmm_k6.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/local_bt100/k6/02_cluster_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$TARGET_CONFIGS/local_bt100/k6/02_cluster_hmm_k6.yaml"

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/mae100/k6" \
  --pseudo-target-root "$MAE_TARGET_ROOT" --k 6 --confidence 1.0 \
  --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 --dry-run
bash "$TARGET_CONFIGS/mae100/k6/03_export_pseudo_targets.sh"

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/bt100/k6" \
  --pseudo-target-root "$BT_TARGET_ROOT" --k 6 --confidence 1.0 \
  --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 --dry-run
bash "$TARGET_CONFIGS/bt100/k6/03_export_pseudo_targets.sh"

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/local_bt100/k6" \
  --pseudo-target-root "$LOCAL_BT_TARGET_ROOT" --k 6 --confidence 1.0 \
  --boundary-alpha 0.0 --boundary-tau 1.0 --schema-version 2 --dry-run
bash "$TARGET_CONFIGS/local_bt100/k6/03_export_pseudo_targets.sh"
```

`--pseudo-target-root`はKを含まない。既存APIが実artifactを各variantの`k6` subdirectoryへ配置するため、configやcommandへ`k6/k6`を作らない。

## 3. 監査1: Stage 1とembedding source

Stage 1の固定予算とobjective identity、embeddingのsource checkpoint、三variantのvalid-token mask一致を確認する。checkpoint payloadの`paths.output_root`は監査対象にせず、実ファイル位置へ書き換えない。

```bash
python - \
  "$MANIFEST" \
  "$MAE100" "$MAE_EMBEDDINGS" \
  "$BT100" "$BT_EMBEDDINGS" \
  "$LOCAL_BT100" "$LOCAL_BT_EMBEDDINGS" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

from seis_ssl_cluster.data import read_manifest_json
from seis_ssl_cluster.training import load_checkpoint

manifest_ids = {
    item.survey_id for item in read_manifest_json(Path(sys.argv[1]))
}
pairs = (
    ('mae', Path(sys.argv[2]), Path(sys.argv[3]), 'train_amp_mae', None, None),
    (
        'bt', Path(sys.argv[4]), Path(sys.argv[5]),
        'barlow_twins_training', 'barlow_twins_3d', None,
    ),
    (
        'local_bt', Path(sys.argv[6]), Path(sys.argv[7]),
        'barlow_twins_training', 'local_barlow_twins_3d', 128,
    ),
)
masks = {}
for method, checkpoint, embedding_root, stage, objective, local_pairs in pairs:
    assert checkpoint.is_file() and checkpoint.name == 'latest.pt'
    payload = load_checkpoint(checkpoint, map_location='cpu')
    assert payload['epoch'] == 100
    assert payload['global_step'] == 62_500
    assert payload['amp_enabled'] is False
    assert payload['config']['stage'] == stage
    assert payload['config']['train']['batch_size'] == 16
    assert payload['config']['train']['samples_per_epoch'] == 10_000
    if objective is not None:
        assert payload['pretraining_method'] == objective
        assert payload['checkpoint_kind'] == 'barlow_twins_pretraining'
        assert payload['training_state']['completed_epoch'] is True
        assert payload['projector_state_dict']
        if local_pairs is not None:
            assert payload['config']['barlow_twins']['local_pairs_per_crop'] == local_pairs

    metadata = {}
    for path in embedding_root.glob('*.embedding_metadata.json'):
        item = json.loads(path.read_text(encoding='utf-8'))
        metadata[item['survey_id']] = item
    assert metadata.keys() == manifest_ids
    assert all(
        Path(item['checkpoint_path']).resolve() == checkpoint.resolve()
        for item in metadata.values()
    )
    if objective is not None:
        assert all(
            item['pretraining_objective']['method'] == objective
            for item in metadata.values()
        )
    masks[method] = {
        survey_id: np.load(embedding_root / f'{survey_id}.valid_tokens.npy')
        for survey_id in metadata
    }

assert all(mask.keys() == masks['mae'].keys() for mask in masks.values())
for method, mask in masks.items():
    for survey_id in masks['mae']:
        np.testing.assert_array_equal(masks['mae'][survey_id], mask[survey_id])
print(f"Stage 1 and embedding audit PASS: {len(masks['mae'])} surveys")
PY
```

## 4. 監査2: target artifact

公開pseudo-target APIでsurvey set、shape、K=6 label、confidence、valid mask、occupancy、boundary weightを確認する。occupancyは空stateだけをfailさせ、境界数は診断値として表示する。

```bash
python - "$MANIFEST" "$MAE_TARGET_ROOT" "$BT_TARGET_ROOT" "$LOCAL_BT_TARGET_ROOT" <<'PY'
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
roots = {
    'mae': Path(sys.argv[2]).resolve(),
    'bt': Path(sys.argv[3]).resolve(),
    'local_bt': Path(sys.argv[4]).resolve(),
}
assert len(set(roots.values())) == len(roots)
valid_masks = {}
for method, root in roots.items():
    inputs = discover_pseudo_target_inputs(root, k=6)
    assert {item.survey_id for item in inputs} == manifest_ids
    occupancy = np.zeros(6, dtype=np.int64)
    boundary_count = 0
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
        valid_masks[method][item.survey_id] = valid.copy()
    assert np.all(occupancy > 0), occupancy
    print(f'{method}: occupancy={occupancy.tolist()} z_boundaries={boundary_count}')

assert all(mask.keys() == valid_masks['mae'].keys() for mask in valid_masks.values())
for method, mask in valid_masks.items():
    for survey_id in valid_masks['mae']:
        np.testing.assert_array_equal(
            valid_masks['mae'][survey_id], mask[survey_id]
        )
PY
```

## 5. HMM dry-run、feasibility、full run

6設定をdry-runする。表示されるK=6、teacher/student、top-1、loss、両LR、epoch/max stepsも確認する。

```bash
for config in \
  "$HMM_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml" \
  "$HMM_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml" \
  "$HMM_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml" \
  "$HMM_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml" \
  "$HMM_CONFIGS/local_bt100/hmm/k6/01_gpu_feasibility_1step.yaml" \
  "$HMM_CONFIGS/local_bt100/hmm/k6/02_full_25ep.yaml"
do
  python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$config" --dry-run
done

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/local_bt100/hmm/k6/01_gpu_feasibility_1step.yaml"

# fresh full run: --resumeを付けない
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/local_bt100/hmm/k6/02_full_25ep.yaml"
```

中断時だけ同じStage 2 runの`latest.pt`を`--resume`へ渡す。Stage 1やMAE25 / BT25 / local BT25を渡してはいけない。特にlocal HMM25のteacher/studentはlocal Stage 1 `latest.pt`であり、`stage2/local_bt100/bt_continue`を参照しない。

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml" --resume "$ARTIFACT_SUITE/stage2/mae100/hmm/k6/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml" --resume "$ARTIFACT_SUITE/stage2/bt100/hmm/k6/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$HMM_CONFIGS/local_bt100/hmm/k6/02_full_25ep.yaml" --resume "$ARTIFACT_SUITE/stage2/local_bt100/hmm/k6/full_25ep/latest.pt"
```

## 6. 監査3: feasibility/full checkpointとconsumer

三baseのfeasibility/full checkpointについてsource binding、objective identity、学習予算、finite state、frozen/updated領域、optimizer group別stepを確認し、公開factoryでbare encoderをstrict loadする。head更新は乱数から初期headを再構築せず、`head` groupの全parameterがoptimizer stateを持つことで確認する。

```bash
python - \
  "$MAE100" "$MAE_TARGET_ROOT" \
  "$ARTIFACT_SUITE/stage2/mae100/hmm/k6/gpu_feasibility_1step/latest.pt" \
  "$ARTIFACT_SUITE/stage2/mae100/hmm/k6/full_25ep/latest.pt" \
  "$BT100" "$BT_TARGET_ROOT" \
  "$ARTIFACT_SUITE/stage2/bt100/hmm/k6/gpu_feasibility_1step/latest.pt" \
  "$ARTIFACT_SUITE/stage2/bt100/hmm/k6/full_25ep/latest.pt" \
  "$LOCAL_BT100" "$LOCAL_BT_TARGET_ROOT" \
  "$ARTIFACT_SUITE/stage2/local_bt100/hmm/k6/gpu_feasibility_1step/latest.pt" \
  "$ARTIFACT_SUITE/stage2/local_bt100/hmm/k6/full_25ep/latest.pt" <<'PY'
import math
from pathlib import Path
import sys

import torch

from seis_ssl_cluster.config.barlow_twins import (
    resolve_barlow_twins_pretraining_method,
)
from seis_ssl_cluster.models.amplitude_encoder_factory import (
    build_model_from_checkpoint_payload,
)
from seis_ssl_cluster.training import load_checkpoint


def audit_base(
    source_text,
    target_text,
    feasibility_text,
    full_text,
    stage,
    objective,
    local_pairs=None,
):
    source_path = Path(source_text).resolve()
    target_root = Path(target_text).resolve()
    source = load_checkpoint(source_path, map_location='cpu')
    for checkpoint_text, expected_epoch, expected_step in (
        (feasibility_text, 1, 1),
        (full_text, 25, 15_625),
    ):
        path = Path(checkpoint_text)
        assert path.is_file() and path.name == 'latest.pt'
        payload = load_checkpoint(path, map_location='cpu')
        config = payload['stratigraphy_config']
        assert payload['epoch'] == expected_epoch
        assert payload['global_step'] == expected_step
        assert payload['amp_enabled'] is False
        assert payload['config']['stage'] == stage
        if objective is not None:
            assert resolve_barlow_twins_pretraining_method(payload['config']) == objective
        if local_pairs is not None:
            assert payload['config']['barlow_twins']['local_pairs_per_crop'] == local_pairs
        assert config['head']['num_prototypes'] == 6
        assert config['pseudo_targets']['k'] == 6
        assert config['student']['unfreeze_top_blocks'] == 1
        assert config['loss'] == {
            'prototype_weight': 1.0,
            'usage_weight': 0.005,
            'entropy_floor': None,
            'distillation_weight': 0.2,
        }
        assert Path(config['teacher']['checkpoint']).resolve() == source_path
        assert Path(config['student']['init_checkpoint']).resolve() == source_path
        assert Path(config['pseudo_targets']['input_dir']).resolve() == target_root
        assert all(math.isfinite(float(value)) for value in payload['metrics'].values())

        source_state = source['model_state_dict']
        student_state = payload['model_state_dict']
        assert source_state.keys() == student_state.keys()
        frozen_prefixes = (
            'patch_projection.',
            *(f'encoder.layers.{index}.' for index in range(7)),
            'mask_token',
            'encoder_to_decoder.',
            'decoder.',
            'prediction_head.',
        )
        frozen = [key for key in source_state if key.startswith(frozen_prefixes)]
        top = [key for key in source_state if key.startswith('encoder.layers.7.')]
        assert frozen and top
        assert all(torch.equal(source_state[key], student_state[key]) for key in frozen)
        assert any(not torch.equal(source_state[key], student_state[key]) for key in top)

        optimizer = payload['optimizer_state_dict']
        assert len(optimizer['param_groups']) == 2
        groups = {group['name']: group for group in optimizer['param_groups']}
        assert set(groups) == {'head', 'encoder'}
        for group in groups.values():
            assert group['lr'] == 1.0e-5
            assert group['params']
            for parameter_id in group['params']:
                state = optimizer['state'][parameter_id]
                step = state['step']
                step = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
                assert step == expected_step

        head_state = payload['stratigraphy_state_dict']
        assert head_state
        assert all(torch.isfinite(tensor).all() for tensor in head_state.values())
        assert set(head_state).isdisjoint(student_state)
        assert 'projector_state_dict' not in payload
        encoder = build_model_from_checkpoint_payload(payload)
        assert encoder.state_dict().keys() == student_state.keys()
        print(f'checkpoint/consumer PASS: {path}')


audit_base(*sys.argv[1:5], stage='train_amp_mae', objective=None)
audit_base(
    *sys.argv[5:9],
    stage='barlow_twins_training',
    objective='barlow_twins_3d',
)
audit_base(
    *sys.argv[9:13],
    stage='barlow_twins_training',
    objective='local_barlow_twins_3d',
    local_pairs=128,
)
PY
```

full監査を通過した3つの`full_25ep/latest.pt`だけをdownstream主比較へ渡す。

## 7. local BT25 control

local BT25はlocal Stage 1 checkpointからbackboneとprojectorのweightだけを初期化し、同じ`local_barlow_twins_3d` objectiveで25 epoch追加学習する。HMM25と同じ追加予算を使うが、両Stage 2 runは互いに独立している。

feasibility/fullをdry-runし、1-step feasibilityを通してから、`--resume`なしでfresh full runを開始する。

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$LOCAL_BT_CONT/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$LOCAL_BT_CONT/02_full_25ep.yaml" --dry-run

python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$LOCAL_BT_CONT/01_gpu_feasibility_1step.yaml"

# fresh full run: Stage 1 sourceを--resumeへ渡さない
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$LOCAL_BT_CONT/02_full_25ep.yaml"
```

中断時だけlocal BT25自身の`latest.pt`からtraining stateをresumeする。Stage 1 sourceはconfigの`continuation.init_checkpoint`だけで指定する。

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$LOCAL_BT_CONT/02_full_25ep.yaml" --resume "$ARTIFACT_SUITE/stage2/local_bt100/bt_continue/full_25ep/latest.pt"
```

full checkpointについて固定予算、source binding、local objective、finite metrics、学習範囲、fresh optimizerを監査する。

```bash
python - \
  "$LOCAL_BT100" \
  "$ARTIFACT_SUITE/stage2/local_bt100/bt_continue/full_25ep/latest.pt" <<'PY'
import json
import math
from pathlib import Path
import sys

import torch

from seis_ssl_cluster.training import load_checkpoint

source_path = Path(sys.argv[1]).resolve()
stage2_path = Path(sys.argv[2]).resolve()
assert source_path.is_file() and source_path.name == 'latest.pt'
assert stage2_path.is_file() and stage2_path.name == 'latest.pt'
source = load_checkpoint(source_path, map_location='cpu')
stage2 = load_checkpoint(stage2_path, map_location='cpu')

assert source['epoch'] == 100
assert source['global_step'] == 62_500
assert source['pretraining_method'] == 'local_barlow_twins_3d'
assert source['checkpoint_kind'] == 'barlow_twins_pretraining'
assert source['config']['barlow_twins']['local_pairs_per_crop'] == 128
assert source['amp_enabled'] is False
assert source['training_state']['completed_epoch'] is True
assert source['projector_state_dict']

assert stage2['epoch'] == 25
assert stage2['global_step'] == 15_625
assert stage2['amp_enabled'] is False
assert stage2['pretraining_method'] == 'local_barlow_twins_3d'
assert stage2['checkpoint_kind'] == 'barlow_twins_pretraining'
assert stage2['training_state']['completed_epoch'] is True
config = stage2['config']
assert config['barlow_twins']['method'] == 'local_barlow_twins_3d'
assert config['barlow_twins']['local_pairs_per_crop'] == 128
assert config['train']['lr'] == 1.0e-5
assert config['continuation']['unfreeze_top_blocks'] == 1
assert Path(config['continuation']['init_checkpoint']).resolve() == source_path

source_backbone = source['model_state_dict']
stage2_backbone = stage2['model_state_dict']
assert source_backbone.keys() == stage2_backbone.keys()
frozen_prefixes = (
    'patch_projection.',
    *(f'encoder.layers.{index}.' for index in range(7)),
    'mask_token',
    'encoder_to_decoder.',
    'decoder.',
    'prediction_head.',
)
frozen = [key for key in source_backbone if key.startswith(frozen_prefixes)]
top = [key for key in source_backbone if key.startswith('encoder.layers.7.')]
assert frozen and top
assert all(torch.equal(source_backbone[key], stage2_backbone[key]) for key in frozen)
assert any(not torch.equal(source_backbone[key], stage2_backbone[key]) for key in top)

source_projector = source['projector_state_dict']
stage2_projector = stage2['projector_state_dict']
assert source_projector.keys() == stage2_projector.keys()
projector_parameters = [
    key for key in source_projector if key.endswith(('.weight', '.bias'))
]
assert projector_parameters
assert any(
    not torch.equal(source_projector[key], stage2_projector[key])
    for key in projector_parameters
)

optimizer_steps = {
    int(state['step'].item())
    if isinstance(state['step'], torch.Tensor)
    else int(state['step'])
    for state in stage2['optimizer_state_dict']['state'].values()
}
assert optimizer_steps == {15_625}

metric_names = (
    'training_loss',
    'gradient_norm',
    'projection_std_mean',
    'projection_std_min',
    'projection_norm_mean',
    'cross_correlation_diag_mean',
    'cross_correlation_offdiag_rms',
)
history = json.loads((stage2_path.parent / 'history.json').read_text())
assert len(history) == 25
assert all(
    math.isfinite(float(record[name]))
    for record in history
    for name in metric_names
)
print(f'local BT100 -> local BT25 continuation PASS: {stage2_path}')
PY
```

GPU feasibilityではOOMがなく、loss、gradient、projection、correlation metricsがfiniteであることも確認する。full監査に通ったlocal BT25 `full_25ep/latest.pt`をcontrolの固定予算checkpointとする。
