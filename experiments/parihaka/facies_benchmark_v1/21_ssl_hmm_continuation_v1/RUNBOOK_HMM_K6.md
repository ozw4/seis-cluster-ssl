# Parihaka paired HMM-K6 execution runbook

このrunbookは、ParihakaのMAE100とBT100から互いに独立したK=6 pseudo targetを作り、single-head HMM pretextを25 epoch実行して、checkpointとbare encoder consumerまで監査する手順である。Phase 1ではK=6だけを扱う。コマンドは記載順に実行し、live artifactを要求する監査は通常のportable pytestには含めない。

主要原則は次のとおりである。

- MAE target、teacher、student初期値はすべてMAE100 Stage 1 `latest.pt`へ結び付ける。
- BT target、teacher、student初期値はすべてBT100 Stage 1 `latest.pt`へ結び付ける。
- MAE25 / BT25 controlをtarget生成、teacher、student初期値、`--resume`に使わない。
- 25 epoch固定予算の主入力にはHMM `full_25ep/latest.pt`を使い、`best.pt`は診断専用とする。
- fresh runでは`--resume`を付けない。`--resume`は同じStage 2 runの中断再開だけに使う。

## 1. 環境

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export SUITE=experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export CUDA_VISIBLE_DEVICES=1

export TARGET_CONFIGS="$SUITE/20_hmm_targets"
export HMM_CONFIGS="$SUITE/30_stage2"
export ARTIFACT_SUITE="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1"
export MAE100="$ARTIFACT_SUITE/stage1/mae/full_100ep/latest.pt"
export BT100="$ARTIFACT_SUITE/stage1/barlow_twins/full_100ep/latest.pt"
export MAE_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/mae100"
export BT_TARGET_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/bt100"
export MAE_HMM25="$ARTIFACT_SUITE/stage2/mae100/hmm/k6/full_25ep/latest.pt"
export BT_HMM25="$ARTIFACT_SUITE/stage2/bt100/hmm/k6/full_25ep/latest.pt"
```

以降の相対pathは`/workspace`基準である。実行前に`git status --short`も記録する。

## 2. targeted tests

base-objective-aware checkpoint consumer、MAE/BT single-head component、paired target/config、および既存head-only/checkpoint extraction回帰をまとめて実行する。

```bash
pytest -q \
  tests/seis_ssl_cluster/test_embedding_extractor.py \
  tests/seis_ssl_cluster/test_strat_checkpoint_extraction.py \
  tests/seis_ssl_cluster/test_strat_hmm_pretraining_head_only.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_k6_target_configs.py \
  tests/seis_ssl_cluster/test_parihaka_hmm_k6_configs.py
```

ここで失敗した場合はlive pipelineを開始しない。

## 3. Stage 1 source監査

MAE100とBT100の固定100 epoch checkpointを監査する。

```bash
python - "$MAE100" "$BT100" <<'PY'
from pathlib import Path
import sys

from seis_ssl_cluster.training import load_checkpoint


def audit(path_text: str, *, method: str) -> None:
    path = Path(path_text)
    assert path.is_file(), path
    assert path.name == 'latest.pt', path
    payload = load_checkpoint(path, map_location='cpu')
    assert payload['epoch'] == 100
    assert payload['global_step'] == 62_500
    assert payload['amp_enabled'] is False
    train = payload['config']['train']
    assert train['amp'] is False
    assert train['batch_size'] == 16
    assert train['samples_per_epoch'] == 10_000
    if method == 'mae':
        assert payload['training_state']['stage'] == 'train_amp_mae'
    else:
        assert payload['training_state']['stage'] == 'barlow_twins_training'
        assert payload['pretraining_method'] == 'barlow_twins_3d'
        assert payload['checkpoint_kind'] == 'barlow_twins_pretraining'
        assert payload['projector_state_dict']
    print(f'{method}: epoch=100 global_step=62500 source={path}')


audit(sys.argv[1], method='mae')
audit(sys.argv[2], method='barlow_twins')
PY
```

## 4. embedding抽出

MAE100、BT100の順に、各configをdry-runしてからfull-volume embeddingを抽出する。このartifactはKに依存しないためbase固有root直下へ保存し、後続のK=6 clustering（将来のK=8 / K=10を含む）から再利用する。

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TARGET_CONFIGS/mae100/01_extract_embeddings.yaml" \
  --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TARGET_CONFIGS/mae100/01_extract_embeddings.yaml"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TARGET_CONFIGS/bt100/01_extract_embeddings.yaml" \
  --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TARGET_CONFIGS/bt100/01_extract_embeddings.yaml"
```

完了後、manifestのsurvey setとembedding metadataのsurvey setを一致させ、各`checkpoint_path`が対応するStage 1 sourceを指すことを確認する。

```bash
python - \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/data/parihaka/facies_benchmark_v1/parihaka_amplitude_manifest.json" \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/mae100/overlap_x64" \
  "$MAE100" \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/bt100/overlap_x64" \
  "$BT100" <<'PY'
import json
from pathlib import Path
import sys

from seis_ssl_cluster.data import read_manifest_json

manifest_ids = {item.survey_id for item in read_manifest_json(Path(sys.argv[1]))}
for root_text, checkpoint_text in ((sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5])):
    root = Path(root_text)
    checkpoint = Path(checkpoint_text).resolve()
    metadata_files = sorted(root.glob('*.embedding_metadata.json'))
    assert metadata_files, root
    metadata = [json.loads(path.read_text(encoding='utf-8')) for path in metadata_files]
    assert {item['survey_id'] for item in metadata} == manifest_ids
    assert all(Path(item['checkpoint_path']).resolve() == checkpoint for item in metadata)
    print(f'{root}: surveys={len(metadata)} checkpoint={checkpoint}')
PY
```

## 5. K=6 clustering

MAE、BTのpaired configをdry-run後に実行する。両者で異なるのはembedding inputとclustering outputだけである。

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$TARGET_CONFIGS/mae100/k6/02_cluster_hmm_k6.yaml" \
  --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$TARGET_CONFIGS/mae100/k6/02_cluster_hmm_k6.yaml"

python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$TARGET_CONFIGS/bt100/k6/02_cluster_hmm_k6.yaml" \
  --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$TARGET_CONFIGS/bt100/k6/02_cluster_hmm_k6.yaml"
```

## 6. pseudo target export

各export scriptと同じ引数へ`--dry-run`を追加して入力を検証し、その後scriptを実行する。`--pseudo-target-root`にはbase固有root（`mae100` / `bt100`）を渡す。pseudo-target APIがK固有directoryを追加するため、実artifactはそれぞれ`$MAE_TARGET_ROOT/k6`と`$BT_TARGET_ROOT/k6`に生成される。

```bash
python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/mae100/k6" \
  --pseudo-target-root "$MAE_TARGET_ROOT" \
  --k 6 \
  --confidence 1.0 \
  --boundary-alpha 0.0 \
  --boundary-tau 1.0 \
  --schema-version 2 \
  --dry-run
bash "$TARGET_CONFIGS/mae100/k6/03_export_pseudo_targets.sh"

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/hmm_targets/bt100/k6" \
  --pseudo-target-root "$BT_TARGET_ROOT" \
  --k 6 \
  --confidence 1.0 \
  --boundary-alpha 0.0 \
  --boundary-tau 1.0 \
  --schema-version 2 \
  --dry-run
bash "$TARGET_CONFIGS/bt100/k6/03_export_pseudo_targets.sh"
```

## 7. target live監査

公開pseudo-target discovery/load APIで両base rootを検証する。`discover_pseudo_target_inputs(root, k=6)`が検索する実directoryは`root/k6`である。shape、label mask、confidence、boundary weightのschema validationはloader内でも実行される。occupancyとz方向の隣接境界数は表示するが、最小占有率は設定せず、空stateだけを失敗とする。

```bash
python - \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/data/parihaka/facies_benchmark_v1/parihaka_amplitude_manifest.json" \
  "$MAE_TARGET_ROOT" \
  "$BT_TARGET_ROOT" <<'PY'
from pathlib import Path
import sys

import numpy as np

from seis_ssl_cluster.data import read_manifest_json
from seis_ssl_cluster.stratigraphy import (
    discover_pseudo_target_inputs,
    load_pseudo_target_arrays,
)

manifest_ids = {item.survey_id for item in read_manifest_json(Path(sys.argv[1]))}
roots = [Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()]
assert roots[0] != roots[1]

for root in roots:
    inputs = discover_pseudo_target_inputs(root, k=6)
    assert {item.survey_id for item in inputs} == manifest_ids
    occupancy = np.zeros(6, dtype=np.int64)
    boundary_count = 0
    valid_count = 0
    for item in inputs:
        arrays = load_pseudo_target_arrays(item, mmap_mode='r')
        labels = np.asarray(arrays.labels)
        confidence = np.asarray(arrays.confidence)
        valid = np.asarray(arrays.valid_tokens)
        boundary_weight = np.asarray(arrays.boundary_weight)
        assert labels.shape == confidence.shape == valid.shape
        assert boundary_weight.shape == labels.shape
        assert np.all(np.isfinite(labels))
        assert np.all(np.isfinite(confidence))
        assert np.all(np.isfinite(boundary_weight))
        assert np.all((confidence >= 0.0) & (confidence <= 1.0))
        assert np.all((labels[valid] >= 0) & (labels[valid] <= 5))
        assert np.all(labels[~valid] == -1)
        assert np.all(confidence[~valid] == 0.0)
        occupancy += np.bincount(labels[valid], minlength=6)[:6]
        valid_count += int(np.count_nonzero(valid))
        adjacent_valid = valid[..., 1:] & valid[..., :-1]
        boundary_count += int(np.count_nonzero(
            adjacent_valid & (labels[..., 1:] != labels[..., :-1])
        ))
    assert valid_count > 0
    assert np.all(occupancy > 0), occupancy
    print(
        f'{root}: surveys={len(inputs)} valid_tokens={valid_count} '
        f'occupancy={occupancy.tolist()} z_boundaries={boundary_count}'
    )
PY
```

## 8. HMM config dry-run

既存resolverで4 configのK、source、loss、LR、budgetを一覧表示する。この確認はlive Stage 1 checkpointとtarget rootの存在も検証する。

```bash
python - \
  "$HMM_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml" \
  "$HMM_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml" \
  "$HMM_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml" \
  "$HMM_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml" <<'PY'
from pathlib import Path
import sys

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config

for value in sys.argv[1:]:
    path = Path(value)
    config = resolve_strat_hmm_pretext_config(load_config(path))
    print(path)
    print(f"  k={config['pseudo_targets']['k']}")
    print(f"  teacher={config['teacher']['checkpoint']}")
    print(f"  student={config['student']['init_checkpoint']}")
    print(f"  top_blocks={config['student']['unfreeze_top_blocks']}")
    print(
        '  loss='
        f"{config['loss']['prototype_weight']}/"
        f"{config['loss']['usage_weight']}/"
        f"{config['loss']['distillation_weight']}"
    )
    print(f"  lr={config['train']['lr']}/{config['train']['encoder_lr']}")
    print(f"  epochs={config['train']['epochs']} max_steps={config['train']['max_steps']}")
PY
```

続けて同じ4 configを既存training CLIでdry-runする。

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml" \
  --dry-run
```

期待値はK=6、対応するStage 1 teacher/student、top-1、loss `1.0 / 0.005 / 0.2`、head/encoder LRともに`1e-5`、feasibility `epochs=1, max_steps=1`、full `epochs=25, max_steps=None`である。

## 9. GPU feasibility

MAE、BTをそれぞれfreshなoutput rootで1 step実行する。

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/mae100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/bt100/hmm/k6/01_gpu_feasibility_1step.yaml"
```

CUDA OOMがないことを確認した後、両checkpointを監査する。

```bash
python - \
  "$ARTIFACT_SUITE/stage2/mae100/hmm/k6/gpu_feasibility_1step/latest.pt" \
  "$ARTIFACT_SUITE/stage2/bt100/hmm/k6/gpu_feasibility_1step/latest.pt" <<'PY'
import math
from pathlib import Path
import sys

import torch

from seis_ssl_cluster.training import load_checkpoint

for value in sys.argv[1:]:
    path = Path(value)
    assert path.is_file(), path
    payload = load_checkpoint(path, map_location='cpu')
    assert payload['epoch'] == 1
    assert payload['global_step'] == 1
    assert payload['amp_enabled'] is False
    assert all(math.isfinite(float(value)) for value in payload['metrics'].values())
    groups = payload['optimizer_state_dict']['param_groups']
    assert [(group['name'], group['lr']) for group in groups] == [
        ('head', 1.0e-5),
        ('encoder', 1.0e-5),
    ]
    optimizer_steps = {
        int(state['step'].item()) if isinstance(state['step'], torch.Tensor)
        else int(state['step'])
        for state in payload['optimizer_state_dict']['state'].values()
    }
    assert optimizer_steps == {1}
    trainable = payload['trainability_summary']['trainable_names']
    assert trainable
    assert all(name.startswith('encoder.layers.7.') for name in trainable)
    assert payload['stratigraphy_config']['student']['unfreeze_top_blocks'] == 1
    print(f'feasibility PASS: {path}')
PY
```

合格条件は、CUDA OOMなし、`epoch=1`、`global_step=1`、finite metrics、`latest.pt`実在、`head`/`encoder`の2 optimizer group、両LR `1e-5`、top-1だけがtrainableであること。

## 10. 25 epoch full runとresume

fresh runでは`--resume`を付けない。

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml"
```

中断した場合だけ、同じStage 2 outputの`latest.pt`を渡して再開する。

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/mae100/hmm/k6/02_full_25ep.yaml" \
  --resume "$ARTIFACT_SUITE/stage2/mae100/hmm/k6/full_25ep/latest.pt"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$HMM_CONFIGS/bt100/hmm/k6/02_full_25ep.yaml" \
  --resume "$ARTIFACT_SUITE/stage2/bt100/hmm/k6/full_25ep/latest.pt"
```

Stage 1 `latest.pt`はweights-onlyのteacher/student sourceであり、`--resume`へ渡してはいけない。MAE25 / BT25 controlも`--resume`へ渡さない。

## 11. full checkpoint監査

両HMM25 checkpointを対応するStage 1 sourceと比較する。frozen backbone/decoderの完全一致、top blockとheadの変化、fresh optimizerの15,625 stepを確認する。

```bash
python - \
  "$MAE100" "$MAE_HMM25" "$MAE_TARGET_ROOT" \
  "$BT100" "$BT_HMM25" "$BT_TARGET_ROOT" <<'PY'
import math
from pathlib import Path
import sys

import torch

from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm import build_strat_hmm_components


def audit(source_text: str, hmm_text: str, target_root_text: str) -> None:
    source_path = Path(source_text)
    hmm_path = Path(hmm_text)
    target_root = Path(target_root_text)
    assert source_path.is_file(), source_path
    assert hmm_path.is_file(), hmm_path
    assert hmm_path.name == 'latest.pt'
    source = load_checkpoint(source_path, map_location='cpu')
    hmm = load_checkpoint(hmm_path, map_location='cpu')
    config = hmm['stratigraphy_config']

    assert hmm['epoch'] == 25
    assert hmm['global_step'] == 15_625
    assert hmm['amp_enabled'] is False
    assert hmm['training_state']['stage'] == 'train_strat_hmm_pretext'
    assert config['train']['amp'] is False
    assert config['head']['num_prototypes'] == 6
    assert config['student']['unfreeze_top_blocks'] == 1
    assert config['loss']['prototype_weight'] == 1.0
    assert config['loss']['usage_weight'] == 0.005
    assert config['loss']['distillation_weight'] == 0.2
    assert Path(config['teacher']['checkpoint']).resolve() == source_path.resolve()
    assert Path(config['student']['init_checkpoint']).resolve() == source_path.resolve()
    assert Path(config['pseudo_targets']['input_dir']).resolve() == target_root.resolve()
    assert all(math.isfinite(float(value)) for value in hmm['metrics'].values())

    source_state = source['model_state_dict']
    student_state = hmm['model_state_dict']
    assert source_state.keys() == student_state.keys()
    unchanged_prefixes = (
        'patch_projection.',
        *(f'encoder.layers.{index}.' for index in range(7)),
        'mask_token',
        'encoder_to_decoder.',
        'decoder.',
        'prediction_head.',
    )
    unchanged_keys = [
        key for key in source_state if key.startswith(unchanged_prefixes)
    ]
    assert unchanged_keys
    assert all(torch.equal(source_state[key], student_state[key]) for key in unchanged_keys)
    top_keys = [key for key in source_state if key.startswith('encoder.layers.7.')]
    assert top_keys
    assert any(not torch.equal(source_state[key], student_state[key]) for key in top_keys)

    torch.manual_seed(config['train']['seed'])
    initial = build_strat_hmm_components(config, device='cpu')
    initial_head = initial.head.state_dict()
    final_head = hmm['stratigraphy_state_dict']
    assert initial_head.keys() == final_head.keys()
    assert any(not torch.equal(initial_head[key], final_head[key]) for key in initial_head)

    trainable = hmm['trainability_summary']['trainable_names']
    assert trainable
    assert all(name.startswith('encoder.layers.7.') for name in trainable)
    groups = hmm['optimizer_state_dict']['param_groups']
    assert [(group['name'], group['lr']) for group in groups] == [
        ('head', 1.0e-5),
        ('encoder', 1.0e-5),
    ]
    states = hmm['optimizer_state_dict']['state']
    assert states
    optimizer_steps = {
        int(state['step'].item()) if isinstance(state['step'], torch.Tensor)
        else int(state['step'])
        for state in states.values()
    }
    assert optimizer_steps == {15_625}
    print(f'full PASS: {hmm_path}')


audit(sys.argv[1], sys.argv[2], sys.argv[3])
audit(sys.argv[4], sys.argv[5], sys.argv[6])
PY
```

optimizer stateの全stepが15,625であることは、Stage 1 optimizerを引き継がず、HMM trainable stateに対してfreshに15,625 updateを実行した証拠とする。

## 12. encoder consumer監査

両HMM25をbare encoder factoryでstrict loadし、extractorが使用するstratigraphy metadata helperでbase objectiveを区別する。HMM headは`stratigraphy_state_dict`にのみ存在し、BT projectorはHMM checkpointへ保存されないことも確認する。

```bash
python - "$MAE_HMM25" "$BT_HMM25" <<'PY'
from pathlib import Path
import sys

from seis_ssl_cluster.embedding.extractor import _stratigraphy_pretext_metadata
from seis_ssl_cluster.models.amplitude_encoder_factory import (
    build_model_from_checkpoint_payload,
)
from seis_ssl_cluster.training import load_checkpoint

expected = (
    ('amp_mae3d', Path(sys.argv[1])),
    ('barlow_twins_3d', Path(sys.argv[2])),
)
for base_objective, path in expected:
    assert path.is_file(), path
    payload = load_checkpoint(path, map_location='cpu')
    model = build_model_from_checkpoint_payload(payload)
    assert model.state_dict().keys() == payload['model_state_dict'].keys()
    assert payload['stratigraphy_state_dict']
    assert 'projector_state_dict' not in payload
    assert not any(
        key.startswith(('backbone.', 'projector.', 'head.', 'stratigraphy.'))
        for key in payload['model_state_dict']
    )
    metadata = _stratigraphy_pretext_metadata(payload)
    assert metadata is not None
    assert metadata['method'] == 'strat_hmm_pretext'
    assert metadata['base_objective'] == base_objective
    assert metadata['head_num_prototypes'] == 6
    print(
        f"consumer PASS: {path} method={metadata['method']} "
        f"base={metadata['base_objective']} k={metadata['head_num_prototypes']}"
    )
PY
```

ここまで通過した`full_25ep/latest.pt`だけを、固定25 epoch比較のdownstream主入力として扱う。`best.pt`はloss診断には使えるが、主比較のcheckpoint selectionには使わない。
