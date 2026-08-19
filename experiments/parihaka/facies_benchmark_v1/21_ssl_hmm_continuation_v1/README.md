# Parihaka SSL pretraining and continuation

## 目的

Parihaka固有の3D MAEと3D Barlow Twinsを同じ学習予算で事前学習し、HMM second-stage pretrainingの初期重みを作成する。

## Paired HMM-K6

target生成からHMM25、live artifact、bare encoder consumerまでの実行・監査手順は[RUNBOOK_HMM_K6.md](RUNBOOK_HMM_K6.md)にまとめる。

| Phase 1条件 | target source | teacher / student初期値 | Stage 2 objective | 固定予算の主checkpoint |
| --- | --- | --- | --- | --- |
| MAE100 → HMM-K6 25 | MAE100 `latest.pt` | MAE100 `latest.pt` | single-head HMM K=6 | HMM `full_25ep/latest.pt` |
| BT100 → HMM-K6 25 | BT100 `latest.pt` | BT100 `latest.pt` | single-head HMM K=6 | HMM `full_25ep/latest.pt` |
| MAE100 → MAE25 control | なし | MAE100 `latest.pt` | MAE reconstruction | MAE25 `full_25ep/latest.pt` |
| BT100 → BT25 control | なし | BT100 `latest.pt` | Barlow Twins | BT25 `full_25ep/latest.pt` |

## 学習条件

両methodの共通条件は次のとおりである。

- Parihaka amplitude manifestとpath list
- amplitude preprocessing（normalized clip 8.0、trace RMS z-score AGC、window 65）とzero mask
- `128 x 128 x 128` crop、`8 x 8 x 8` patch
- encoder dim 384 / depth 8 / heads 6
- compatibility decoder dim 256 / depth 4 / heads 4
- batch size 16、learning rate `1.0e-4`、weight decay 0.05
- FP32（`amp: false`）、seed 42、gradient clipping 1.0

full runは100 epoch、10,000 samples/epochであり、625 steps/epoch、合計62,500 global stepsとなる。

MAEはspatial mask ratio 0.75、MSE reconstruction、gradient weight 0.0、visible reconstruction weight 0.1、patch z-score target normalizationを使用する。Barlow Twinsはhorizontal flip probability 0.5、projector dim 384、redundancy weight 0.005、normalization epsilon `1.0e-4`を使用する。

artifact output rootは次の配下に分離して保存する。

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/
├── mae/
│   ├── gpu_feasibility_1step/
│   └── full_100ep/
└── barlow_twins/
    ├── gpu_feasibility_1step/
    └── full_100ep/
```

## Stage 1 実行順

リポジトリrootで環境変数を設定する。

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export STAGE1=experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1/10_stage1
```

1. 専用の契約テストを実行する。

   ```bash
   pytest -q tests/seis_ssl_cluster/test_parihaka_stage1_ssl_configs.py
   ```

2. 4設定をdry-runして解決結果を確認する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/01_gpu_feasibility_1step.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/02_full_100ep.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/01_gpu_feasibility_1step.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/02_full_100ep.yaml" --dry-run
   ```

3. Barlow TwinsのCUDA 1-step feasibilityを実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/01_gpu_feasibility_1step.yaml"
   ```

4. MAEのCUDA 1-step feasibilityを実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/01_gpu_feasibility_1step.yaml"
   ```

5. 両方のfeasibilityが完了条件を満たした場合に限り、100 epoch本学習を実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$STAGE1/barlow_twins/02_full_100ep.yaml"
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$STAGE1/mae/02_full_100ep.yaml"
   ```

## 完了判定

各GPU feasibilityでは、CUDA OOMが発生しないこと、lossとgradient normがfiniteであること、`latest.pt`が保存されること、peak CUDA memoryに本学習を継続できる余裕があることを確認する。Barlow Twinsではprojection metricsとcorrelation metricsもfiniteであることを確認する。

各100 epoch runは、epoch 100、global step 62,500に到達し、対応する`full_100ep/latest.pt`が存在することを確認する。resolved configがFP32 / `amp: false`であり、lossとgradient metricsが全期間でfiniteであることも必要とする。

Stage 2の初期値には`best.pt`ではなく、固定学習予算を完了した各methodの`full_100ep/latest.pt`を使用する。

## Stage 2 MAE continuation

MAE100 → MAE25 controlでは、Stage 1 MAE 100 epochの`latest.pt`からmodel weightだけを読み込み、encoderの上位1 blockとMAE reconstruction modulesを25 epoch追加学習する。optimizer、epoch、global step、RNGなどのtraining stateは引き継がず、Stage 2用に新規作成する。

このsuiteの`continuation`は特定のobjective固有の目的名ではなく、初期weightを受け取る第二段階学習を表す実行契約である。

artifact output rootは次の配下に分離して保存する。

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/mae_continue/
├── gpu_feasibility_1step/
└── full_25ep/
```

### 実行順

リポジトリrootで環境変数を設定する。

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export SUITE=experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export MAE_CONT="$SUITE/30_stage2/mae100/mae_continue"
export CUDA_VISIBLE_DEVICES=1
```

1. 専用のconfig契約テストを実行する。このテストは外部artifactを要求しない。

   ```bash
   pytest -q tests/seis_ssl_cluster/test_parihaka_mae_continuation_configs.py
   ```

2. feasibilityとfull設定をdry-runし、source checkpointとtrainable encoder block数を含む解決結果を確認する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$MAE_CONT/01_gpu_feasibility_1step.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$MAE_CONT/02_full_25ep.yaml" --dry-run
   ```

3. Stage 1 source checkpointの実在と固定学習予算の完了状態をlive artifact上で確認する。

   ```bash
   python - "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt" <<'PY'
   from pathlib import Path
   import sys

   from seis_ssl_cluster.training import load_checkpoint

   checkpoint_path = Path(sys.argv[1])
   assert checkpoint_path.is_file(), checkpoint_path
   payload = load_checkpoint(checkpoint_path, map_location='cpu')
   assert payload['epoch'] == 100
   assert payload['global_step'] == 62_500
   assert payload['amp_enabled'] is False
   assert payload['config']['train']['amp'] is False
   assert payload['config']['train']['batch_size'] == 16
   assert payload['config']['train']['samples_per_epoch'] == 10_000
   assert payload['training_state']['stage'] == 'train_amp_mae'
   assert payload['training_state']['checkpoint_kind'] == 'epoch'
   assert payload['training_state']['batch_index'] is None
   print(f'Stage 1 source checkpoint verified: {checkpoint_path}')
   PY
   ```

4. GPU 1-step feasibilityを実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$MAE_CONT/01_gpu_feasibility_1step.yaml"
   ```

5. feasibilityの完了条件を満たした場合、freshな25 epoch本学習を実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$MAE_CONT/02_full_25ep.yaml"
   ```

6. Stage 2本学習を中断した場合は、Stage 2の`latest.pt`からfull resumeする。

   ```bash
   python proc/seis_ssl_cluster/train_amp_mae.py --config "$MAE_CONT/02_full_25ep.yaml" --resume "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/mae_continue/full_25ep/latest.pt"
   ```

Stage 1のcheckpointは`continuation.init_checkpoint`で指定するweights-only sourceであり、`--resume`へ渡してはいけない。`--resume`はStage 2自身のoptimizer、counter、RNGを含むtraining stateを復元する場合だけ使用する。

### 完了判定

GPU feasibilityではCUDA OOMが発生せず、lossとgradient normがfiniteであり、専用output rootに`latest.pt`が保存されることを確認する。

full run完了時は、`epoch = 25`、`global_step = 15,625`、`amp_enabled = false`であること、checkpoint configが`train.lr = 1.0e-5`および`continuation.unfreeze_top_blocks = 1`であることを確認する。`full_25ep/latest.pt`が存在し、lossとgradient normがfiniteであることも必要とする。

## Stage 2 Barlow Twins continuation

BT100 → BT25 controlでは、Stage 1 Barlow Twins 100 epochの`latest.pt`からbackboneとprojectorのweightだけを読み込み、encoderの上位1 blockとprojectorをBarlow Twins objectiveで25 epoch追加学習する。Stage 1のoptimizer、epoch、global step、RNG、historyは引き継がず、Stage 2用にAdamW optimizerを新規作成する。このcontrolは将来のBT100 → HMM25と同じ追加学習予算で比較するためのものである。

artifact output rootは次の配下に分離して保存する。

```text
${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/bt100/bt_continue/
├── gpu_feasibility_1step/
└── full_25ep/
```

### 実行順

リポジトリrootで環境変数を設定する。

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export SUITE=experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export BT_CONT="$SUITE/30_stage2/bt100/bt_continue"
export CUDA_VISIBLE_DEVICES=1
```

1. 専用config testとBarlow Twins continuation runnerのtargeted testsを実行する。これらはlive artifactを要求しない。

   ```bash
   pytest -q \
     tests/seis_ssl_cluster/test_parihaka_barlow_continuation_configs.py \
     tests/seis_ssl_cluster/test_barlow_twins_continuation.py \
     tests/seis_ssl_cluster/test_barlow_twins_training_contract.py
   ```

2. feasibilityとfull設定をdry-runし、weights-only source checkpointとtrainable encoder block数を含む解決結果を確認する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$BT_CONT/01_gpu_feasibility_1step.yaml" --dry-run
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$BT_CONT/02_full_25ep.yaml" --dry-run
   ```

3. Stage 1 source checkpointの実在、method identity、projector、および固定学習予算の完了状態をlive artifact上で確認する。

   ```bash
   python - "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/barlow_twins/full_100ep/latest.pt" <<'PY'
   from pathlib import Path
   import sys

   from seis_ssl_cluster.training import load_checkpoint

   checkpoint_path = Path(sys.argv[1])
   assert checkpoint_path.is_file(), checkpoint_path
   payload = load_checkpoint(checkpoint_path, map_location='cpu')
   assert payload['epoch'] == 100
   assert payload['global_step'] == 62_500
   assert payload['amp_enabled'] is False
   assert payload['pretraining_method'] == 'barlow_twins_3d'
   assert payload['checkpoint_kind'] == 'barlow_twins_pretraining'
   training_state = payload['training_state']
   assert training_state['stage'] == 'barlow_twins_training'
   assert training_state['resume_boundary'] == 'epoch'
   assert training_state['completed_epoch'] is True
   assert payload['config']['train']['amp'] is False
   assert payload['config']['train']['batch_size'] == 16
   assert payload['config']['train']['samples_per_epoch'] == 10_000
   assert payload['projector_state_dict']
   print(f'Stage 1 Barlow Twins source verified: {checkpoint_path}')
   PY
   ```

4. GPU 1-step feasibilityを実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$BT_CONT/01_gpu_feasibility_1step.yaml"
   ```

5. feasibilityの完了条件を満たした場合、既存outputのないfreshなrootへ25 epoch本学習を実行する。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$BT_CONT/02_full_25ep.yaml"
   ```

6. Stage 2本学習を中断した場合は、Stage 2自身の`latest.pt`からfull resumeする。

   ```bash
   python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$BT_CONT/02_full_25ep.yaml" --resume "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/bt100/bt_continue/full_25ep/latest.pt"
   ```

Stage 1 checkpointは`continuation.init_checkpoint`で指定するbackbone + projectorのweights-only sourceであり、`--resume`へ渡してはいけない。`--resume`はStage 2自身のoptimizer、counter、RNGを含むtraining stateを復元する場合だけ使用する。

7. full checkpointの完了状態、finite metrics、学習範囲、およびfresh optimizerをStage 1 sourceと比較して確認する。

   ```bash
   python - \
     "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/barlow_twins/full_100ep/latest.pt" \
     "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/parihaka/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/bt100/bt_continue/full_25ep/latest.pt" <<'PY'
   import json
   import math
   from pathlib import Path
   import sys

   import torch

   from seis_ssl_cluster.training import load_checkpoint

   source_path = Path(sys.argv[1])
   stage2_path = Path(sys.argv[2])
   assert source_path.is_file(), source_path
   assert stage2_path.is_file(), stage2_path
   source = load_checkpoint(source_path, map_location='cpu')
   stage2 = load_checkpoint(stage2_path, map_location='cpu')

   assert stage2['epoch'] == 25
   assert stage2['global_step'] == 15_625
   assert stage2['amp_enabled'] is False
   assert stage2['pretraining_method'] == 'barlow_twins_3d'
   assert stage2['checkpoint_kind'] == 'barlow_twins_pretraining'
   assert stage2['training_state']['stage'] == 'barlow_twins_training'
   assert stage2['training_state']['resume_boundary'] == 'epoch'
   assert stage2['training_state']['completed_epoch'] is True
   assert stage2['config']['train']['lr'] == 1.0e-5
   assert stage2['config']['continuation']['unfreeze_top_blocks'] == 1
   assert Path(stage2['config']['continuation']['init_checkpoint']) == source_path

   source_backbone = source['model_state_dict']
   stage2_backbone = stage2['model_state_dict']
   assert source_backbone.keys() == stage2_backbone.keys()

   unchanged_prefixes = (
       'patch_projection.',
       *(f'encoder.layers.{index}.' for index in range(7)),
       'mask_token',
       'encoder_to_decoder.',
       'decoder.',
       'prediction_head.',
   )
   unchanged_keys = [
       key for key in source_backbone if key.startswith(unchanged_prefixes)
   ]
   assert unchanged_keys
   assert all(
       torch.equal(source_backbone[key], stage2_backbone[key])
       for key in unchanged_keys
   )

   top_block_keys = [
       key for key in source_backbone if key.startswith('encoder.layers.7.')
   ]
   assert top_block_keys
   assert any(
       not torch.equal(source_backbone[key], stage2_backbone[key])
       for key in top_block_keys
   )

   source_projector = source['projector_state_dict']
   stage2_projector = stage2['projector_state_dict']
   assert source_projector.keys() == stage2_projector.keys()
   projector_parameter_keys = [
       key
       for key in source_projector
       if key.endswith(('.weight', '.bias'))
   ]
   assert projector_parameter_keys
   assert any(
       not torch.equal(source_projector[key], stage2_projector[key])
       for key in projector_parameter_keys
   )

   optimizer_steps = {
       int(parameter_state['step'].item())
       if isinstance(parameter_state['step'], torch.Tensor)
       else int(parameter_state['step'])
       for parameter_state in stage2['optimizer_state_dict']['state'].values()
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
   print(f'BT100 -> BT25 continuation verified: {stage2_path}')
   PY
   ```

### 完了判定

GPU feasibilityではCUDA OOMが発生せず、training loss、gradient、projection、correlation metricsがfiniteであり、専用output rootに`latest.pt`が保存されることを確認する。

full run完了時は、`epoch = 25`、`global_step = 15,625`、`amp_enabled = false`であること、checkpoint configが`train.lr = 1.0e-5`および`continuation.unfreeze_top_blocks = 1`であることを確認する。`full_25ep/latest.pt`が存在し、全epochのtraining loss、gradient、projection、correlation metricsがfiniteでなければならない。

Stage 1 sourceと比較して、backboneのpatch projection、encoder layer 0〜6、mask token、encoder-to-decoder、MAE decoder、prediction headが不変であり、encoder layer 7とprojectorでは少なくとも1 parameterが変化していることを確認する。Stage 2 optimizerのstepは15,625であり、Stage 1 optimizer stateを引き継いでいないことも必要とする。
