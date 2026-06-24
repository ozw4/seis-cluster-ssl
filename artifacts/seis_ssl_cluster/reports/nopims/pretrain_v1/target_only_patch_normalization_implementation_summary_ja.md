# Target-only Patch-wise Target Normalization 実装まとめ

## 変更したconfig契約

`loss.target_normalization`をtraining YAMLの必須項目として追加しました。

- `mode: none`: 従来挙動。`eps`と`min_std`を書いた場合はvalidationで拒否します。
- `mode: patch_zscore`: `eps`と`min_std`が必須です。どちらもfiniteかつ0より大きい必要があります。
- `patch_zscore`では`loss.gradient_weight`は必ず`0.0`です。

`proc/configs/seis_ssl_cluster/train_amp_mae.yaml`は従来挙動として`mode: none`を明示します。

## target-onlyであること

encoder入力`x`とDatasetの`target`はこれまで通りsurvey-wise正規化済み振幅のままです。patch-wise normalizationはloss内部でpatch化されたtargetにだけ適用されます。predictionはpatch z-score空間を予測します。

## normalizationの計算式

patchごとにvalid voxelのみを使って次を計算します。

```text
mean = sum(target * valid) / max(valid_count, 1)
var = sum(((target - mean) * valid)^2) / max(valid_count, 1)
std = sqrt(var + eps)
std_eff = max(std, min_std)
target_for_loss = (target - mean) / std_eff
```

分散はpopulation varianceです。fully invalid patchは`mean=0`, `std_eff=1`, `target_for_loss=0`としてfiniteに保ちます。

## valid maskの扱い

mean/variance計算は常に`local_valid_mask == true`のvoxelだけを使います。loss selectionは従来通り`spatial_mask AND local_valid_patch_voxels`です。invalid voxelは統計から除外され、normalized targetでは0になります。

## gradient lossとの制約

v1では`patch_zscore`と現行gradient lossの併用を拒否します。理由はpredictionがpatch z-score空間になり、既存gradient lossがsurvey-wise正規化振幅空間のtarget gradientと比較するためです。

## debug visualizationの変更

`patch_zscore`時はtarget patch statisticsをlossと同じhelperで計算し、predictionを`pred * patch_std_eff + patch_mean`でsurvey-wise正規化振幅空間へoracle denormalizeしてから表示・abs error計算します。JSON metadataには`prediction_space`, `target_normalization_mode`, `oracle_target_statistics_used_for_denormalization`を記録します。

## checkpoint/resumeの後方互換

新規checkpoint/resolved configには`loss.target_normalization`が保存されます。resume compatibilityは`mode`, `eps`, `min_std`差分を検出します。legacy checkpointに`loss.target_normalization`がない場合は読み取り時のcompatibility viewで`mode: none`として扱うため、現在configが`mode: none`ならresume可能です。legacy checkpointから`patch_zscore` configへのresumeは拒否されます。

## embedding extractionへの影響

embedding extractionはencoder出力だけを使うため、target normalizationを入力へ適用しません。patch_zscore checkpointはvalidatorで受け入れ、embedding metadataへcheckpoint-ownedな`pretraining_objective`として`reconstruction`, `gradient_weight`, `target_normalization`を保存します。

## 追加した実験YAML

`experiments/nopims/pretrain_v1/10_pretrain/amp_mae_m025_mse_g0_patchnorm_v1/03_full_100ep.yaml`を追加しました。主要設定は`masking.spatial_mask_ratio: 0.25`, `loss.reconstruction: mse`, `loss.gradient_weight: 0.0`, `loss.target_normalization.mode: patch_zscore`, `eps: 1.0e-6`, `min_std: 0.05`です。学習は開始していません。

## 実行したtestと結果

- `PYTHONPATH=/workspace/src pytest -q tests/seis_ssl_cluster/test_mae_losses.py tests/seis_ssl_cluster/test_config.py tests/seis_ssl_cluster/test_training_smoke.py tests/seis_ssl_cluster/test_mae_debug_visualization.py tests/seis_ssl_cluster/test_embedding_extractor.py`: 217 passed
- `PYTHONPATH=/workspace/src pytest -q tests/seis_ssl_cluster/test_proc_dry_run.py tests/seis_ssl_cluster/test_end_to_end_mvp.py tests/seis_ssl_cluster/test_checkpoint.py`: 31 passed
- `PYTHONPATH=/workspace/src python proc/seis_ssl_cluster/train_amp_mae.py --config experiments/nopims/pretrain_v1/10_pretrain/amp_mae_m025_mse_g0_patchnorm_v1/03_full_100ep.yaml --dry-run`: passed, training skipped
- `PYTHONPATH=/workspace/src python proc/seis_ssl_cluster/train_amp_mae.py --config experiments/nopims/pretrain_v1/10_pretrain/amp_mae_v1/03_full_100ep.yaml --dry-run`: passed, training skipped
- `PYTHONPATH=/workspace/src pytest -q tests`: 372 passed

## 未解決事項

学習実行、既存run/checkpointの書き換え、gradient lossのpatch-zscore対応、入力側patch normalizationは行っていません。
