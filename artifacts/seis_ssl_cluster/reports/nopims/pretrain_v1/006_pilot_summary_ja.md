# Issue 006 実行まとめ

- 状態: 完了
- 使用config: `experiments/nopims/pretrain_v1/10_pretrain/amp_mae_v1/02_pilot_1k.yaml`
- output root: `/workspace/artifacts/seis_ssl_cluster/runs/nopims/pretrain_v1/amp_mae_v1/pilot_1k`
- global step: 1000
- 最終loss: 0.9014941312372684（checkpoint/log の epoch 集計値）
- reconstruction loss: 0.8875221141576767（checkpoint/log の epoch 集計値）
- gradient loss: 0.2794403456710279（checkpoint/log の epoch 集計値）
- gradient norm: 0.04762365753389895（checkpoint/log の epoch 集計値）
- loss推移の所見: debug JSON で確認できる step 250/500/750/1000 の loss は 1.7070916891098022 -> 0.8112055063247681 -> 0.7940655946731567 -> 0.19579805433750153。確認可能な範囲では異常発散なし。debug 時点の最小 loss は 0.19579805433750153、最大 loss は 1.7070916891098022。
- checkpoint: `/workspace/artifacts/seis_ssl_cluster/runs/nopims/pretrain_v1/amp_mae_v1/pilot_1k/mae_latest.pt` を `torch.load(..., map_location="cpu")` で読み込み確認済み。`mae_epoch_0001.pt` も生成済み。checkpoint には `config`、`optimizer_state_dict`、`rng_state`、`training_state` があり、`training_state.schema_version=1`、`training_state.stage=train_amp_mae`、`training_state.checkpoint_kind=epoch`。
- debug visualization生成時点: step 250/500/750/1000 で xy/xz の png/json が生成済み。
- non-finite diagnostic: diagnostic/nonfinite 系ファイルなし。checkpoint metrics と debug JSON metrics はすべて finite。
- resumeを使用したか: いいえ。実行前に output root は存在せず、新規実行した。
- full学習へ進める判定: 進めてよい。1000 steps 完走、non-finite なし、loss の異常発散なし、checkpoint・可視化・snapshot は揃っている。
- 備考: dry-run 実施後に本実行を実施した。正式実行では CLI による `--max-steps`、`--output-root` の条件変更はしていない。要求の 1000 step と debug visualization step 250/500/750/1000 を満たすため、実行前に pilot YAML の `train.samples_per_epoch` を 4000、`visualization.mae_debug.enabled` を true、`visualization.mae_debug.every_steps` を 250 に修正した。pilot の manifest/split snapshot は clean manifest/split と SHA-256 が一致した。smoke と pilot の共通実験条件は、run 長・output root・可視化周期など意図的差分を除き一致した。ログ: `/workspace/artifacts/seis_ssl_cluster/logs/nopims/pretrain_v1/amp_mae_v1/pilot_1k.log`。
