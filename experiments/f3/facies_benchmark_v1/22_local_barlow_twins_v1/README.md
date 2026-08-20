# F3 local Barlow Twins v1

F3 の未ラベル振幅 volume から、同じ物理 token の対応だけを使って
`local_barlow_twins_3d` を事前学習し、frozen full-volume token embedding を
抽出する実験です。

比較元は
`experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/10_stage1/barlow_twins/`
です。学習設定の差分は `barlow_twins.method: local_barlow_twins_3d`、
`barlow_twins.local_pairs_per_crop: 128`、および衝突しない出力先だけです。
crop、model、augmentation、optimizer、学習予算は比較元と同一です。抽出時は
projector を使わず、bare `AmplitudeMAE3D.encode_tokens()` の encoder 次元を
そのまま出力します。

リポジトリ root で artifact root と実験ディレクトリを設定します。

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export LOCAL_BT_CONFIG_DIR=experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1
```

Feasibility config の dry-run と 1 step 実行:

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$LOCAL_BT_CONFIG_DIR/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$LOCAL_BT_CONFIG_DIR/01_gpu_feasibility_1step.yaml"
```

100 epoch config の dry-run と実行:

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$LOCAL_BT_CONFIG_DIR/02_full_100ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$LOCAL_BT_CONFIG_DIR/02_full_100ep.yaml"
```

Frozen embedding extraction の dry-run と実行:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$LOCAL_BT_CONFIG_DIR/03_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$LOCAL_BT_CONFIG_DIR/03_extract_embeddings.yaml"
```

Full checkpoint の method と resolved local 条件は読み取り専用で確認できます。

```bash
export LOCAL_BT_CHECKPOINT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pretraining/f3/facies_benchmark_v1/local_barlow_twins_v1/full_100ep/latest.pt"
python -c 'import json, os, torch; p=torch.load(os.environ["LOCAL_BT_CHECKPOINT"], map_location="cpu", weights_only=False); print(p["pretraining_method"]); print(json.dumps(p["config"], indent=2, sort_keys=True))'
```

抽出後の metadata では method、encoder feature 次元、pair 数を確認できます。

```bash
export LOCAL_BT_METADATA="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/f3/facies_benchmark_v1/local_barlow_twins_v1/overlap_x64/f3_facies_benchmark.embedding_metadata.json"
python -c 'import json, os; m=json.load(open(os.environ["LOCAL_BT_METADATA"], encoding="utf-8")); o=m["pretraining_objective"]; print(o["method"], m["model_geometry"]["encoder_dim"], o["local_pairs_per_crop"])'
```
