# F3 facies benchmark v2

`facies_benchmark_v2`は、F3岩相推定の五者比較（MAE / MAE+HMM-K6 /
Local Barlow Twins / Local BT+HMM-K6 / Random）を、section本数で定義した
data size（small 1+1 / medium 2+2 / large 4+4 lines）と、v2 candidate statistics
から新規に決めたline配置・教師量で実行するartifact namespaceである。

v1（`../facies_benchmark_v1`）は実装・前処理・decoder・checkpoint・評価方法の
参照元であり、v2のline配置・教師量の正解ではない。前処理条件はv1と同じで、
出力pathとdataset versionだけがv2である。

| Stage | Directory | Producer |
|---|---|---|
| raw inspection | `00_inspection/` | `inspect_f3_*`, `visualize_f3_quicklook`, `check_f3_label_consistency`, `preview_f3_tokenization` |
| prepared volume / canonical supervision | `10_prepare/` | `prepare_f3_facies_volume.py`, `build_f3_lithology_voxel_dataset.py` |
| section-layout contract / 15 datasets | `109_f3_voxel_section_layout_v2/` | `prepare_f3_lithology_voxel_section_layout_contract.py`, `build_f3_lithology_voxel_section_layout_datasets.py` |
| five-way comparison | `110_lithology_mae_local_bt_five_way_v2/` | `extract_embeddings.py`, `audit_f3_lithology_five_way_sources.py`, `run_f3_lithology_five_way.py`, `summarize_f3_lithology_five_way.py` |

全手順は
[`110_lithology_mae_local_bt_five_way_v2/README.md`](110_lithology_mae_local_bt_five_way_v2/README.md)
のrunbookに順番どおり記載する。configは`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}`、
`${F3_ROOT}`、`${SEIS_SSL_CLUSTER_WORKSPACE}`で場所を指定する。
