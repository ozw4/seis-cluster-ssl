# F3 Local BT `rot90_asym_g060_3ep` + ordered HMM K=6 25 epoch

- 日付: 2026-09-07
- 実験: [123 Local BT noise/rotation search](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/README.md)
- 状態: **FINAL**。手法、control、HMM target、candidate completion、下流 15 セル、
  artifact provenance を paired summary と live files から再検証済み。
- 対比較: `local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep` −
  `local_bt_nr_rot90_asym_g060_3ep`
- 主指標: unique validation voxels 上の `macro_f1`

この文書は人間向けのレビュー用レポートであり、pipeline input ではない。
数値は exact paired summary と live artifact identities から転記した。

## 固定した実験条件

control は純回転、片側だけへの Gaussian noise σ0.60、positive window
`[2,2,1]` で 3 epochs 学習した Local Barlow Twins である。採択根拠と
Random 比の結果は
[noise/rotation report](local_bt_rot90_asymmetric_noise_recipe.md)に記録している。
candidate はこの control checkpoint から作った専用 embedding を ordered
stratigraphic HMM K=6 でクラスタリングし、その hard pseudo-target を使って
25 epochs 学習する。

teacher と student の初期値は同じ control checkpoint に固定する。HMM stage で
更新する encoder は最上位 block 1 層だけで、prototype head は 6 prototypes、
projection dimension 128、temperature 0.1、L2 normalization を使う。loss weight は
prototype 1.0、usage 0.005、distillation 0.2。batch size 16、10,000 samples/epoch、
encoder/head learning rate `1e-5`、weight decay 0.05、seed 42 で、完了点は
epoch 25 / global step 15,625 とする。実行定義は
[`02_full_25ep.yaml`](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/50_hmm_pretraining/rot90_asym_g060_3ep/hmm/k6/02_full_25ep.yaml)
を正典とする。

### Ordered HMM K=6 target

| 項目 | 固定値 |
|---|---|
| target embedding | control 3 epoch、overlap `[64,64,64]` |
| clustering | `stratigraphic_hmm_kmeans`、K=6、L2、seed 42 |
| residualization | `local_token_position` / `token_phase`、global mean を戻す、最小 group count 32 |
| PCA | 64 components、whitening なし |
| HMM | 10 iterations、Z axis 2、increasing downward、edge margin `[8,8,0]` |
| transition cost | same 0.03、advance 0.0、jump 1.0、reverse `1e6`、reverse 禁止、max jump 1 |
| path prior | shallow/deep anchor weight 0.25、expected boundaries 無効 |
| initialization/update | `mean_z` / empty cluster は previous center を維持 |
| pseudo-target export | confidence 1.0、boundary alpha 0.0 / tau 1.0 |

生成済み target の token grid は `[76,113,32]`、有効 162,960 tokens、無効
111,856 tokens である。cluster counts は K0..K5 の順に
27,374 / 29,964 / 21,044 / 29,866 / 19,180 / 35,532 で、6 cluster はすべて
非空である。

### 下流契約

評価は canonical five-way v3 の `small` / `medium` / `large` ×
`layout_000`..`layout_004` の exact 15 cells を使う。encoder は凍結し、control と
candidate は同じ voxel supervision、validation mask、decoder 初期値、tile manifest、
50-epoch decoder 契約を共有する。decoder は
`frozen_embedding_decoder_nearest_voxel_ln_v1`、seed 42,000、440 steps/epoch である。

size ごとの paired t statistic は 5 layouts を統計単位にする。`all_15` は
layout × size cell の記述統計だけを示す。全体の paired t statistic は、各 layout
内で 3 sizes の差を平均した 5 layout means を統計単位にする。

## Paired 15-cell 結果

candidate、control、差分は完了済みの paired 15 jobs から再検証した
`macro_f1` である。

| size | layout | candidate | control | candidate − control |
|---|---|---:|---:|---:|
| small | layout_000 | `0.504364649` | 0.513537237 | `-0.009172588` |
| small | layout_001 | `0.449741428` | 0.427616588 | `+0.022124841` |
| small | layout_002 | `0.525055380` | 0.507696047 | `+0.017359333` |
| small | layout_003 | `0.568586095` | 0.564557037 | `+0.004029058` |
| small | layout_004 | `0.493991511` | 0.487793548 | `+0.006197963` |
| medium | layout_000 | `0.568604325` | 0.565661389 | `+0.002942936` |
| medium | layout_001 | `0.511133123` | 0.509442870 | `+0.001690254` |
| medium | layout_002 | `0.616251310` | 0.598918111 | `+0.017333199` |
| medium | layout_003 | `0.613967217` | 0.604680061 | `+0.009287156` |
| medium | layout_004 | `0.522540046` | 0.486501732 | `+0.036038314` |
| large | layout_000 | `0.673733123` | 0.689526378 | `-0.015793255` |
| large | layout_001 | `0.678106493` | 0.669301913 | `+0.008804580` |
| large | layout_002 | `0.667907908` | 0.647964589 | `+0.019943318` |
| large | layout_003 | `0.659314049` | 0.639339494 | `+0.019974555` |
| large | layout_004 | `0.616282940` | 0.589663429 | `+0.026619511` |

### Aggregates

| scope | statistical unit | n | candidate mean | control mean | delta mean | median delta | sample std | wins/ties/losses | paired t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | layout | 5 | `0.508347813` | 0.500240091 | `+0.008107721` | `+0.006197963` | `0.012257281` | `4/0/1` | `+1.4791` |
| medium | layout | 5 | `0.566499204` | 0.553040832 | `+0.013458372` | `+0.009287156` | `0.014065295` | `5/0/0` | `+2.1396` |
| large | layout | 5 | `0.659068903` | 0.647159161 | `+0.011909742` | `+0.019943318` | `0.016756006` | `4/0/1` | `+1.5893` |
| all_15 | layout × size cell | 15 | `0.577971973` | 0.566813361 | `+0.011158612` | `+0.009287156` | `0.013604500` | `13/0/2` | n/a |
| overall layout-clustered | layout mean across 3 sizes | 5 | n/a | n/a | `+0.011158612` | `+0.011096923` | `0.011523364` | `4/0/1` | `+2.1653` |

判定: `15セル平均では3 epoch controlを0.011159上回り13/15セルで改善したが、layout単位の両側paired t検定では5%水準の有意差には達しなかった。`

## Artifact provenance

### Versioned definitions

| 定義 | repository path | SHA-256 |
|---|---|---|
| canonical five-way v3 | [`60_five_way.yaml`](../../experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml) | `285b0233ff82fe83808f82e929b611f570a67f01fa983ef191dda23d1858061b` |
| 3 epoch Local BT | [`rot90_asym_g060_3ep.yaml`](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/10_pretraining/rot90_asym_g060_3ep.yaml) | `57ae0cd5ff52dc1afbe7f469dd303c3f7638564c439883d24bbab863d86b86b0` |
| HMM 25 epoch | [`02_full_25ep.yaml`](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/50_hmm_pretraining/rot90_asym_g060_3ep/hmm/k6/02_full_25ep.yaml) | `50bff466ffab658f9ce8ed27cc2f9d809e3b4bb544a0d883dd502b1749971a93` |
| candidate downstream | [`rot90_asym_g060_3ep_hmm_k6_25ep.yaml`](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/70_hmm_downstream/rot90_asym_g060_3ep_hmm_k6_25ep.yaml) | `eefa55b572ef5ea24382219431ddbfefd9c2011fcbd9795609c9a6adb9782498` |
| paired summary | [`01_hmm_k6_25ep_vs_rot90_asym_g060_3ep.yaml`](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/80_hmm_summary/01_hmm_k6_25ep_vs_rot90_asym_g060_3ep.yaml) | `0ce46e0606703a3353476f9ea3d2333e0c36e9c22d8d19b1f29ab08acd9f48cc` |

### Completed source and target artifacts

Paths below are relative to `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}` unless another variable is
shown.

| artifact | path | SHA-256 |
|---|---|---|
| control checkpoint | `pretraining/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/rot90_asym_g060/full_3ep/latest.pt` | `c77fa1b58f644641012d1e4cae19358a43c8e480707be3d7ddd8d224c81f7130` |
| control downstream embeddings | `embeddings/f3/facies_benchmark_v2/local_bt_noise_rotation_search_v1/local_bt_nr_rot90_asym_g060_3ep/overlap_x64/f3_facies_benchmark.embeddings.npy` | `2775466a7f19f1bb84737c088bca9780d7ada44b0e5fa3596ebbff8f598f9abb` |
| control downstream valid tokens | `embeddings/f3/facies_benchmark_v2/local_bt_noise_rotation_search_v1/local_bt_nr_rot90_asym_g060_3ep/overlap_x64/f3_facies_benchmark.valid_tokens.npy` | `3bfeb8db8a47420ae7671db90a7e4d6e5a07fceba27648ec76213df3c2b38fd7` |
| control downstream embedding metadata | `embeddings/f3/facies_benchmark_v2/local_bt_noise_rotation_search_v1/local_bt_nr_rot90_asym_g060_3ep/overlap_x64/f3_facies_benchmark.embedding_metadata.json` | `2512c5d041973c39c4d972ed9581fe4f2cda783e136fbedd88d4f64d2ab97313` |
| HMM target embedding metadata | `embeddings/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/hmm_targets/rot90_asym_g060_3ep/overlap_x64/f3_facies_benchmark.embedding_metadata.json` | `7b2fdf199f82fd491fb6043809215b7ec7f4c75f8433df7c92de498a73c9d622` |
| cluster labels | `clustering/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/hmm_targets/rot90_asym_g060_3ep/k6/labels/k6/f3_facies_benchmark.cluster_labels_token.npy` | `48fc5652aa6f8fa637abf71d6fb05d99c059c559e974577ee73cc6c1a8b9d94f` |
| cluster label metadata | `clustering/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/hmm_targets/rot90_asym_g060_3ep/k6/labels/k6/f3_facies_benchmark.cluster_label_metadata.json` | `781f34d2237f2863926bf028122797e5127b1060db9e95f1acda580d8939c6e5` |
| clustering metadata | `clustering/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/hmm_targets/rot90_asym_g060_3ep/k6/models/k6/clustering_metadata.json` | `74ec820bd2da48d866fcd7c311448676e4a3ac057e4f011ffc3057ecf91cefff` |
| pseudo-target metadata | `pseudo_targets/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/k6/f3_facies_benchmark.pseudo_target_metadata.json` | `2ff4034a3928e291468c0d959561ab49de9f29911975d40b76a566e2e76b983c` |
| pseudo labels | `pseudo_targets/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/k6/f3_facies_benchmark.hmm_labels_token.npy` | `48fc5652aa6f8fa637abf71d6fb05d99c059c559e974577ee73cc6c1a8b9d94f` |
| pseudo valid tokens | `pseudo_targets/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/k6/f3_facies_benchmark.valid_tokens.npy` | `b420c53b5cdc2f939d0399d349c712f5212fbf8d0d885ddae5cfe64c8e21e526` |
| pseudo confidence | `pseudo_targets/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/k6/f3_facies_benchmark.hmm_confidence_token.npy` | `ffc856ec3b615fecd1b6210b6006996ff70857b817540d2681597eb69dc88dc2` |
| pseudo boundary weight | `pseudo_targets/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/k6/f3_facies_benchmark.hmm_boundary_weight_token.npy` | `ffc856ec3b615fecd1b6210b6006996ff70857b817540d2681597eb69dc88dc2` |

Pseudo labels は cluster labels と byte-identical で、pseudo valid mask は
`cluster_labels >= 0` と array-equal である。target embedding metadata、cluster recipe、
HMM transition/path recipe、source labels、pseudo-target export、control checkpoint の
結合は live-file audit を通過した。

### Shared evaluation inputs

全 15 control jobs で evaluation voxel count 470,136、validation mask identity
`0a6d134e4ea276ea15c29381cc8a1dd85cbdd928ece3df6882aa06e324129c70`、
validation tile manifest identity
`1f7e24f1a68d0020567af2966c4ac1472335c85d59d08526c3a0174bdebdc65b`、
prediction valid-mask SHA
`a1e4ad9b2ea27d10dffa1d91a2c7a64251b198201af6b5167910f977f4d93088`
を共有する。

| ground-truth input | path | SHA-256 |
|---|---|---|
| label volume | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/registry/volumes/f3/facies_benchmark_v2/f3_facies_labels.npy` | `daf2b900a6c68cc1dc5864f5ef0a1bd527c48c9f29842453d0b889378b3bf09d` |
| class info | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/inspection/f3/facies_benchmark_v2/inventory/class_info.json` | `97ae41d06e57c5b2fed5a6005c7378a179e522fff27825327e9f857209ea9f01` |
| source label SEG-Y | `${F3_ROOT}/f3_labels.sgy` | `100667f54cf07a8283b8f14a96d90db5bf693c1fd09295e5f190a4cb8a34a139` |
| PNG label inventory | [`section_inventory_v2.csv`](../../experiments/f3/facies_benchmark_v2/10_prepare/section_inventory_v2.csv) | `dd218dd94db06440c5e3a59a59167e9c9c8db327970cb5fc411070065f738c20` |
| SEG-Y geometry | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/inspection/f3/facies_benchmark_v2/segy/segy_geometry.json` | `02e7d0f1b2b652207ef9e2f3b019dee53aa5868c8b2835c1140f67ae5a3ad23d` |

### Completion identities

| artifact | expected path | completion identity |
|---|---|---|
| candidate 25 epoch checkpoint | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/hmm/k6/full_25ep/latest.pt` | `0ab25d0545eea325f0a2fafb6276c2d9e92720abdf556beb24049d82e0852fa1` |
| candidate downstream embedding metadata | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/embeddings/f3/facies_benchmark_v2/local_bt_noise_rotation_search_v1/local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep/overlap_x64/f3_facies_benchmark.embedding_metadata.json` | `8f2c8c98d337b268571d2af0cc2d06b6e4c7c1e4e7b1d850384e0e2e7bdf5e98` |
| candidate downstream embeddings | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/embeddings/f3/facies_benchmark_v2/local_bt_noise_rotation_search_v1/local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep/overlap_x64/f3_facies_benchmark.embeddings.npy` | `2bf9b62a5457ee56261872bc321352a1f50aabe631d72c449b79b34a02318bf3` |
| candidate downstream valid tokens | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/embeddings/f3/facies_benchmark_v2/local_bt_noise_rotation_search_v1/local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep/overlap_x64/f3_facies_benchmark.valid_tokens.npy` | `3bfeb8db8a47420ae7671db90a7e4d6e5a07fceba27648ec76213df3c2b38fd7` |
| paired summary | `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/f3_lithology_benchmark/local_bt_noise_rotation_search_v1/paired_summary/local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep_vs_local_bt_nr_rot90_asym_g060_3ep/` | `comparison.csv SHA-256=95370c094af0eb8c0fcf88ef55501025f4c0ab9929087636d83dd822ed6aeb05; summary.json SHA-256=354baf87af8a15675aaa58f258b0f30bbc680d2f9880af75de29ef5f50722976; summary.md SHA-256=de7927082467c7cef660de9be509d158d92e74a81794fb3fe6eb9f831b0ab5a5` |

最終 paired `summary.json` と `comparison.csv` が、各 cell の metrics、evaluation
metadata、prediction metadata、prediction arrays、decoder completion、supervision、
ground-truth input の path/SHA を所有する。

## Provenance の Low 制約

現行 clustering metadata schema の `embedding_inputs` は target embedding、valid-token
mask、metadata の path と metadata SHA を記録するが、`embeddings.npy` と
`valid_tokens.npy` 自体の content SHA は記録しない。現在の audit は正しい絶対 path、
survey 名、ファイルの存在、embedding metadata SHA、control checkpoint path/SHA を
検証する。そのため、clustering 実行後に同じ path の embedding array が metadata を
変えずに置換された場合、生成時に読んだ array bytes との一致は暗号学的に再証明
できない。

この制約は clustering より上流の履歴再構成に限られる。cluster labels と
pseudo-target の label/valid identities、HMM candidate checkpoint が記録する
pseudo-target input identities、下流 15-cell の paired provenance は個別に SHA と
内容整合性を検証する。
