# F3 voxel section-layout v2

F3 five-way v2 用の model 非依存 section-layout supervision を生成します。v2 は位置だけで section を分割し、stable-hash 選択で旧 benchmark の教師量へ校正する layout です。line inventory、layout、target、selection と validation 契約は [01_prepare_section_layout_contract.yaml](01_prepare_section_layout_contract.yaml)、[02_layout_lines.yaml](02_layout_lines.yaml)、[03_build_section_layout_datasets.yaml](03_build_section_layout_datasets.yaml) を正本とします。

教師量の由来は legacy の [paired metrics](../../../../reports/f3/legacy/facies_benchmark_v1/voxel_lithology_label_budget_v1/tables/paired_metrics.csv) です。この report は provenance であり、実行時入力ではありません。

## 実行

再現が必要な場合は、[current v3 layout runbook](../109_f3_voxel_section_layout_v3/README.md)
の inspection、finalize、dataset-build 順序を、この directory の3つの YAML に
適用します。生成物の実測値と再生成可否は finalized contract、dataset manifest、
producer validation を正本とし、この履歴文書には複製しません。
