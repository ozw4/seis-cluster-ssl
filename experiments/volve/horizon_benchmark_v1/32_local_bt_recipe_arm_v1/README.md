# Volve horizon Local Barlow Twins recipe arms v1

F3 で得た Local Barlow Twins の処方が Volve のホライゾン推定にも移送できるかを、
同じ Random encoder と対比較する実験である。元になった F3 の結果と解釈は
[`local_bt_rot90_asymmetric_noise_recipe.md`](../../../../reports/f3/local_bt_rot90_asymmetric_noise_recipe.md)
を参照する。

この実験は事前学習 recipe だけを変え、Volve の decoder、split、tile、seed、
embedding 抽出、評価 support は five-way benchmark と共有する。arm の一覧と正確な
差分は `10_pretraining/`、`20_embeddings/`、`30_downstream/` の YAML が正本である。
checkpoint と YAML の recipe が一致することは下流 runner が実行前に検証する。

## 実行順

shared [Volve benchmark environment](../README.md) を設定する。各 write command
には、対応 CLI が提供する `--dry-run` を先に適用する。

```bash
export EXP="$VOLVE_EXP/32_local_bt_recipe_arm_v1"
```

事前学習と embedding 抽出は同名の config を順に使う。

```bash
for config in "$EXP"/10_pretraining/*.yaml; do
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$config"
done

for config in "$EXP"/20_embeddings/*.yaml; do
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config" --device cuda --skip-existing
done
```

下流は one-cell CLI で実行する。まず各 arm の dry-run で source、recipe、split を
監査する。

```bash
for config in "$EXP"/30_downstream/*.yaml; do
  arm=$(basename "$config" .yaml)
  python proc/seis_ssl_cluster/run_volve_horizon_recipe_arm.py \
    --config "$config" --model "$arm" \
    --layout layout_000 --size small \
    --layout-config "$VOLVE_LAYOUTS" --dry-run
done
```

全 arm が同じ runs root と Random source を共有するため、Random の各 cell は
いずれか 1 つの downstream config で一度だけ作る。その後、各 arm の cell を作る。

```bash
set -- "$EXP"/30_downstream/*.yaml
baseline_config=$1

for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  for size in small medium large; do
    python proc/seis_ssl_cluster/run_volve_horizon_recipe_arm.py \
      --config "$baseline_config" --model random \
      --layout "$layout" --size "$size" \
      --layout-config "$VOLVE_LAYOUTS" --device cuda
  done
done

for config in "$EXP"/30_downstream/*.yaml; do
  arm=$(basename "$config" .yaml)
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for size in small medium large; do
      python proc/seis_ssl_cluster/run_volve_horizon_recipe_arm.py \
        --config "$config" --model "$arm" \
        --layout "$layout" --size "$size" \
        --layout-config "$VOLVE_LAYOUTS" --device cuda
    done
  done
done
```

各 arm は completeness audit を通してから集計する。判定値と成果物形式は summary
実装を正本とする。

```bash
for config in "$EXP"/30_downstream/*.yaml; do
  python proc/seis_ssl_cluster/summarize_volve_horizon_recipe_arm.py \
    --config "$config" --check-only
  python proc/seis_ssl_cluster/summarize_volve_horizon_recipe_arm.py \
    --config "$config"
done
```

実験結果の解釈を追跡する場合は、
[`report sharing policy`](../../../../docs/report_sharing_policy.md)に従って
`reports/volve/` に 1 つだけ置く。本実験の公開ファイルは
[`local_bt_recipe_arm_v1.md`](../../../../reports/volve/horizon_benchmark_v1/local_bt_recipe_arm_v1.md)
である。
