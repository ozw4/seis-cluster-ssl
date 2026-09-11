# F3 Local Barlow Twins noise × rotation search v1

## 目的と範囲

実験 122 の Random-parity 候補から、noise distribution、片側だけを劣化させる
asymmetric noise、純回転と鏡映の違いを分離した探索である。粗い単一セル評価の後、
候補を canonical five-way と同じ layout / size 条件で比較し、学習期間による変化も
確認した。

## 系譜

比較元は実験 122 と `22_local_barlow_twins_v1`、下流条件は canonical five-way
v3 である。augmentation、region、epoch continuation は YAML と resolver が定義し、
config/implementation test はそれらを検証する。

## 結果

採択した処方と全条件の統計は
[`local_bt_rot90_asymmetric_noise_recipe.md`](../../../../reports/f3/local_bt_rot90_asymmetric_noise_recipe.md)、
学習期間の比較は
[`local_bt_epoch_scaling_v1.md`](../../../../reports/f3/local_bt_epoch_scaling_v1.md)
に分けて記録する。

3 epoch 採択処方を起点とする ordered stratigraphic HMM K=6 の 25 epoch
継続学習と、同じ 15 downstream cell での 3 epoch control 対比較は
[`local_bt_rot90_asym_g060_hmm_k6_25ep.md`](../../../../reports/f3/local_bt_rot90_asym_g060_hmm_k6_25ep.md)
に記録する。

短期 run、continuation、比較 cell は
[shared candidate runbook](../LOCAL_BT_CANDIDATE_RUNBOOK.md) の順序で実行する。

## HMM K=6 25 epoch 継続

HMM branch は `rot90_asym_g060_3ep` の checkpoint から専用 target embedding、
K=6 target、25 epoch checkpoint、v2 embedding を順に生成する。teacher と student
初期値は同じ 3 epoch checkpoint に固定し、HMM stage では encoder の最上位 block
1 層だけを更新する。control は既存の
`local_bt_nr_rot90_asym_g060_3ep` 15 cell を再利用する。

```bash
set -euo pipefail
export EXP=experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export F3_ROOT=/home/dcuser/data/public_data/field/F3

for flag in --dry-run ''; do
	python proc/seis_ssl_cluster/extract_embeddings.py \
		--config "$EXP/40_hmm_targets/rot90_asym_g060_3ep/01_extract_embeddings.yaml" \
		$flag
done
for flag in --dry-run ''; do
	python proc/seis_ssl_cluster/cluster_embeddings.py \
		--config "$EXP/40_hmm_targets/rot90_asym_g060_3ep/k6/02_cluster_hmm_k6.yaml" \
		$flag
done
bash "$EXP/40_hmm_targets/rot90_asym_g060_3ep/k6/03_export_pseudo_targets.sh" \
	--dry-run
bash "$EXP/40_hmm_targets/rot90_asym_g060_3ep/k6/03_export_pseudo_targets.sh"

for config in \
	"$EXP/50_hmm_pretraining/rot90_asym_g060_3ep/hmm/k6/01_gpu_feasibility_1step.yaml" \
	"$EXP/50_hmm_pretraining/rot90_asym_g060_3ep/hmm/k6/02_full_25ep.yaml"; do
	python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
		--config "$config" --dry-run
	python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
		--config "$config"
done

python - <<'PY'
from pathlib import Path

import torch

path = Path(
	'/workspace/artifacts/seis_ssl_cluster/pretraining/f3/facies_benchmark_v1/'
	'local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/hmm/k6/'
	'full_25ep/latest.pt'
)
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == 25
assert checkpoint['global_step'] == 15_625
assert checkpoint['training_state']['checkpoint_kind'] == 'epoch'
PY

for flag in --dry-run ''; do
	python proc/seis_ssl_cluster/extract_embeddings.py \
		--config "$EXP/60_hmm_embeddings/rot90_asym_g060_3ep_hmm_k6_25ep.yaml" \
		$flag
done

CONFIG="$EXP/70_hmm_downstream/rot90_asym_g060_3ep_hmm_k6_25ep.yaml"
for size in small medium large; do
	for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
		python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
			--config "$CONFIG" --layout "$layout" --size "$size" --dry-run
	done
done
for size in small medium large; do
	for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
		python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
			--config "$CONFIG" --layout "$layout" --size "$size"
	done
done

SUMMARY_CONFIG="$EXP/80_hmm_summary/01_hmm_k6_25ep_vs_rot90_asym_g060_3ep.yaml"
python proc/seis_ssl_cluster/summarize_f3_lithology_paired_candidates.py \
	--config "$SUMMARY_CONFIG" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_paired_candidates.py \
	--config "$SUMMARY_CONFIG"
```

中断した HMM full run だけは、自身の `full_25ep/latest.pt` を `--resume` に
渡して再開する。fresh HMM stage に 3 epoch checkpoint を `--resume` として渡さない。
