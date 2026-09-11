# F3 local Barlow Twins v1

This experiment pretrains Local Barlow Twins from corresponding physical tokens
in the unlabeled F3 amplitude volume, then extracts frozen full-volume token
embeddings.

Run feasibility, full pretraining, and embedding extraction in numbered order:

```bash
export EXP=experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/02_full_100ep.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/03_extract_embeddings.yaml" --skip-existing
```

The extraction provenance file is
`f3_facies_benchmark.embedding_metadata.json`; its configured directory belongs
outside the repository.
