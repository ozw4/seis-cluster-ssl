# Local VICReg lithology extension v1

This directory adds a gated VICReg extension to the existing exact-five F3
lithology benchmark. It never changes or writes the canonical 75-job runs or
their summary. The final seven-way report reads those 75 jobs in place and
combines them with exactly 30 new jobs (2 VICReg arms x 5 layouts x 3 sizes).

The medium-only screen is intentionally separate. It compares the existing
100-epoch VICReg checkpoint with the canonical random baseline across the five
medium layouts. Full extraction and the 30-job extension stay blocked until the
screen summary reports `VICREG_BASELINE_GATE_PASS`.

## Environment and dry runs

Set the same roots used by the canonical benchmark, then resolve every command
before launching work:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/absolute/path/to/artifacts
export F3_ROOT=/absolute/path/to/f3

CONFIG="$SEIS_SSL_CLUSTER_WORKSPACE/experiments/f3/facies_benchmark_v2/116_local_vicreg_extension_v1/60_extension.yaml"
SCREEN_LOG_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_vicreg_screen_v1/job_logs"
EXTENSION_LOG_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_vicreg_extension_v1/job_logs"
mkdir -p "$SCREEN_LOG_ROOT" "$EXTENSION_LOG_ROOT"

python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode screening-source --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --suite screening --model local_vicreg_100 \
  --layout layout_000 --size medium --dry-run
```

## 1. Audit the screening source

The source audit verifies the checkpoint identity, SHA-256 lineage, v2
embedding metadata and geometry, and exact valid-token-mask identity against
the canonical random source. It also audits the canonical exact-five sources.

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode screening-source
```

## 2. Run the five-layout medium screen

Run in layout-then-model order. `random` resolves to the already completed
canonical random job and is inspection-only; the runner will never write it.
Existing completed candidate jobs are identity-checked and skipped.

```bash
set -o pipefail
for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  for model in local_vicreg_100 random; do
    python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
      --config "$CONFIG" --suite screening --model "$model" \
      --layout "$layout" --size medium 2>&1 \
      | tee "$SCREEN_LOG_ROOT/${layout}_${model}_medium.log"
  done
done

python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --mode screening
```

The gate uses paired `macro_f1` deltas defined as
`local_vicreg_100 - random`. It passes only when the five-layout mean and median
are both positive and at least three layouts are wins. The summary is written
atomically to
`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/f3_lithology_benchmark/local_vicreg_screen_v1/summary/summary.json`.
Do not delete or overwrite an existing summary; investigate any identity or
matrix failure and use a new output root for a new attempt.

### Interrupted-decoder recovery

Resume only an interrupted decoder for the exact same suite/model/layout/size
cell. Pass that cell's own `decoder/latest.pt`; never use a checkpoint from a
different cell. Completed jobs are identity-checked and skipped without
`--resume`.

```bash
SCREEN_RESUME="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_vicreg_screen_v1/runs/model=local_vicreg_100/layout=layout_000/size=medium/decoder/latest.pt"
python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --suite screening --model local_vicreg_100 \
  --layout layout_000 --size medium --resume "$SCREEN_RESUME"

EXTENSION_RESUME="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_vicreg_extension_v1/runs/model=local_vicreg/layout=layout_000/size=small/decoder/latest.pt"
python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --suite extension --model local_vicreg \
  --layout layout_000 --size small --resume "$EXTENSION_RESUME"
```

## 3. Extract the two extension embeddings

Only after the gate passes, dry-run and then execute both fixed-geometry v2
extractions:

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$SEIS_SSL_CLUSTER_WORKSPACE/experiments/f3/facies_benchmark_v2/116_local_vicreg_extension_v1/50_embeddings/01_extract_local_vicreg.yaml" \
  --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$SEIS_SSL_CLUSTER_WORKSPACE/experiments/f3/facies_benchmark_v2/116_local_vicreg_extension_v1/50_embeddings/02_extract_local_vicreg_hmm_k6.yaml" \
  --dry-run

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$SEIS_SSL_CLUSTER_WORKSPACE/experiments/f3/facies_benchmark_v2/116_local_vicreg_extension_v1/50_embeddings/01_extract_local_vicreg.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$SEIS_SSL_CLUSTER_WORKSPACE/experiments/f3/facies_benchmark_v2/116_local_vicreg_extension_v1/50_embeddings/02_extract_local_vicreg_hmm_k6.yaml"

python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --mode sources
```

## 4. Preflight and run the exact 30-job extension

Preflight both arms on `layout_000/small` before the complete size-then-layout-
then-model matrix. Completed jobs are source- and identity-checked before they
are skipped.

```bash
set -o pipefail
for model in local_vicreg local_vicreg_hmm_k6; do
  python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
    --config "$CONFIG" --suite extension --model "$model" \
    --layout layout_000 --size small --dry-run 2>&1 \
    | tee "$EXTENSION_LOG_ROOT/layout_000_${model}_small_dry_run.log"
  python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
    --config "$CONFIG" --suite extension --model "$model" \
    --layout layout_000 --size small 2>&1 \
    | tee "$EXTENSION_LOG_ROOT/layout_000_${model}_small.log"
done

for size in small medium large; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    for model in local_vicreg local_vicreg_hmm_k6; do
      python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
        --config "$CONFIG" --suite extension --model "$model" \
        --layout "$layout" --size "$size" 2>&1 \
        | tee "$EXTENSION_LOG_ROOT/${layout}_${model}_${size}.log"
    done
  done
done
```

## 5. Produce extension and seven-way summaries

Both commands require exact, duplicate-free matrices and current checkpoint,
embedding, decoder, metrics, supervision, validation, and decoder-initial-state
identities. Reports are created atomically and refuse overwrite.

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --mode extension --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --mode extension

python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --mode combined --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
  --config "$CONFIG" --mode combined
```

The combined report has exactly 105 rows: the canonical five models plus
`local_vicreg` and `local_vicreg_hmm_k6`, for all three sizes and five layouts.
It is written only under `combined_seven_way_summary`; the canonical five-way
summary remains untouched and keeps its original exact-five API and results.
