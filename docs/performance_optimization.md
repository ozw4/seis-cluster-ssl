# Performance optimization and operations

The optimized paths preserve the existing model, checkpoint, config, and
artifact contracts. Use the synthetic CPU benchmark to separate code changes
from CUDA, SEGY, storage, and real-data variability. Importing the benchmark or
timing utilities does not start work.

## Optimization settings

The defaults favor compatibility and bounded resource use. Increase concurrency
or persistent disk caching only after measuring on the target machine.

| Area | Default | Recommended use and trade-off |
|---|---|---|
| NPY volume reads | Process-local LRU, 8 open volumes | Keep enabled for repeated crops. More open mappings reduce reopen cost but consume file descriptors and virtual address space. In-bounds crops remain read-only views; padded crops allocate. |
| Spatial masking | Automatic block-1 vectorized path | Use `(1, 1, 1)` when independent token masking is intended. Other block sizes retain general block semantics. |
| Amplitude preprocessing | `finite_check_mode: strict` | Use `strict` for validation and reproducibility. `output_only` or `off` removes scans but should be used only when upstream finiteness is guaranteed. Normalization and zero-mask work buffers never modify source memmaps. |
| Crop admission | Cheap validity phase before normalization/AGC | Automatic. Raising `min_valid_fraction` can avoid expensive work on poor crops but can increase retries and data-selection bias. |
| MAE batch/loss memory | No duplicate full target or visible mask; one patchify | Automatic. The reconstruction target is derived from `x`; checkpoint parameter keys and shapes are unchanged. |
| Position/visible tokens | Bounded position cache and equal-visible-count vectorization | Automatic. Cache entries are device/dtype specific and are not in `state_dict`; variable visible counts use the general padded path. |
| Runtime checks | `train.runtime_check_mode: once` | Use `strict` while diagnosing data or numeric failures, `once` for normal training, and `minimal` only with trusted inputs. Non-finite loss/gradient protection remains active. |
| Training input pipeline | `num_workers: 8`, `prefetch_factor: 2`, `persistent_workers: true` | Tune workers to storage and CPU capacity. Prefetch and pinned CUDA transfers improve overlap but increase host memory. With `num_workers: 0`, worker prefetch/persistence are not used. |
| Training precision | `amp: false`, `amp_dtype: auto`, timing off | FP32 is the compatibility baseline. On supported CUDA hardware, enable AMP and prefer `auto`/BF16; FP16 uses a scaler. AMP can produce finite, acceptably close results that are not bitwise identical. |
| Embedding extraction | `batch_size: 1`, `prefetch_queue_depth: 0`, `amp: false` | Increase batch size first, then queue depth while watching CPU/GPU memory. Queue depth 0 is the synchronous rollback setting. CUDA transfer and autocast follow device support. |
| Survey preprocessing cache | `mode: off`, `chunk_size_x: 16`, `reuse: true`, `cleanup: false` | Use `memmap` for overlapping windows on large surveys and `memory` only when the prepared survey fits comfortably in RAM. Disk mode exchanges repeated normalization for temporary disk capacity and I/O. Unsupported window-local settings stay on the uncached path. |
| Embedding merge/reconstruction | Average and token-axis chunk size 16 | Increase chunks when memory is available; reduce them to bound temporary arrays. Vectorized merge and reconstruction preserve invalid fills, integer dtype, and non-divisible volume boundaries. |
| Clustering input/residualization | Process-local LRU, 16 arrays; residual fit chunks 65,536 tokens | The dense token-phase representation avoids group scans. More cached arrays use more file descriptors; smaller chunks reduce peak memory. Legacy sparse residualizer artifacts remain readable. |
| Multi-k k-means output | Shared feature pass; bounded staged outputs | Request all required `k_values` together so read/residualize/PCA work is shared. Labels and model files publish only after the whole staged set succeeds. |
| HMM prepared features | 65,536-token chunks, `reuse: true`, `force_rebuild: false`, persistent by default | Persist for repeated HMM iterations/runs when disk permits. Set `cleanup: true, persist: false` for disposable caches. Fingerprints cover sources and transforms; partial or mismatched caches are rebuilt. |
| HMM emission/update | Matrix-form squared distance and chunked center sums | Automatic. It avoids the `[T,K,D]` temporary and cluster-by-token scans. Float32 round-off is clipped only when needed, so very small numeric differences from a broadcast reference can occur. |

`stage_timing` is off by default for training, embedding, and clustering. Enable
it for diagnosis, then disable it for the final throughput measurement. Timers
cover data wait, transfers, model/loss, optimizer, preprocessing, prediction,
write, HMM emission, Viterbi, and center update where applicable.

## Capture and compare benchmark reports

Run from the repository root on an otherwise idle machine. Pin thread counts
when repeatability matters:

```bash
mkdir -p artifacts/performance
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 \
  python tools/benchmark_seis_ssl_cluster_performance.py \
  --seed 248 --warm-up 3 --repeat 20 \
  --output-json artifacts/performance/baseline.json \
  --output-markdown artifacts/performance/baseline.md
```

The report covers memmap reads, block-1 masking, amplitude preprocessing,
position/visible selection, merge/reconstruction, residualization, and HMM
emission. It records input conditions, shape, case version, input fingerprint,
median, p25/p75, environment, and the Git commit when available. Keep reports
under ignored `artifacts/`; do not commit them.

Capture a candidate and compare it in one command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 \
  python tools/benchmark_seis_ssl_cluster_performance.py \
  --seed 248 --warm-up 3 --repeat 20 \
  --baseline-json artifacts/performance/baseline.json \
  --output-json artifacts/performance/candidate.json \
  --output-markdown artifacts/performance/candidate.md
```

Speedup is `baseline median / current median`. The JSON and Markdown omit the
multiplier when the case name, case version, or input fingerprint differs, or
when the current median is zero. A shape or environment difference is also a
reason to rerun under matching conditions. Large p25/p75 overlap usually means
the timing is too noisy for a conclusion. There is deliberately no time
threshold in tests. The portable smoke command is:

```bash
python tools/benchmark_seis_ssl_cluster_performance.py --smoke
```

## Stage timing

`seis_ssl_cluster.utils.StageTimer` is an opt-in wall-clock accumulator:

```python
from seis_ssl_cluster.utils import StageTimer

timer = StageTimer(enabled=collect_timings)
with timer.stage('load', sample_count=batch_size):
	batch = load_batch()
with timer.stage('model', sample_count=batch_size):
	outputs = model(batch)

timer.write_json('artifacts/performance/stages.json')
```

Nested stages are paths such as `model/encoder`. Failed contexts are counted
before the exception propagates. A zero sample count yields `null` seconds per
sample. When disabled, the timer does not read the clock, synchronize, log, or
create accumulator records.

## Cache cleanup and rollback

Before cleanup, stop jobs using the cache and inspect the configured directory.
Embedding preprocessing caches live below the configured
`embedding.preprocessing_cache.directory` (or the extraction output's cache
area). HMM prepared features live below
`clustering.stratigraphic_hmm.prepared_feature_cache.directory` or the
clustering output's prepared-feature area. Only remove a cache after confirming
its path is under `artifacts/`; raw data, checkpoints, final embeddings, and
cluster labels are not caches.

Normal cleanup is configuration-driven:

```yaml
embedding:
  preprocessing_cache:
    mode: memmap
    reuse: false
    cleanup: true

clustering:
  stratigraphic_hmm:
    prepared_feature_cache:
      reuse: false
      force_rebuild: true
      cleanup: true
      persist: false
```

For a no-cache rollback, set embedding `preprocessing_cache.mode: off`,
`prefetch_queue_depth: 0`, `batch_size: 1`, and `amp: false`; set training
`amp: false`, `runtime_check_mode: strict`, and `stage_timing: false`. For HMM,
use a disposable prepared cache (`cleanup: true`, `persist: false`) rather than
reusing old prepared features. Atomic writers reject partial artifacts and
fingerprint mismatches; do not rename `.partial`, `.tmp`, or interrupted cache
directories into final locations manually.
