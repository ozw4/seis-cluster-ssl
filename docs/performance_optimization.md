# Performance operations

Performance changes must preserve scientific identity, checkpoint compatibility,
and artifact contents. Compare them with the synthetic benchmark before using
real-data or CUDA timings, which also include storage and device variability.

## Reproducible comparison

Run on an otherwise idle machine and keep the environment, seed, warm-up count,
repeat count, and benchmark case fingerprints fixed. Thread limits reduce noise
on shared hosts.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/benchmark_seis_ssl_cluster_performance.py \
  --seed 248 --warm-up 3 --repeat 20 \
  --output-json artifacts/performance/baseline.json \
  --output-markdown artifacts/performance/baseline.md

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/benchmark_seis_ssl_cluster_performance.py \
  --seed 248 --warm-up 3 --repeat 20 \
  --baseline-json artifacts/performance/baseline.json \
  --output-json artifacts/performance/candidate.json \
  --output-markdown artifacts/performance/candidate.md
```

Treat a comparison as invalid when case identity or input fingerprint differs.
Wide overlap between timing quartiles indicates that the run is too noisy for a
conclusion. There is intentionally no universal pass threshold; report the
measured distribution and the host context instead.

For a quick portability check:

```bash
python tools/benchmark_seis_ssl_cluster_performance.py --smoke
```

The benchmark's `--help` and implementation define the available cases and
report schema. Stage-specific timing fields are defined by the producing stage
and written with its artifacts.

## Cache operations

Caches trade repeated preprocessing for memory or disk use. Measure on the
target machine before enabling them for a final run. A cache is reusable only
when its recorded source and transformation identity matches the requested run;
an interrupted or mismatched cache must be rebuilt.

Before cleanup, stop jobs that may use the cache and resolve its concrete path.
Only remove a verified cache below the configured artifact root. Raw data,
checkpoints, final embeddings, and cluster labels are not caches. Prefer the
producer's configuration-driven cleanup and rebuild controls; their accepted
keys and defaults are defined by the resolver and example YAML, not this page.

Benchmark outputs follow the repository
[report sharing policy](report_sharing_policy.md).
