# Performance baselines and stage timing

Use the synthetic benchmark before and after an optimization to separate code
performance from CUDA, SEGY, storage, and real-data variability. It exercises
the core memmap, masking, preprocessing, positional-embedding, merge,
token-to-voxel, and residualization paths on CPU. Benchmarking starts only when
the CLI is invoked; importing the module does not execute it.

## Capture a baseline

Run from the repository root in the same environment, on an otherwise idle
machine. Pin thread-count environment variables when repeatability across runs
matters:

```bash
mkdir -p artifacts/performance
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 \
  python tools/benchmark_seis_ssl_cluster.py \
  --seed 248 --warm-up 3 --repeat 20 \
  --output-json artifacts/performance/baseline.json
```

The output records the seed, warm-up and measured repeat counts, environment,
case shapes, median, p25, and p75 wall-clock seconds. It also records the Git
commit when Git is available. Inputs are generated synthetically from the seed;
the case order, names, and logical shapes are stable for the same arguments.
Keep measured JSON under ignored `artifacts/`; do not commit it.

## Compare a change

Capture the candidate with the same command and machine conditions, changing
only the output path. Compare each case's median and use the p25/p75 interval to
identify noisy results:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 \
  python tools/benchmark_seis_ssl_cluster.py \
  --seed 248 --warm-up 3 --repeat 20 \
  --output-json artifacts/performance/candidate.json
```

For each case, calculate
`(candidate median / baseline median - 1) * 100`. A negative percentage is an
improvement. Re-run results whose interquartile ranges overlap substantially or
whose environment and shape fields differ. This benchmark deliberately has no
pass/fail time threshold because shared runners and developer machines have
different performance characteristics.

## Instrument pipeline stages

`seis_ssl_cluster.utils.StageTimer` is an opt-in wall-clock accumulator. Keep it
disabled by default in normal execution:

```python
from seis_ssl_cluster.utils import StageTimer

timer = StageTimer(enabled=collect_timings)
with timer.stage('load', sample_count=batch_size):
	batch = load_batch()
with timer.stage('model', sample_count=batch_size):
	outputs = model(batch)

timer.write_json('artifacts/performance/stages.json')
```

Nested stages are reported as paths such as `model/encoder`. Failed contexts are
timed and counted before the exception propagates. A zero `sample_count` is
valid and produces `null` for `seconds_per_sample`. When disabled, entering a
stage does not read the clock, synchronize, log, or add an accumulator record.
The clock and optional synchronization callback can be injected for tests or
device-specific timing.
