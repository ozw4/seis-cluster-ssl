# Performance operations

Performance changes must preserve scientific identity, checkpoint compatibility,
and artifact contents. Use matched synthetic benchmarks where representative,
and targeted device or I/O measurements for changes they cannot assess.
Real-data and CUDA timings also include storage and device variability.

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

## Strat HMM training synchronization

The head-only, multi-head (hard labels and posterior targets), and center-trace
masked epoch loops group loss and gradient finite flags on the device before
reading one combined flag per device. Center-trace loss diagnostics and the
pre/post-clipping norm checks use the same grouping. Every tensor is still
checked each step; invalid losses fail before backward, and invalid gradients
or clipping norms fail before the optimizer update, including after AMP
unscaling. The head-only loop relies on the full loss check instead of checking
the total loss twice.

Hard multi-head and center-trace target preparation also groups the finite and
nonnegative checks for confidence and boundary weights. With three heads, the
12 individual host reads become one boolean-vector transfer per device. Checks
still inspect every weight, including masked-out tokens, in its original dtype
before conversion to the logits dtype. Pending failures take precedence over
later target-preparation errors, preserving the original head/field error order.
Label-range, shared-mask, and unity-boundary checks retain their existing order.
All target validation completes before computing the losses.

The hard multi-head loss computes mask emptiness once per head after student
validity and minimum-confidence filtering. Prototype loss, usage loss, and the
mean-confidence metric reuse that result within the current head and batch.
With both losses enabled, their nine mask reads across three heads become three.
Different filtered head masks, empty supervision, disabled losses, and the
graph-connected zero branches retain their original behavior. The masked-mean
arithmetic and loss/gradient accumulation order are unchanged.

Step metrics are detached, stacked by device and dtype, and copied to the host
once per group. Keeping dtype groups separate preserves scalar conversion even
for mixed precision and large integer counts. Metric keys, callback frequency,
Python-float accumulation order, epoch averages, and resume totals are unchanged.
Loss arithmetic, clipping calculations, model/optimizer state, random sampling,
configuration, and checkpoint formats are unchanged.

Input batches request the existing `move_batch_to_device(..., non_blocking=True)`
path, including nested per-head targets. The shared helper uses asynchronous
copies only for pinned tensors going to CUDA; CPU and unpinned inputs retain
their blocking fallback. The strat HMM dataloaders already enable pinning for
CUDA. Copies and consumers use the same stream, preserving dependencies.

Run the regression checks on CPU and, when available, CUDA:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  tests/seis_ssl_cluster/test_strat_hmm_synchronization.py \
  tests/seis_ssl_cluster/test_strat_hmm_target_validation.py \
  tests/seis_ssl_cluster/test_strat_hmm_mask_reuse.py
```

These checks compare parameters, optimizer/scaler state, per-step metrics, and
epoch summaries exactly against scalar host reads; they also cover NaN/Inf
rejection and grouped CUDA host reads. AMP comparisons cover head-only, hard
multi-head, and center-trace training. Posterior targets are checked in FP32:
with PyTorch 2.13, the existing posterior AMP path raises
`RuntimeError: Float did not match Half` in its target row-sum validation before
the epoch guards. This is also reproducible with the original epoch code.

The general synthetic benchmark does not measure these CUDA synchronization
costs. For timings, use matched inputs and saved pre-change epoch/loss code,
synchronize at timing boundaries, warm up both paths, and alternate their
measurement order. Check parameter, optimizer, and metric equality as well as
timing distributions. Small CUDA fixtures measure synchronization overhead;
real-data training throughput still needs measurement on the target workload.

## Sharing prototype usage entropy

Head-only, hard multi-head, posterior-target, and center-trace masked losses
reuse the usage entropy computed for the entropy-floor penalty as their
`prototype_usage_entropy` metric. With usage loss enabled, each head computes
its selected-token mean and entropy once instead of twice; K=(6, 8, 10) uses
three such reductions instead of six. Reuse stays within the same head and
batch, after the route's existing student-validity and confidence filtering.

`usage_entropy_floor_loss_with_entropy` returns `(loss, entropy)` and preserves
the original validation order, reduction order, dtype, epsilon, and gradient
path. The existing `usage_entropy_floor_loss` API still returns a scalar loss.
Disabled usage loss retains the metric-only calculation. Empty masks retain
their existing route-specific zero or error behavior. Configuration, sampling,
checkpoint formats, and target-entropy calculations are unchanged.

Regression checks compare loss and metric values, gradients, optimizer/scaler
state, model updates, and epoch metrics exactly on CPU and CUDA. They include
filtered masks that differ between heads, disabled losses, empty supervision,
strided probability tensors, and the existing AMP training paths (posterior
targets remain FP32 for the pre-existing limitation described above):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/seis_ssl_cluster/test_strat_hmm_entropy_reuse.py
```

The reduction in duplicate entropy work does not imply a corresponding
reduction in total training time. Measure the full target workload separately;
encoder computation, data loading, and other losses still contribute.

## Sharing the frozen strat HMM encoder prefix

Unmasked head-only and multi-head training (hard or posterior targets) compares
the student and teacher's input projection and leading frozen encoder blocks
once per epoch. When their parameter bytes, geometry, and execution settings
match, each batch runs that transformer prefix once and then executes the
student and teacher suffixes separately. Eight-layer encoders with only the top
student block trainable perform nine block forwards instead of sixteen. This
is a reduction in forward block calls, not a claim of 44% less training time.

The input projections remain separate to preserve each model's input validation
and position-cache lifecycle. The shared prefix uses `no_grad`, while the
student suffix retains its normal gradient path. Dtypes, attention arithmetic,
losses, optimizer groups, and checkpoint/configuration formats are unchanged.
No features are cached across batches. Center-trace replacement-masked training
continues to encode the different student and teacher inputs independently.

Sharing is restricted to the standard MAE/transformer implementations. Different
prefix weights, trainable input projection, non-evaluation prefix blocks,
incompatible position caches, custom modules/calls, and module hooks use the
original two-forward path. Inputs requiring gradients also use that path.
Parameter identities, storage/version counters, module settings, and cached
position tensor versions are checked before reuse; normal in-place changes,
checkpoint loads, or parameter replacement invalidate the epoch's sharing plan.
As with autograd, mutations through `.data` that bypass version counters are not
supported. A new epoch rebuilds the plan from the actual loaded model state.

Run the focused parity and fallback coverage with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/seis_ssl_cluster/test_strat_hmm_shared_prefix.py
```

The tests compare outputs, gradients, optimizer updates, model state, random
state, and epoch metrics exactly on CPU and CUDA, including the existing AMP
head-only and hard multi-head paths. They also check strict mask errors,
partial/full freezing, state changes, hooks, and exclusion of masked inputs.
Measure target-workload throughput separately; one-time equality checks and
per-batch guards are included in the implementation's cost.

Large CUDA attention backward passes can already vary between repetitions of
the original, unshared computation. Sharing does not enable deterministic
algorithms or alter CUDA backend settings. For a bitwise comparison of the
4,096-token training case, run the dedicated test with deterministic algorithms
and a cuBLAS workspace configured before starting Python:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  tests/seis_ssl_cluster/test_strat_hmm_shared_prefix.py::test_4096_token_cuda_updates_match_with_deterministic_algorithms
```

This CUDA-dependent slow test restores the prior deterministic-algorithm
settings when it finishes. Exact agreement under those settings does not promise
bitwise reproducibility for the existing nondeterministic CUDA execution mode.

## Embedding extraction prefetch

The K6810 multi-source extraction definitions for F3, Parihaka, and Volve use
`embedding.prefetch_queue_depth: 2`. The existing bounded FIFO producer reads
and preprocesses upcoming windows while the main thread performs inference
and merges results. Batch size stays at 1, and window order, merge order,
precision, and preprocessing settings retain their reference values. Set the
queue depth to 0 to restore synchronous extraction.

The experiment preflight allows only this nonnegative integer runtime setting
to differ from the historical embedding reference. It still rejects changes
to numerical settings and source lineage. Prefetch depth does not change
scientific output metadata, so matching completed outputs can still be reused.
The historical references and the F3 MAE read-only reuse arm are not modified.

Regression coverage compares embedding, validity-mask, and metadata file bytes
between queue depths 0 and 2 on CPU and CUDA, for float16/float32 output,
CUDA AMP on/off, and with/without window-local AGC. It also checks reuse of
completed outputs and producer/consumer failure cleanup:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/seis_ssl_cluster/test_embedding_extractor.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_definition.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_configs.py
```

The queue holds at most two prepared batches in addition to batches actively
processed by the producer/consumer. Measure throughput on the target workload:
overlap reduces serial waiting, but its benefit depends on CPU preprocessing,
storage, and GPU utilization. Keep batch size and precision fixed for parity
measurements; changing either is a separate optimization.

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
