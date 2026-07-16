# Issue #259 embedding vectorization benchmark

The repository synthetic benchmark was run on the parent of issue #259
(`649f2165a1ce4584560ff67e82e8f143caacf4dd`) and on the issue #259 change
(`b34e3a73cae21edaa44698471335d67a7e557c43`). The later batch-review commits
do not modify the benchmarked merge or reconstruction implementation.

Command:

```text
python tools/benchmark_seis_ssl_cluster.py \
  --seed 259 --warm-up 2 --repeat 10 --output-json <output.json>
```

Case: `embedding_merge_token_to_voxel`, with a 16 x 16 x 16 token grid,
128-dimensional embeddings, two merge windows, and a 128 x 128 x 128 voxel
output.

| Revision | Median (ms) | P25 (ms) | P75 (ms) |
| --- | ---: | ---: | ---: |
| Before (`649f216`) | 6.003 | 5.987 | 6.008 |
| After (`b34e3a7`) | 3.143 | 3.132 | 3.156 |

The vectorized implementation reduced median elapsed time by 47.6% (1.91x
speedup) for this deterministic case.

Environment: CPU, Python 3.10.12, NumPy 1.24.4, PyTorch 2.13.0+cu130,
Linux x86_64, 64 PyTorch threads. Measurements were collected sequentially in
the same workspace on 2026-07-16.
