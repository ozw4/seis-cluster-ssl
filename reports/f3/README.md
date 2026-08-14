# F3 Reports

`reports/f3/legacy/facies_benchmark_v1/` is the frozen reference for the F3
`facies_benchmark_v1` voxel-count, multiple-seed protocol. Its producer Git
revision is recorded in [legacy/README.md](legacy/README.md). The frozen files
are not pipeline inputs.

The next active F3 benchmark will be implemented separately as
`facies_benchmark_v2`, using section-count conditions and one seed. Its future
report location is `reports/f3/facies_benchmark_v2/`; that directory and its
scientific conditions are not defined yet.

Do not compare v1 and v2 as though they used the same protocol. Complete
execution outputs, intermediate products, and downstream inputs belong under
`artifacts/`, not `reports/`.
