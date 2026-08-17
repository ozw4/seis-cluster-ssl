# Volve horizon supervision

This benchmark reads the five Official TWT horizons in binding schema 2 from the
read-only root selected by `SEIS_SSL_CLUSTER_VOLVE_ROOT`. It requires PASS status
for the binding, manual review, horizon visual QC, and fault visual QC. Native TWT,
fractional sample, integer sample, and validity values are retained without
smoothing, sorting, swapping, or crossing correction.

The explicit layout contract uses physical line numbers. `small`, `medium`, and
`large` are the first 1+1, 2+2, and 4+4 sections, respectively. Every available
per-horizon observation on an active section is used; point subsampling, partial
sections, and target-count calibration are not part of this contract. Inline and
crossline intersections are represented once by boolean lateral masks.

Validation is the same fixed inline/crossline pair for all conditions and is only
for checkpoint selection. The common test removes validation and the union of all
five layouts' large candidates before applying either five-horizon common support
(primary) or native per-horizon support (secondary). Unused small/medium candidate
lines therefore never return to test.

Each plan records binding and review hashes, layout-config SHA-256, physical lines
and array indices, train/validation/test counts, mask hashes, and a canonical plan
identity. Frozen and end-to-end benchmarks consume that same identity. The fixed
TWT window is samples `[552, 768)`, derived once from native bound samples with a
16-sample margin and outward alignment to the 8-sample grid.
