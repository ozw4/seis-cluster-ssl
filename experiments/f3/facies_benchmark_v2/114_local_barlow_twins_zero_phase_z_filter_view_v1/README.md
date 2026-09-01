# Local Barlow Twins zero-phase Z-filter history

Tried view: horizontal flip with zero-phase Z-filter side weight 0.25. Common
conditions were canonical five-way v3, fixed section layouts/teacher amounts, a
frozen encoder, and unchanged decoder, inference, and evaluation settings.

The run used 1 base epoch plus 25 continuation epochs. Its medium five-layout
macro-F1 was 0.504512030 (delta -0.007632197 versus random; 2/5 positive), so
it did not support an improvement.

Static report:
`reports/f3/facies_benchmark_v2/local_barlow_twins_zero_phase_z_filter_view_v1/base1ep/`.

Historical runner commit: `222b8f50d47b209c0cb5090c47907d12e8c3e526`.
The experiment-specific runner and report builder were removed; evaluate future
candidates with `run_f3_lithology_candidate.py` and
`summarize_f3_lithology_candidate.py`.
