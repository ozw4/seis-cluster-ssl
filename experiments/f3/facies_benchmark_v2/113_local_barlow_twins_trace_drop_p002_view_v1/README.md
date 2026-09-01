# Local Barlow Twins trace-drop p=.02 history

Tried view: horizontal flip with horizontal trace drop probability 0.02. Common
conditions were canonical five-way v3, fixed section layouts/teacher amounts, a
frozen encoder, and unchanged decoder, inference, and evaluation settings.

The run used 1 base epoch plus 25 continuation epochs. Its medium five-layout
macro-F1 was 0.510727766 (delta -0.001416461 versus random; 2/5 positive), so
it did not support an improvement.

Static report:
`reports/f3/facies_benchmark_v2/local_barlow_twins_trace_drop_p002_view_v1/base1ep/`.

Historical runner commit: `222b8f50d47b209c0cb5090c47907d12e8c3e526`.
The experiment-specific runner and report builder were removed; evaluate future
candidates with `run_f3_lithology_candidate.py` and
`summarize_f3_lithology_candidate.py`.
