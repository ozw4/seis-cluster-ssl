# Local Barlow Twins Gaussian-view history

Tried views: horizontal flip with Gaussian noise (std 0.05 and 0.10), and an
identity-plus-Gaussian-noise std 0.10 control. The common downstream conditions
were canonical five-way v3, fixed section layouts/teacher amounts, a frozen
encoder, and the unchanged decoder, inference, and evaluation settings.

Runs used 25+25 epochs; follow-up branches used 5+25 and 1+25 epochs. The
medium five-layout results were: 25+25 std 0.10 macro-F1 0.461918010 (0/5
positive versus random), 5+25 0.489563919 (delta -0.022580308; 0/5), and 1+25
0.512369394 (delta +0.000225167; 3/5). None supported an improvement.

Static reports: `reports/f3/facies_benchmark_v2/local_barlow_twins_gaussian_view_v1/`,
including `base5ep/` and `base1ep/`.

Historical runner commit: `222b8f50d47b209c0cb5090c47907d12e8c3e526`.
The experiment-specific runner and report builder were removed; evaluate future
candidates with `run_f3_lithology_candidate.py` and
`summarize_f3_lithology_candidate.py`.
