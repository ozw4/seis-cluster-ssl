# Completion and recovery

All 42 missing decoder cells completed on 2026-11-15, with the three existing
Barlow layout_000/large cells reused. F3 common-test evaluation also completed:
45 v1 zero-distillation cells and 30 Barlow v2 cells (including 15 rescored
0.2 controls). The original 0.2 control values matched exactly.

The detached worker stopped at final aggregation at 21:30 UTC: a manifest path
string was passed to the Path-only CSV reader. This was a postprocessing error;
training, selected checkpoints and evaluations had already completed. The
collector now converts the path explicitly. Its regression fixture enforces
the real reader's Path contract so the same mismatch cannot be masked again.
All 25 focused completion/common-test tests pass, together with Ruff lint and
format checks. Only summary generation was rerun; no model or score was rerun.

The exact ten-file summary is now published. Independent numerical verification
checked 90 Barlow cells / 18 means, 405 aligned v1 cells / 81 means, and 540
factorial cells / 108 means. Every group has five unique layouts; all values are
finite. Means and sample SDs were checked for both principal/secondary metrics
(414 group-metric checks), as were all 135 paired contrasts, to 1e-12 tolerance.
The recovery receipt records the output hashes and corrected module hash.

Evidence remains under
`artifacts/seis_ssl_cluster/operations/hmm_v2/missing_controls_completion_v1/`:
`complete.json`, `recovery_complete.json`, `summary_recovery.log`, and the
original failed `exit.json` / `summary_failure_status.json`. The historical
failure records are retained; the current `status.json` records completion.
Results are under the matching artifact study's `summary/` directory.

Barlow Twins' descriptive 15-cell mean favors zero distillation on each survey:

| Survey | Metric | Weight 0 | Weight 0.2 |
| --- | --- | ---: | ---: |
| F3 | Common-test mean IoU | 0.453518 | 0.447653 |
| Parihaka | Test channel IoU | 0.171067 | 0.166576 |
| Volve | Test horizon MAE [samples] | 11.555585 | 12.498038 |

These means span five layouts and three nested budgets. They are descriptive,
not 15 independent encoder repetitions. The fixed continuation seed is 42.
