# Current-code single-head K=6 control

Status: `CONTROL_READY_POSITIVE`

## Scope

F3 original split; cap25/cap50/cap100; five paired subsample seeds; fixed frozen voxel decoder. Results are descriptive paired-seed summaries.

## Readiness

- Current K6 vs MAE positive budgets: ['cap25', 'cap50', 'cap100']
- Historical-M1 drift triggers: []
- Monitored-class major degradations: []

## Files

- `control_paired_deltas.csv` contains all per-seed paired deltas.
- `control_summary_by_budget.csv` contains means, medians, sample SD, ranges, wins/losses/ties, and worst seeds.
- `control_monitored_class_summary.csv` contains class 3/5 F1, IoU, and boundary-recall summaries.

## Final repository provenance

- HEAD: `371ec7ae4156fdf551a11dd1a9d7ae31ec2ed3ec`
- `git diff --binary HEAD` SHA-256: `cbe069e5b564daf1a4d7e0024cbc502f391c590de3e40cfb3d167c6d02a54af4`
