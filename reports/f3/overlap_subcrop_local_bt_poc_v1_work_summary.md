# F3 overlapping-subcrop Local Barlow Twins PoC v1

Date: 2026-09-02

The experimental question, candidate definitions, and execution sequence are in
the [experiment runbook](../../experiments/f3/facies_benchmark_v1/112_local_bt_overlap_subcrop_poc_v1/README.md).
This report records only the measured screen and its interpretation.

## Screen result

The Random baseline reached macro-F1 `0.4965410800` on 470,136 unique validation
voxels for `layout_001 / medium`.

| Candidate | Shift | Macro-F1 | Delta vs Random | Decision |
|---|---:|---:|---:|---|
| `shift04_proj384_pairs128_lambda005` | `[4,4,0]` | 0.4115269085 | -0.0850141715 | fail |
| `shift02_proj384_pairs128_lambda005` | `[2,2,0]` | 0.4476944851 | -0.0488465949 | fail |

Both candidates failed the strict screen, so the other four layouts were not
run and neither candidate was adopted.

## Representation diagnostics

| Candidate | Raw norm | Token std | Raw effective rank | LayerNorm std | LayerNorm effective rank |
|---|---:|---:|---:|---:|---:|
| Random | 26.4523632615 | 0.6585601699 | 23.9998007592 | 0.5220072650 | 25.4812122319 |
| `shift04_proj384_pairs128_lambda005` | 95.7963889138 | 4.7375605914 | 21.8244936985 | 0.9725560302 | 21.9308540078 |
| `shift02_proj384_pairs128_lambda005` | 95.6020204637 | 4.7402329837 | 21.8931642113 | 0.9752670668 | 22.0107362799 |

The trained candidates changed feature scale substantially but did not recover
the Random baseline in this downstream screen. The rank diagnostics therefore
do not justify advancing either arm by themselves.

`shift06_proj384_pairs128_lambda005` is configured but has no recorded result;
no conclusion is made for that arm.
