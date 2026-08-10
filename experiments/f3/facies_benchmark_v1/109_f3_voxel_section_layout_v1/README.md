# F3 voxel section-layout benchmark v1

This directory preregisters the closed model roster for the section-layout
downstream benchmark. `00_model_roster.yaml` resolves exactly 14 existing
frozen-embedding sources. It does not discover models from directories or file
names, and it does not authorize pretraining or embedding extraction.

The later calibration tool will write a separate
`f3_voxel_section_layout_contract_v1` mapping containing five concrete layouts.
Each layout must contain nested small, medium, and large selections with 1+1,
2+2, and 4+4 inline/crossline teacher sections. It will also copy the three
fixed integer target voxel counts calculated from historical cap-dataset
medians. Benchmark builders and runners consume only that generated contract;
they must not read an old cap manifest.

The statistical unit is `layout_id`, the validation mask is shared across all
jobs, and every decoder uses seed 42000. Models with `selection_role:
diagnostic` produce metrics but are ineligible for formal adoption.

This directory contains no runner config. Creating datasets, training decoders,
running inference, or producing a summary is outside this issue.
