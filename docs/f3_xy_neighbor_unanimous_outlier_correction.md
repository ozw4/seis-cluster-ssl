# F3 unanimous XY-neighbour outlier correction v1

`xy_neighbor_unanimous_outlier_correction_v1` is a conservative, independent
successor to the 3-of-4 XY-neighbour hard-label target. For a valid center it
uses only same-z `(x-1, y)`, `(x+1, y)`, `(x, y-1)`, and `(x, y+1)` neighbours,
with invalid neighbours excluded. Four valid neighbours must agree 4/4; three
must agree 3/3; two or fewer never change the center.

All proposals read frozen source hard labels, are applied synchronously in one
pass, and retain the existing internal-valid-token ordered-trace guard. The
policy never uses z or diagonal neighbours, posterior values, embeddings,
amplitudes, confidence tuning, iterative passes, or downstream metrics.

The immutable target representation is
`xy_neighbor_unanimous_hard_labels_v1`; its schema-6 training identity is
separate from schemas 2 through 5. A target-only audit validates source,
3-of-4, and unanimous manifests, requires each unanimous change to be a
3-of-4 change with the same output label, and issues only
`XYUNANIM_TARGET_GO` or `XYUNANIM_TARGET_HOLD`.
