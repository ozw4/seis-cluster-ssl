# Performance migration validation summary

- Status: `COMPLETE`
- Current git SHA: `332478be21a021e46ee6c1d9423f14859b0cd819`
- Historical baseline SHA: `7731f341a293ea0c5cb5c5dfabba574148861e3a`
- Migration decision: `PASS_WITH_NUMERIC_DRIFT`
- Required rerun scope: no historical rerun; add a future current-code K=6 control
- Multi-head policy: train a current-code single-head K=6 control under the same conditions before comparing multi-head K=6/8/10.
- Atomic path provenance: Producer runtime configuration may retain its temporary staging path; the committed artifact directory and completion manifest identify the final location. These are path-only provenance fields, not scientific identity.

No voxel-decoder or M3-V/M3-V-LB full-job retraining was performed by this migration validation.
