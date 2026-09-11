# Volve horizon supervision layouts

The scientific split and claim boundary are maintained in
[`docs/volve_horizon_supervision.md`](../../../../docs/volve_horizon_supervision.md).
The layout YAML is the source of truth for physical lines and condition sizes.
Use the shared [Volve benchmark environment](../README.md).

Inspect all plans without writing, then publish the deterministic split-plan
artifacts:

```bash
python proc/seis_ssl_cluster/inspect_volve_horizon_sections.py --dry-run
python proc/seis_ssl_cluster/inspect_volve_horizon_sections.py
```
