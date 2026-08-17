# Volve horizon supervision layouts

The layout file records physical inline and crossline numbers. Each condition uses
the first 1+1 (`small`), 2+2 (`medium`), or 4+4 (`large`) sections and every
native-valid observation on those sections. The fixed validation pair and the
union of all large candidates are reserved for every condition.

Validate binding v2 and display all 15 plans without writing outputs:

```bash
export SEIS_SSL_CLUSTER_VOLVE_ROOT=/home/dcuser/public_data/field/volve
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/absolute/path/to/artifacts

python proc/seis_ssl_cluster/inspect_volve_horizon_sections.py --dry-run
```

Write the deterministic section CSV and compact split-plan identities below the
artifact root:

```bash
python proc/seis_ssl_cluster/inspect_volve_horizon_sections.py
```
