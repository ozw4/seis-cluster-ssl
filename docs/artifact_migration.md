# Artifact migration

The [report sharing policy](report_sharing_policy.md) owns the artifact layout.
This procedure changes storage locations while retaining the contents and
historical paths of existing outputs. Migration plans and execution evidence
belong in the artifact root's `operations/artifact_migration/`, not in Git.

## Plan and preflight

An inventory records every payload, its old and proposed location, references,
and any frozen identity. Confirm the desired layout, external consumers, and
retention requirements. Identify processes using the affected artifacts and
arrange a maintenance window; the migration lock only coordinates this tool.

An execution plan is a JSON array with one object per independent move:

```json
[
  {
    "source": "old-relative-path",
    "destination": "new-relative-path",
    "device": 123,
    "inode": 456
  }
]
```

Device and inode must come from the actual source's stat inventory. Paths are
relative to the artifact root. Sources and destinations in one plan must be
disjoint, stay inside that root, and remain on the same filesystem. Destinations
must not exist; the tool does not merge or overwrite. Keep journals outside
every moved subtree.

Use the default read-only mode or an explicit dry-run before each phase:

```bash
python -m seis_ssl_cluster.artifact_migration \
  --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT" \
  --plan "$MIGRATION_PLAN" \
  --dry-run
```

For an existing mixed layout, use separate, ordered plans:

1. Move historical top-level entries into `.legacy/`, leaving root aliases.
   Keep `operations/` and the migration evidence in place.
2. Move the inventoried units out of `.legacy/` into canonical destinations,
   leaving links in the historical directory structure.
3. Extract nested archive candidates into `archive/`, retaining links from
   their former locations and preserving all data.

Validate each phase after the preceding phase, because its source locations
may not exist until that preceding phase has completed. New definitions use
canonical paths; retained historical definitions continue through aliases.
Physical payloads have one location. Root aliases remain visible while
historical readers still need them.

## Apply and verify

```bash
python -m seis_ssl_cluster.artifact_migration \
  --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT" \
  --plan "$MIGRATION_PLAN" \
  --journal "$MIGRATION_JOURNAL" \
  --apply
```

The tool renames each payload, preserving its inode and contents, and installs
a relative compatibility link. The journal records intent before each rename
and completion after the link is installed. Repeating the same apply command
with the same journal completes an interrupted move. Do not substitute a new
journal to resume an existing move. A completed rollback needs a fresh plan
and journal for any subsequent migration.

Verify all original payload paths still resolve to their original file
identities, check for broken links or unexpected remaining payloads in
`.legacy/`, and compare the frozen source checks before and after migration.
Read-only HMM checks include:

```bash
python tools/freeze_hmm_v1.py \
  --check-source --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"

python -m seis_ssl_cluster.hmm.multi_source_receipts --check \
  --matrix experiments/hmm_v2/k6810_multi_source_evaluation_v1/matrix.yaml \
  --controls experiments/hmm_v2/k6810_multi_source_evaluation_v1/hmm_v1_controls_receipt.json \
  --reuse experiments/hmm_v2/k6810_multi_source_evaluation_v1/f3_mae_k6810_reuse_receipt.json \
  --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"
```

Use applicable stage dry-runs or driver command planning to check preserved
configuration contracts. Do not execute training to validate a storage move.
Record the actual commands and results with the migration evidence.

## Rollback

Stop artifact writers first. Roll back phases in reverse order:
nested archives, canonical units, then the historical top-level view.

```bash
python -m seis_ssl_cluster.artifact_migration \
  --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT" \
  --plan "$MIGRATION_PLAN" \
  --journal "$MIGRATION_JOURNAL" \
  --rollback
```

Rollback removes only the compatibility links made by that phase and renames
the payloads back. It preserves files subsequently written inside a moved
directory. An occupied original location or a changed payload identity stops
recovery rather than overwriting it. Empty destination scaffolding, journals,
and the root's `.artifact_migration.lock` file are retained.
