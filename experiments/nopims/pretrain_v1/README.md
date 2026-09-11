# NOPIMS pretrain v1

This namespace stores versioned NOPIMS amplitude-only registry, pretraining,
embedding, clustering, and visualization definitions. It does not declare a
complete end-to-end chain.

The registry definitions are ordered, but downstream definitions may require
explicit manifests, checkpoints, or other artifacts that are not produced by
the preceding tracked definitions. Treat each YAML's paths as its compatibility
contract, verify those inputs before execution, and do not infer a handoff from
directory or model-tag names. Use a read-only CLI preflight where supported.
