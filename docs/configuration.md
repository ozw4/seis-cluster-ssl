# Configuration

Versioned experiment definitions belong under `experiments/`. The generic YAML
files under `proc/configs/seis_ssl_cluster/` are examples for the generic
entrypoints.

The shared loader expands required `${NAME}` environment references before
validation. Some compact experiment YAMLs explicitly name a canonical config
and add only candidate-specific values. Other workflows derive fixed or
job-local paths from declared roots and identities such as model, layout, and
data size. These forms of composition are part of the relevant workflow's
resolver contract.

The resolver selected by each entrypoint—whether it lives in the common config
package or a survey module—defines the accepted YAML shape, fixed values,
defaults, and validation. Supported command-line overrides are applied by that
entrypoint, so a reproducible invocation consists of the owning YAML, every
explicitly referenced canonical config, the required environment values, and
the supplied overrides.

Repository storage rules are defined only in the
[report sharing policy](report_sharing_policy.md).
