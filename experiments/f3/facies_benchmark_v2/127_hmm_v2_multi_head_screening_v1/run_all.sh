#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: run_all.sh [--plan | --dry-run | --execute]
  [--from STAGE] [--to STAGE] [--candidate ID]
  [--layout layout_000..layout_004] [--size small|medium|large] [--resume]
Stages: cluster-targets cluster-replay export manifests smoke full audit embeddings downstream summary
Default: --plan, all stages and all four candidates / 60 cells.
--plan prints commands without Python, artifacts, logs, or locks.
--dry-run invokes read-only CLI checks; prerequisites must already exist.
--execute runs sequentially and logs to the screening artifact namespace.
--resume requires one stage (smoke/full/downstream) and one candidate;
          downstream also requires one layout and size. Only its own latest.pt is used.
Completed outputs are not silently skipped. See RUNBOOK.md for restart boundaries.
HELP
}
fail() { echo "error: $*" >&2; exit 2; }
mode=plan
mode_set=false
first=cluster-targets
last=summary
selected_candidate=
selected_layout=
selected_size=
resume=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --plan|--dry-run|--execute)
      [[ "$mode_set" == false ]] || fail 'choose one execution mode'
      mode="${1#--}"; mode_set=true; shift ;;
    --from|--to|--candidate|--layout|--size)
      (($# >= 2)) || fail "missing value for $1"
      case "$1" in
        --from) first="$2" ;; --to) last="$2" ;;
        --candidate) selected_candidate="$2" ;;
        --layout) selected_layout="$2" ;; --size) selected_size="$2" ;;
      esac
      shift 2 ;;
    --resume) resume=true; shift ;;
    *) fail "unknown argument: $1" ;;
  esac
done
stages=(cluster-targets cluster-replay export manifests smoke full audit embeddings downstream summary)
candidates=(mae100_hmm_v2_mh_k46_distill020 mae100_hmm_v2_mh_k68_distill020 mae100_hmm_v2_mh_k468_distill020 mae100_hmm_v2_mh_k6810_distill020)
layouts=(layout_000 layout_001 layout_002 layout_003 layout_004)
sizes=(small medium large)
contains() { local value="$1"; shift; for item in "$@"; do [[ "$item" != "$value" ]] || return 0; done; return 1; }
contains "$first" "${stages[@]}" || fail "unknown stage: $first"
contains "$last" "${stages[@]}" || fail "unknown stage: $last"
start_index=0; end_index=0
for index in "${!stages[@]}"; do
  [[ "${stages[$index]}" != "$first" ]] || start_index="$index"
  [[ "${stages[$index]}" != "$last" ]] || end_index="$index"
done
((start_index <= end_index)) || fail 'stage range is reversed'
if [[ -n "$selected_candidate" ]]; then
  contains "$selected_candidate" "${candidates[@]}" || fail 'unknown candidate'
  [[ "$first" == "$last" && "$first" =~ ^(manifests|smoke|full|embeddings|downstream)$ ]] || fail 'candidate selection requires one candidate stage'
  candidates=("$selected_candidate")
fi
if [[ -n "$selected_layout" || -n "$selected_size" ]]; then
  [[ "$first" == downstream && "$last" == downstream ]] || fail 'cell selection requires downstream only'
fi
if [[ -n "$selected_layout" ]]; then
  contains "$selected_layout" "${layouts[@]}" || fail 'unknown layout'
  layouts=("$selected_layout")
fi
if [[ -n "$selected_size" ]]; then
  contains "$selected_size" "${sizes[@]}" || fail 'unknown size'
  sizes=("$selected_size")
fi
if [[ "$resume" == true ]]; then
  [[ "$first" == "$last" && "$first" =~ ^(smoke|full|downstream)$ && -n "$selected_candidate" ]] || fail 'resume requires one training stage and candidate'
  if [[ "$first" == downstream ]]; then
    [[ -n "$selected_layout" && -n "$selected_size" ]] || fail 'decoder resume requires one layout and size'
  fi
fi

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$experiment_dir/../../../.." && pwd)"
cd "$repository_dir"
if [[ "$mode" != plan ]]; then
  : "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export the existing artifact root}"
  : "${F3_ROOT:?export the existing F3 root}"
  [[ "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT" == /* && "$F3_ROOT" == /* ]] || fail 'artifact and F3 roots must be absolute'
  if [[ -n "${SEIS_SSL_CLUSTER_WORKSPACE:-}" ]]; then
    [[ "$(cd "$SEIS_SSL_CLUSTER_WORKSPACE" && pwd)" == "$repository_dir" ]] || fail 'workspace differs from this checkout'
  fi
fi
export SEIS_SSL_CLUSTER_WORKSPACE="$repository_dir"
export PYTHONPATH="$repository_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
artifact_root='${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
if [[ -n "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:-}" ]]; then artifact_root="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"; fi
namespace=hmm_v2_multi_head_screening_v1
pretraining_root="$artifact_root/pretraining/f3/facies_benchmark_v1/$namespace"
benchmark_root="$artifact_root/f3_lithology_benchmark/$namespace"
current_step=setup
if [[ "$mode" == execute ]]; then
  : "${CUDA_VISIBLE_DEVICES:?select the GPU for scheduled execution}"
  log_root="$benchmark_root/logs"
  mkdir -p "$log_root"
  exec 9>"$log_root/driver.lock"
  flock -n 9 || fail 'another screening driver holds the lock'
  invocation_dir="$(mktemp -d "$log_root/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
  trap 'status=$?; printf "%s exit=%s step=%s\n" "$(date -u +%FT%TZ)" "$status" "$current_step" >> "$invocation_dir/events.log"' EXIT
  echo "logs: $invocation_dir"
fi
run() {
  current_step="$1"; shift
  printf '[%s] ' "$current_step"; printf '%q ' "$@"; printf '\n'
  [[ "$mode" != plan ]] || return 0
  if [[ "$mode" == execute ]]; then
    printf '%s start=%s\n' "$(date -u +%FT%TZ)" "$current_step" >> "$invocation_dir/events.log"
    { printf 'command: '; printf '%q ' "$@"; printf '\n'; } >> "$invocation_dir/$current_step.log"
    "$@" 2>&1 | tee -a "$invocation_dir/$current_step.log"
    printf '%s complete=%s\n' "$(date -u +%FT%TZ)" "$current_step" >> "$invocation_dir/events.log"
  else
    "$@"
  fi
}
write_step() {
  if [[ "$mode" == dry-run ]]; then run "$@" --dry-run; else run "$@"; fi
}
audit_done=false
ensure_audit() {
  if [[ "$audit_done" == false ]]; then
    write_step completed_sources python proc/seis_ssl_cluster/audit_strat_hmm_multi_head_sources.py --config "$experiment_dir/30_pretraining/03_audit_completed_sources.yaml"
    audit_done=true
  fi
}
manifest_bridge='import sys; from pathlib import Path; from seis_ssl_cluster.config import load_config; from proc.seis_ssl_cluster.build_strat_hmm_multi_head_targets import main; c=load_config(Path(sys.argv[1])); args=["--source-embedding-dir",c["source_embedding_dir"],"--manifest",c["manifest"],"--replay-k6-root",c["replay_k6_root"],"--only-missing"]; args += [v for k,r in c["head_roots"].items() for v in ("--head-root",f"{k}={r}")]; main([*args,*sys.argv[2:]])'
for ((index=start_index; index<=end_index; index++)); do
  stage="${stages[$index]}"
  current_step="$stage"
  case "$stage" in
    cluster-targets|cluster-replay)
      recipe=01_cluster_hmm_k4810; suffix=k4810
      if [[ "$stage" == cluster-replay ]]; then recipe=02_replay_hmm_k6; suffix=k6_replay; fi
      output="$artifact_root/clustering/f3/facies_benchmark_v1/$namespace/mae100/$suffix"
      if [[ "$mode" == execute && -e "$output" ]]; then fail "clustering output exists; inspect it and restart at a later stage: $output"; fi
      write_step "$stage" python proc/seis_ssl_cluster/cluster_embeddings.py --config "$experiment_dir/10_targets/$recipe.yaml"
      ;;
    export) write_step export bash "$experiment_dir/10_targets/03_export_pseudo_targets.sh" ;;
    manifests)
      for candidate in "${candidates[@]}"; do
        write_step "manifest_$candidate" python -c "$manifest_bridge" "$experiment_dir/20_manifests/$candidate.yaml"
      done ;;
    smoke|full)
      recipe=01_gpu_feasibility_1step; subdir=smoke_1step
      if [[ "$stage" == full ]]; then recipe=02_full_25ep; subdir=full_25ep; fi
      for candidate in "${candidates[@]}"; do
        tag="${candidate#mae100_hmm_v2_mh_k}"; tag="${tag%_distill020}"
        manifest="$artifact_root/pseudo_targets/f3/facies_benchmark_v1/$namespace/mh_k$tag/multi_head_target_manifest.json"
        digest='<sha256 of selected live manifest>'
        if [[ "$mode" != plan ]]; then
          current_step="manifest_hash_$candidate"
          digest="$(python -c 'import sys; from seis_ssl_cluster.clustering.features import file_sha256; print(file_sha256(sys.argv[1]))' "$manifest")"
        fi
        resume_args=()
        [[ "$resume" == false ]] || resume_args=(--resume "$pretraining_root/$candidate/$subdir/latest.pt")
        write_step "${stage}_$candidate" env "SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256=$digest" python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$experiment_dir/30_pretraining/$candidate/$recipe.yaml" "${resume_args[@]}"
      done ;;
    audit) ensure_audit ;;
    embeddings)
      ensure_audit
      for candidate in "${candidates[@]}"; do
        write_step "embedding_$candidate" python proc/seis_ssl_cluster/extract_embeddings.py --config "$experiment_dir/40_embeddings/$candidate.yaml" --device cuda --skip-existing
      done ;;
    downstream)
      ensure_audit
      for candidate in "${candidates[@]}"; do
        for layout in "${layouts[@]}"; do
          for size in "${sizes[@]}"; do
            resume_args=()
            [[ "$resume" == false ]] || resume_args=(--resume "$benchmark_root/runs/model=$candidate/layout=$layout/size=$size/decoder/latest.pt")
            write_step "${candidate}_${layout}_$size" python proc/seis_ssl_cluster/run_f3_lithology_candidate.py --config "$experiment_dir/50_downstream/$candidate.yaml" --layout "$layout" --size "$size" "${resume_args[@]}"
          done
        done
      done ;;
    summary) write_step paired_summary python proc/seis_ssl_cluster/summarize_f3_multi_head_screening.py --config "$experiment_dir/60_summary/01_paired_screening.yaml" ;;
  esac
done
printf 'finished mode=%s stages=%s..%s\n' "$mode" "$first" "$last"
