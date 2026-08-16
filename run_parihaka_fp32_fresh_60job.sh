#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

EXPECTED_SHA="726031f0f508f374867d59e00b2ed24e008a9911"
CURRENT_SHA="$(git rev-parse HEAD)"

if false; then  # SHA guard disabled
  echo "ERROR: expected HEAD=$EXPECTED_SHA, actual=$CURRENT_SHA" >&2
  exit 1
fi

if false; then  # tracked-worktree guard disabled
  echo "ERROR: tracked worktree changes exist" >&2
  git status --short >&2
  exit 1
fi

source /workspace/parihaka_channel_env.sh

export PYTHONPATH=/workspace/src
export PYTHONUNBUFFERED=1
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LOG_ROOT=/workspace/logs/parihaka_channel_fp32_fresh_60job

mkdir -p \
  "$LOG_ROOT/audit" \
  "$LOG_ROOT/frozen" \
  "$LOG_ROOT/end_to_end" \
  "$LOG_ROOT/summary"

if pgrep -af 'run_parihaka_channel_(decoder|end_to_end)\.py' \
  > "$LOG_ROOT/audit/running_processes.txt"
then
  echo "ERROR: Channel worker is already running" >&2
  cat "$LOG_ROOT/audit/running_processes.txt" >&2
  exit 1
fi

test -f "$FROZEN_CONFIG"
test -f "$E2E_CONFIG"
test -f "$LAYOUT_CONFIG"

grep -q '^[[:space:]]*amp: false$' "$FROZEN_CONFIG"
grep -q '^[[:space:]]*amp: false$' "$E2E_CONFIG"

RESET_MARKER="$LOG_ROOT/audit/fresh_roots_initialized"

if [[ ! -f "$RESET_MARKER" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  ARCHIVE_LOG="$LOG_ROOT/audit/archived_roots.txt"

  : > "$ARCHIVE_LOG"

  for root in \
    "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark" \
    "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_end_to_end"
  do
    if [[ -e "$root" ]]; then
      archived="${root}.fp32_partial_${STAMP}"
      test ! -e "$archived"
      mv "$root" "$archived"
      printf '%s -> %s\n' "$root" "$archived" | tee -a "$ARCHIVE_LOG"
    fi
  done

  test ! -e "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark"
  test ! -e "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_end_to_end"

  {
    echo "execution_sha=$CURRENT_SHA"
    echo "initialized_at=$STAMP"
  } > "$RESET_MARKER"
fi

git rev-parse HEAD > "$LOG_ROOT/audit/execution_sha.txt"
git status --short > "$LOG_ROOT/audit/git_status_at_start.txt"
nvidia-smi > "$LOG_ROOT/audit/nvidia_smi.txt"

run_frozen_job() {
  local model="$1"
  local layout="$2"
  local size="$3"

  local job_dir="$FROZEN_RUN_ROOT/model=$model/layout=$layout/size=$size"
  local log="$LOG_ROOT/frozen/${model}_${layout}_${size}.log"

  if [[ -f "$job_dir/metrics.json" ]]; then
    echo "completed: frozen/$model/$layout/$size"
    return
  fi

  if [[ -f "$job_dir/latest.pt" ]]; then
    echo "resume: frozen/$model/$layout/$size"

    python /workspace/proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
      --config "$FROZEN_CONFIG" \
      --model "$model" \
      --layout "$layout" \
      --size "$size" \
      --layout-config "$LAYOUT_CONFIG" \
      --device cuda \
      --resume "$job_dir/latest.pt" \
      2>&1 | tee -a "$log"
  else
    echo "start: frozen/$model/$layout/$size"

    python /workspace/proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
      --config "$FROZEN_CONFIG" \
      --model "$model" \
      --layout "$layout" \
      --size "$size" \
      --layout-config "$LAYOUT_CONFIG" \
      --device cuda \
      2>&1 | tee "$log"
  fi

  test -f "$job_dir/metrics.json"
}

run_e2e_job() {
  local encoder_init="$1"
  local layout="$2"
  local size="$3"

  local job_dir="$E2E_RUN_ROOT/encoder_init=$encoder_init/layout=$layout/size=$size"
  local log="$LOG_ROOT/end_to_end/${encoder_init}_${layout}_${size}.log"

  if [[ -f "$job_dir/metrics.json" ]]; then
    echo "completed: e2e/$encoder_init/$layout/$size"
    return
  fi

  if [[ -f "$job_dir/latest.pt" ]]; then
    echo "resume: e2e/$encoder_init/$layout/$size"

    python /workspace/proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
      --config "$E2E_CONFIG" \
      --encoder-init "$encoder_init" \
      --layout "$layout" \
      --size "$size" \
      --layout-config "$LAYOUT_CONFIG" \
      --device cuda \
      --resume "$job_dir/latest.pt" \
      2>&1 | tee -a "$log"
  else
    echo "start: e2e/$encoder_init/$layout/$size"

    python /workspace/proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py \
      --config "$E2E_CONFIG" \
      --encoder-init "$encoder_init" \
      --layout "$layout" \
      --size "$size" \
      --layout-config "$LAYOUT_CONFIG" \
      --device cuda \
      2>&1 | tee "$log"
  fi

  test -f "$job_dir/metrics.json"
}

for layout in \
  layout_000 \
  layout_001 \
  layout_002 \
  layout_003 \
  layout_004
do
  for size in small medium large; do
    run_frozen_job pretrained "$layout" "$size"
    run_frozen_job random "$layout" "$size"
  done
done

FROZEN_COUNT="$(
  find "$FROZEN_RUN_ROOT" -type f -name metrics.json | wc -l
)"
echo "frozen completed: $FROZEN_COUNT/30"
test "$FROZEN_COUNT" -eq 30

for layout in \
  layout_000 \
  layout_001 \
  layout_002 \
  layout_003 \
  layout_004
do
  for size in small medium large; do
    run_e2e_job pretrained "$layout" "$size"
    run_e2e_job random "$layout" "$size"
  done
done

E2E_COUNT="$(
  find "$E2E_RUN_ROOT" -type f -name metrics.json | wc -l
)"
echo "end-to-end completed: $E2E_COUNT/30"
test "$E2E_COUNT" -eq 30

if [[ ! -f \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/summary/summary.json" ]]
then
  python /workspace/proc/seis_ssl_cluster/summarize_parihaka_channel_benchmark.py \
    --config "$FROZEN_CONFIG" \
    --dry-run \
    2>&1 | tee "$LOG_ROOT/summary/frozen_dry_run.log"

  python /workspace/proc/seis_ssl_cluster/summarize_parihaka_channel_benchmark.py \
    --config "$FROZEN_CONFIG" \
    2>&1 | tee "$LOG_ROOT/summary/frozen.log"
fi

if [[ ! -f \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_end_to_end/summary/summary.json" ]]
then
  python \
    /workspace/proc/seis_ssl_cluster/summarize_parihaka_channel_end_to_end.py \
    --config "$E2E_CONFIG" \
    --dry-run \
    2>&1 | tee "$LOG_ROOT/summary/end_to_end_dry_run.log"

  python \
    /workspace/proc/seis_ssl_cluster/summarize_parihaka_channel_end_to_end.py \
    --config "$E2E_CONFIG" \
    2>&1 | tee "$LOG_ROOT/summary/end_to_end.log"
fi

if [[ ! -f \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_end_to_end/four_way_summary/four_way_summary.json" ]]
then
  python \
    /workspace/proc/seis_ssl_cluster/summarize_parihaka_channel_four_way.py \
    --config "$E2E_CONFIG" \
    --frozen-config "$FROZEN_CONFIG" \
    --dry-run \
    2>&1 | tee "$LOG_ROOT/summary/four_way_dry_run.log"

  python \
    /workspace/proc/seis_ssl_cluster/summarize_parihaka_channel_four_way.py \
    --config "$E2E_CONFIG" \
    --frozen-config "$FROZEN_CONFIG" \
    2>&1 | tee "$LOG_ROOT/summary/four_way.log"
fi

cat > "$LOG_ROOT/audit/final_counts.json" <<JSON
{
  "frozen_completed": $FROZEN_COUNT,
  "end_to_end_completed": $E2E_COUNT,
  "total_completed": $((FROZEN_COUNT + E2E_COUNT)),
  "execution_sha": "$CURRENT_SHA"
}
JSON

git status --short > "$LOG_ROOT/audit/git_status_at_end.txt"
nvidia-smi > "$LOG_ROOT/audit/nvidia_smi_at_end.txt"

touch "$LOG_ROOT/audit/complete"

echo "all 60 FP32 jobs and summaries completed"
