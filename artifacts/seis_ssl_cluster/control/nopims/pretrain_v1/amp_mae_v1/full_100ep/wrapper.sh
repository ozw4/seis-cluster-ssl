#!/usr/bin/env bash
set -o pipefail
CONTROL_DIR="/workspace/artifacts/seis_ssl_cluster/control/nopims/pretrain_v1/amp_mae_v1/full_100ep"
LOG_PATH="/workspace/artifacts/seis_ssl_cluster/logs/nopims/pretrain_v1/amp_mae_v1/full_100ep.log"
cd /workspace || exit 2
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${CONTROL_DIR}/started_at_utc.txt"
{
  echo "===== full_100ep launch $(cat "${CONTROL_DIR}/started_at_utc.txt") ====="
  echo "cwd: /workspace"
  echo "command: PYTHONPATH=/workspace/src python proc/seis_ssl_cluster/train_amp_mae.py --config experiments/nopims/pretrain_v1/10_pretrain/amp_mae_v1/03_full_100ep.yaml"
} >> "${LOG_PATH}"
PYTHONPATH=/workspace/src python proc/seis_ssl_cluster/train_amp_mae.py --config experiments/nopims/pretrain_v1/10_pretrain/amp_mae_v1/03_full_100ep.yaml 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
echo "${status}" > "${CONTROL_DIR}/exit_code.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${CONTROL_DIR}/finished_at_utc.txt"
exit "${status}"
