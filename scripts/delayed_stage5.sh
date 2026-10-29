#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DELAY_SECONDS="${1:-10800}"
if [[ ! "$DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
    echo 'Delay must be a nonnegative integer in seconds.' >&2
    exit 2
fi

# Use the training environment even after the SSH shell disconnects.
source /home/xixiangtang/miniconda3/etc/profile.d/conda.sh
conda activate mvsplat
cd "$PROJECT_DIR"
mkdir -p outputs/scheduled
LOG_FILE="outputs/scheduled/stage5_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1

check_branch() {
    local branch
    branch="$(git branch --show-current)"
    if [[ "$branch" != full_shared ]]; then
        echo "Expected full_shared branch, found '$branch'; training aborted."
        exit 1
    fi
}

check_branch
echo 'Checking training config before waiting.'
python -m src.main +experiment=stage5 mode=train --cfg job --resolve > "${LOG_FILE%.log}.config.yaml"
echo "Log: $PROJECT_DIR/$LOG_FILE"
echo "Scheduled start (Beijing): $(TZ=Asia/Shanghai date -d "+$DELAY_SECONDS seconds" '+%F %T %Z')"
echo "Waiting $DELAY_SECONDS seconds."
sleep "$DELAY_SECONDS"
check_branch
echo "Starting training: $(TZ=Asia/Shanghai date '+%F %T %Z')"
echo "Python: $(command -v python)"
set +e
PYTHONUNBUFFERED=1 python -m src.main +experiment=stage5 mode=train
TRAIN_STATUS=$?
echo "Training exited with status $TRAIN_STATUS at $(TZ=Asia/Shanghai date '+%F %T %Z')"
exit "$TRAIN_STATUS"
