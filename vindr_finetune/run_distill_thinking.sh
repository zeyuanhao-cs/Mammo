#!/usr/bin/env bash
set -euo pipefail
: "${JOB_ID:?Slurm-Web JOB_ID is required}"
: "${MAMMO_COMMIT:?Pinned code commit is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
# A resume must explicitly point at an existing run; the Python lock prevents overlap.
DISTILL_RUN_DIR="${DISTILL_RUN_DIR:-/data/me/mammo/distillation/flash-next-${JOB_ID}}"
python3 "$SCRIPT_DIR/distill_thinking.py" \
    --train "$SCRIPT_DIR/data/train_balanced_2to1.json" \
    --test "$SCRIPT_DIR/data/direct_test.json" \
    --image-root /mammo --run-dir "$DISTILL_RUN_DIR" \
    --git-commit "$MAMMO_COMMIT"
