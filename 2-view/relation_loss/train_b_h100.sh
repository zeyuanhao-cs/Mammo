#!/bin/bash
# relation 正式训练 (b) lambda=0.5 — 在 gpu_hold_h100 allocation 内经 srun --overlap 执行
set -e
source /usr/share/lmod/lmod/init/bash
module load Python/3.14.4-foss-2026
PY=/nesi/project/massey04900/Mammography/venv/bin/python
cd /home/zhao/Mammography/Mammo/2-view/relation_loss

$PY -u train_rel.py --lambda-rel 0.5
echo "=== train (b) done ==="
