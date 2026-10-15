#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --job-name=rel_train_a
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/rel_train_a_%j.log

# 正式训练 (a): lambda=0, 纯 LM loss, 其余与 os_sub 完全一致
set -e
source /usr/share/lmod/lmod/init/bash
module load Python/3.14.4-foss-2026
PY=/nesi/project/massey04900/Mammography/venv/bin/python
cd /home/zhao/Mammography/Mammo/2-view/relation_loss

$PY -u train_rel.py --lambda-rel 0.0
echo "=== train (a) done ==="
