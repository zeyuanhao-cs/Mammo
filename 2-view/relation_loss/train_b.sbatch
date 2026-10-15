#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --job-name=rel_train_b
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/rel_train_b_%j.log

# 正式训练 (b): lambda=0.5, LM loss + 0.5*(R1+R2), 其余与 (a) 完全一致
set -e
source /usr/share/lmod/lmod/init/bash
module load Python/3.14.4-foss-2026
PY=/nesi/project/massey04900/Mammography/venv/bin/python
cd /home/zhao/Mammography/Mammo/2-view/relation_loss

$PY -u train_rel.py --lambda-rel 0.5
echo "=== train (b) done ==="
