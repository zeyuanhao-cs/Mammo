#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=cnn_cc
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/train_cnn_cc.log

# CNN baseline --views cc (仅 CC 单视图, natural 配方)
# 输出: 2-view/outputs/vision_baseline/natural_views-cc/
# 断点续跑: 重新 sbatch 同一脚本即可 (last.pt 自动恢复)

echo "$(date '+%F %T') job $SLURM_JOB_ID start on $SLURMD_NODENAME"
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load Python/3.14.4-foss-2026

PY=/nesi/project/massey04900/Mammography/venv/bin/python
SCRIPT=/home/zhao/Mammography/Mammo/2-view/scripts/train_vision_baseline.py

$PY -u $SCRIPT --version natural --views cc \
  --epochs 16 --batch-size 8 --grad-accum 2 \
  --num-workers 11

echo "$(date '+%F %T') exit code: $?"
