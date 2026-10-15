#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=cnn_four
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/train_cnn_four.log

# CNN baseline --views four (同一检查左右乳各 CC+MLO, 共享 backbone 拼接 4 份特征,
# 标签仍取本侧乳房; natural 配方)。每样本 4 张图, 显存/耗时约为 ipsi 的 2 倍,
# 若 OOM 可降为 --batch-size 4 --grad-accum 4 (等效 batch 16 不变)。
# 输出: 2-view/outputs/vision_baseline/natural_views-four/
# 断点续跑: 重新 sbatch 同一脚本即可 (last.pt 自动恢复)

echo "$(date '+%F %T') job $SLURM_JOB_ID start on $SLURMD_NODENAME"
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load Python/3.14.4-foss-2026

PY=/nesi/project/massey04900/Mammography/venv/bin/python
SCRIPT=/home/zhao/Mammography/Mammo/2-view/scripts/train_vision_baseline.py

$PY -u $SCRIPT --version natural --views four \
  --epochs 16 --batch-size 8 --grad-accum 2 \
  --num-workers 11

echo "$(date '+%F %T') exit code: $?"
