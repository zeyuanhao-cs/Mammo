#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=ablation_2view
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/ablation.log

# Run A (Repeat-OS Qwen3.5-4B LoRA) 图像消融: smoke -> gate -> 全量 -> 汇总
# 断点续跑: smoke 不过 gate (exit 2/3) 则不进入全量; 全量按条件/按 id 断点续跑

echo "$(date '+%F %T') job $SLURM_JOB_ID start on $SLURMD_NODENAME"
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load Python/3.14.4-foss-2026

PY=/nesi/project/massey04900/Mammography/venv/bin/python
DIR=/home/zhao/Mammography/Mammo/2-view/scripts/ablation

# ---- 1. smoke: 每条件 5 个乳房 + orig 对拍 + 耗时预估 ----
echo "=== [1/3] smoke ==="
$PY -u $DIR/run_ablation.py --mode smoke
SMOKE_RC=$?
echo "$(date '+%F %T') smoke exit code: $SMOKE_RC"

if [ $SMOKE_RC -ne 0 ]; then
  echo "=== smoke 未通过, 不进入全量 ==="
  exit $SMOKE_RC
fi

# ---- 2. 全量: 5 个消融条件 x 2000 乳房 (orig 复用已有打分文件) ----
echo "=== [2/3] full ==="
$PY -u $DIR/run_ablation.py --mode full --budget-min 150
FULL_RC=$?
echo "$(date '+%F %T') full exit code: $FULL_RC"

# ---- 3. 汇总 (纯 CPU; full 失败也尝试汇总已完成部分) ----
echo "=== [3/3] summarize ==="
$PY -u $DIR/summarize_ablation.py
SUM_RC=$?
echo "$(date '+%F %T') summarize exit code: $SUM_RC"
echo "$(date '+%F %T') ALL DONE (full=$FULL_RC summary=$SUM_RC)"
