#!/bin/bash
#SBATCH --job-name=pz_all
#SBATCH --partition=genoa
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE

# patch-zoom 单个 2 小时作业:
#   1) 三尺度冒烟 (20 步 + 16 块推理) 实测每步耗时/峰值显存/推理速度
#   2) 预算闸门: 预估总时长 > 105 min 则只写 gate.json 并退出, 不开始正式训练
#   3) 依次 x1 -> x2 -> x074: 训练(2 epoch, save_steps=50) -> test 推理, 每组写完成标记
#   4) 三组齐全后评测 + 生成 patch_zoom_report.md
# 重提时: 已完成组按 done/<scale>.done 跳过; 未完成组从最近 checkpoint 续训

set -x
cd /home/zhao/Mammography/Mammo

module purge
module load NeSI
module load Python/3.14.4-foss-2026
source /nesi/project/massey04900/Mammography/venv/bin/activate

LOGDIR=/home/zhao/Mammography/Mammo/2-view/logs
OUT=/home/zhao/Mammography/Mammo/2-view/outputs/patch_zoom
mkdir -p "$LOGDIR" "$OUT"
LOG=$LOGDIR/patch_zoom_all_${SLURM_JOB_ID}.log

echo "=== node: $(hostname) start: $(date) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

python -u 2-view/scripts/patch_zoom/run_all.py \
  --out "$OUT" \
  --model-root /nesi/project/massey04900/Mammography/model \
  --logdir "$LOGDIR" \
  --gate-min 105 \
  --budget-min 112 \
  --smoke-n 16 \
  --epochs 2.0 \
  --stages all 2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "=== driver rc=$rc end: $(date) ===" | tee -a "$LOG"
exit $rc
