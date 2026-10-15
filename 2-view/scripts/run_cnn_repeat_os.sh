#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=cnn_repeat_os
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/train_cnn_repeat_os.log

# CNN repeat_os 版: 训练 (16 epoch, batch 8 x grad-accum 2, 早停 8/4) + 自动评估
# 与 natural 版同参数; 断点续跑内置 (last.pt)
LOGDIR=/home/zhao/Mammography/Mammo/2-view/logs
OUTDIR=/home/zhao/Mammography/Mammo/2-view/outputs

echo "$(date '+%F %T') job $SLURM_JOB_ID start on $SLURMD_NODENAME"

# ---- 1. 训练 ---- 
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load Python/3.14.4-foss-2026
/nesi/project/massey04900/Mammography/venv/bin/python -u \
  /home/zhao/Mammography/Mammo/2-view/scripts/train_vision_baseline.py \
  --version repeat_os --epochs 16 --batch-size 8 --grad-accum 2 --num-workers 7
TRAIN_RC=$?
echo "$(date '+%F %T') training exit code: $TRAIN_RC"

# ---- 2. 训练完成后评估 (纯 CPU, 系统 python3) ----
if [ -f "$OUTDIR/vision_baseline/repeat_os/DONE.json" ]; then
  echo "$(date '+%F %T') training DONE, run eval"
  cd /home/zhao/Mammography/Mammo/2-view/scripts/eval
  /usr/bin/python3 fit_thresholds.py \
    --val "$OUTDIR/vision_baseline/repeat_os/val_probs.jsonl" \
    --out "$OUTDIR/eval_cnn/cnn_repeat_os_thresholds.json" 2>&1 | tee "$LOGDIR/eval_cnn_repeat_os.log"
  /usr/bin/python3 eval_test.py \
    --test "$OUTDIR/vision_baseline/repeat_os/test_probs.jsonl" \
    --thresholds "$OUTDIR/eval_cnn/cnn_repeat_os_thresholds.json" \
    --out "$OUTDIR/eval_cnn/cnn_repeat_os_test_eval.json" \
    --tag cnn_repeat_os 2>&1 | tee -a "$LOGDIR/eval_cnn_repeat_os.log"
  echo "$(date '+%F %T') ALL DONE"
else
  echo "$(date '+%F %T') no DONE.json - training incomplete, eval skipped"
fi
