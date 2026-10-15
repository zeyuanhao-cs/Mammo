#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --job-name=cnn_lowres
#SBATCH --partition=genoa
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/train_cnn_lowres_%j.log

# EfficientNet-B2 repeat_os 低分辨率版: 912x1520 -> 684x1140 (779,760 px, 与 VLM 786,432 上限对齐)
# 其余设置与原版完全一致 (训练集/batch 8x2/lr/早停/seed 42/16 epochs)。
# 输出: 2-view/outputs/vision_baseline/repeat_os_lowres/
# 流程: CPU 2-sample 快速验证 resize 路径 -> GPU smoke 64条x2ep -> 正式训练(OOM 则 4x4 续跑)
#       -> fit_thresholds + eval_test (tag cnn_repeat_os_lowres)

set -e
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load Python/3.14.4-foss-2026
PY=/nesi/project/massey04900/Mammography/venv/bin/python
SCRIPT=/home/zhao/Mammography/Mammo/2-view/scripts/train_vision_baseline.py
OUT=/home/zhao/Mammography/Mammo/2-view/outputs/vision_baseline/repeat_os_lowres
EVAL_DIR=/home/zhao/Mammography/Mammo/2-view/outputs/eval_cnn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 60 \
  > /home/zhao/Mammography/Mammo/2-view/logs/train_cnn_lowres_gpu_$SLURM_JOB_ID.csv 2>&1 &
POLL_PID=$!
trap "kill $POLL_PID 2>/dev/null || true" EXIT

echo "=== [0] GPU smoke 64条 x 2 epoch (lowres) ==="
$PY -u $SCRIPT --version repeat_os --out-suffix _lowres --img-size 684x1140 \
  --smoke --epochs 2 --batch-size 8 --grad-accum 2 --num-workers 11

echo "=== [1] 正式训练 (batch 8 x accum 2, 同原版) ==="
set +e
$PY -u $SCRIPT --version repeat_os --out-suffix _lowres --img-size 684x1140 \
  --epochs 16 --batch-size 8 --grad-accum 2 --num-workers 11
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "=== [1] 失败(rc=$RC), 检查是否 OOM ==="
  if tail -50 /home/zhao/Mammography/Mammo/2-view/logs/train_cnn_lowres_$SLURM_JOB_ID.log | \
      grep -q "CUDA out of memory\|OutOfMemoryError"; then
    echo "=== [1b] OOM: 改 batch 4 x accum 4 (等效 batch 16 不变), last.pt 断点续跑 ==="
    $PY -u $SCRIPT --version repeat_os --out-suffix _lowres --img-size 684x1140 \
      --epochs 16 --batch-size 4 --grad-accum 4 --num-workers 11
  else
    echo "=== 非 OOM 失败, 停止 ==="; exit $RC
  fi
fi

echo "=== [2] 评估: thresholds (val) + test_eval ==="
$PY /home/zhao/Mammography/Mammo/2-view/scripts/eval/fit_thresholds.py \
  --val $OUT/val_probs.jsonl --out $EVAL_DIR/cnn_repeat_os_lowres_thresholds.json
$PY /home/zhao/Mammography/Mammo/2-view/scripts/eval/eval_test.py \
  --test $OUT/test_probs.jsonl --thresholds $EVAL_DIR/cnn_repeat_os_lowres_thresholds.json \
  --out $EVAL_DIR/cnn_repeat_os_lowres_test_eval.json --tag cnn_repeat_os_lowres

echo "=== cnn_lowres job done ==="
