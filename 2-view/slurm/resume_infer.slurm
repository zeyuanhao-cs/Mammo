#!/bin/bash
# 续训 LoRA (checkpoint-400 → 500) + 500 条推理, 打包成一个 2h 批处理作业
#SBATCH --account=massey04900
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=lora_resume
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/slurm_%j.out

export LD_LIBRARY_PATH=/opt/nesi/zen3/Arrow/24.0.0-GCC-15.2.0/lib:/opt/nesi/zen3/Python/3.14.4-foss-2026/lib:/opt/nesi/zen3/FlexiBLAS/3.5.0-GCC-15.2.0/lib:/opt/nesi/zen3/OpenBLAS/0.3.32-GCC-15.2.0/lib:/opt/nesi/zen3/GCCcore/15.2.0/lib64:$LD_LIBRARY_PATH

source /nesi/project/massey04900/Mammography/venv/bin/activate
cd /home/zhao/Mammography/Mammo

echo "=== resume training start: $(date) ==="
llamafactory-cli train 2-view/configs/train_lora_2view.yaml || { echo "=== TRAIN_FAILED ==="; exit 1; }
echo "=== TRAIN_OK: $(date) ==="

echo "=== inference start: $(date) ==="
python 2-view/scripts/infer_2view.py \
  --model qwen3.5-4b \
  --adapter /nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view \
  --limit 500 \
  --out-suffix _lora_500 && echo "=== INFER_OK: $(date) ==="

echo "=== pipeline end: $(date) ==="
