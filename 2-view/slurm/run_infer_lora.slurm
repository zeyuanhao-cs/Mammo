#!/bin/bash
# 2-view LoRA 推理: Qwen3.5-4B + 微调后的 LoRA, test.json 前 500 条
# 依赖训练作业完成后自动运行: sbatch --dependency=afterok:<train_jobid> run_infer_lora.slurm
#SBATCH --account=massey04900
#SBATCH --job-name=2view_lora_infer
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/infer_lora_%j.out
#SBATCH --error=/home/zhao/Mammography/Mammo/2-view/logs/infer_lora_%j.err

set -e
cd /home/zhao/Mammography/Mammo

ADAPTER=/nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view

module purge
module load PyTorch/2.13.0-foss-2026-Python-3.14.4-CUDA-13.2.1
source /nesi/project/massey04900/Mammography/venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi || true
echo "=== adapter: ==="
ls -la "$ADAPTER" | head -20
echo "=== start: $(date) ==="

python 2-view/scripts/infer_2view.py \
  --model qwen3.5-4b \
  --adapter "$ADAPTER" \
  --template qwen3_5_nothink \
  --out-suffix _lora \
  --limit 500

echo "=== end: $(date) ==="
