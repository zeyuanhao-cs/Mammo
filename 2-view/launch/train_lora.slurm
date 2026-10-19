#!/bin/bash
# 2-view LoRA 微调: Qwen3.5-4B, 全量 7999 条, 2 epochs
# 提交: sbatch train_lora.slurm
#SBATCH --account=massey04900
#SBATCH --job-name=2view_lora_train
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/train_%j.out
#SBATCH --error=/home/zhao/Mammography/Mammo/2-view/logs/train_%j.err

set -e
cd /home/zhao/Mammography/Mammo

module purge
module load PyTorch/2.13.0-foss-2026-Python-3.14.4-CUDA-13.2.1
source /nesi/project/massey04900/Mammography/venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi || true
echo "=== start: $(date) ==="

llamafactory-cli train 2-view/configs/train_lora_2view.yaml

echo "=== end: $(date) ==="
ls -la /nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view/
