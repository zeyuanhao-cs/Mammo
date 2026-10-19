#!/bin/bash
# 2-view 推理: Qwen3.5-4B 零样本, VinDr test.json 前 500 条
# 提交: sbatch run_infer_500.slurm
#SBATCH --account=massey04900
#SBATCH --job-name=2view_infer500
#SBATCH --partition=genoa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/infer_%j.out
#SBATCH --error=/home/zhao/Mammography/Mammo/2-view/logs/infer_%j.err

set -e
cd /home/zhao/Mammography/Mammo

module purge
module load PyTorch/2.13.0-foss-2026-Python-3.14.4-CUDA-13.2.1
source /nesi/project/massey04900/Mammography/venv/bin/activate

echo "=== node: $(hostname) ==="
echo "=== gpu: ==="
nvidia-smi || true
echo "=== start: $(date) ==="

python 2-view/scripts/vlm/infer_2view.py --limit 500

echo "=== end: $(date) ==="
