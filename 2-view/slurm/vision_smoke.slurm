#!/bin/bash
# 视觉 baseline 冒烟测试: 64 条 x 2 epoch + val/test 小批量推理 + jsonl 输出
#SBATCH --account=massey04900
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=vis_smoke
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/slurm_vis_smoke_%j.out

export LD_LIBRARY_PATH=/opt/nesi/zen3/Arrow/24.0.0-GCC-15.2.0/lib:/opt/nesi/zen3/Python/3.14.4-foss-2026/lib:/opt/nesi/zen3/FlexiBLAS/3.5.0-GCC-15.2.0/lib:/opt/nesi/zen3/OpenBLAS/0.3.32-GCC-15.2.0/lib:/opt/nesi/zen3/GCCcore/15.2.0/lib64:$LD_LIBRARY_PATH

source /nesi/project/massey04900/Mammography/venv/bin/activate
cd /home/zhao/Mammography/Mammo

echo "=== vision baseline smoke start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
python 2-view/scripts/train_vision_baseline.py --version natural --smoke --epochs 2 \
  2>&1 | tee 2-view/logs/vis_smoke.log
rc=$?
echo "=== smoke exit=$rc: $(date) ==="
exit $rc
