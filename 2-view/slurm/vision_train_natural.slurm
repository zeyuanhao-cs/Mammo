#!/bin/bash
# 视觉 baseline 正式训练: natural 版 (train_sub_raw.json 6399 条, 24 epoch)
#SBATCH --account=massey04900
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=vis_nat
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/slurm_vis_nat_%j.out

export LD_LIBRARY_PATH=/opt/nesi/zen3/Arrow/24.0.0-GCC-15.2.0/lib:/opt/nesi/zen3/Python/3.14.4-foss-2026/lib:/opt/nesi/zen3/FlexiBLAS/3.5.0-GCC-15.2.0/lib:/opt/nesi/zen3/OpenBLAS/0.3.32-GCC-15.2.0/lib:/opt/nesi/zen3/GCCcore/15.2.0/lib64:$LD_LIBRARY_PATH

source /nesi/project/massey04900/Mammography/venv/bin/activate
cd /home/zhao/Mammography/Mammo

echo "=== vision baseline [natural] start: $(date) ==="
python 2-view/scripts/train_vision_baseline.py --version natural --epochs 24 \
  2>&1 | tee -a 2-view/logs/vis_natural.log
rc=$?
echo "=== [natural] exit=$rc: $(date) ==="
exit $rc
