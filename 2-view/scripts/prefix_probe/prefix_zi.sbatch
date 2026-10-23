#!/bin/bash
#SBATCH --job-name=prefix_zi
#SBATCH --partition=genoa
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/prefix_zi_%j.log

# zero/inject 前缀干预: os_sub, test 2000, 只推理
set -u
cd /home/zhao/Mammography/Mammo/2-view/scripts/prefix_probe
LOG=/home/zhao/Mammography/Mammo/2-view/logs/prefix_zi_${SLURM_JOB_ID}.log

module purge 2>/dev/null
module load Python/3.14.4-foss-2026
source /nesi/project/massey04900/Mammography/venv/bin/activate

echo "=== [1/3] smoke (各 5 例) ===" | tee -a "$LOG"
python run_probe.py --mode smoke --stages prefix_zero,prefix_inject --skip-report 2>&1 | tee -a "$LOG"

echo "=== [2/3] full (2000 x 2 条件, 断点续跑) ===" | tee -a "$LOG"
python run_probe.py --mode full --stages prefix_zero,prefix_inject --skip-report --budget-min 150 2>&1 | tee -a "$LOG"

echo "=== [3/3] 分析 (CPU) ===" | tee -a "$LOG"
python analyze_prefix_zi.py 2>&1 | tee -a "$LOG"

echo "=== DONE ===" | tee -a "$LOG"
