#!/bin/bash
#SBATCH --job-name=norule
#SBATCH --partition=genoa
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/norule_%j.log

# norule prompt 消融: os_sub, test 2000, 只推理 (infer -> greedy 打分 -> zero/inject 前缀 -> 分析)
set -u
cd /home/zhao/Mammography/Mammo/2-view/scripts/prefix_probe
LOG=/home/zhao/Mammography/Mammo/2-view/logs/norule_${SLURM_JOB_ID}.log

module purge 2>/dev/null
module load Python/3.14.4-foss-2026
source /nesi/project/massey04900/Mammography/venv/bin/activate

echo "=== [1/2] smoke (各 5 例, 不分析) ===" | tee -a "$LOG"
python run_norule.py --phase all --limit 5 --skip-analyze 2>&1 | tee -a "$LOG"

echo "=== [2/2] full (2000, 断点续跑; 预算 240-25=215min) ===" | tee -a "$LOG"
python run_norule.py --phase all --limit 0 --budget-min 215 2>&1 | tee -a "$LOG"

echo "=== DONE ===" | tee -a "$LOG"
