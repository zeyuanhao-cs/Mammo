#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --job-name=dens_probe_d
#SBATCH --partition=genoa
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/density_probe_d_%j.log

# 密度前缀干预补跑: A/B/C 已完成 (各2000行), D 只剩 7/2000, 断点续跑后生成最终报告。
set -e
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load Python/3.14.4-foss-2026
PY=/nesi/project/massey04900/Mammography/venv/bin/python
cd /home/zhao/Mammography/Mammo/2-view/scripts/prefix_probe

$PY -u run_density_probe.py --mode full --budget-min 45 --skip-report
$PY -u run_density_probe.py --report-only
echo "=== density probe D 补跑 + 报告 done ==="
