#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --job-name=dens_probe_l4
#SBATCH --partition=genoa
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/density_probe_l4_%j.log

# 密度前缀干预: smoke (5x4 条件 + gate) -> gate 通过则全量 (budget 控制断点续跑)。
set -e
source /usr/share/lmod/lmod/init/bash
module load Python/3.14.4-foss-2026
PY=/nesi/project/massey04900/Mammography/venv/bin/python
cd /home/zhao/Mammography/Mammo/2-view/scripts/prefix_probe

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 60 \
  > /home/zhao/Mammography/Mammo/2-view/logs/density_probe_l4_gpu_$SLURM_JOB_ID.csv 2>&1 &
POLL_PID=$!
trap "kill $POLL_PID 2>/dev/null || true" EXIT

echo "=== density probe smoke (L4) ==="
$PY -u run_density_probe.py --mode smoke --skip-report

PASS=$($PY -c "import json; print(json.load(open('/home/zhao/Mammography/Mammo/2-view/outputs/prefix_probe/density_gate.json'))['pass'])")
echo "=== gate pass=$PASS ==="
if [ "$PASS" = "True" ]; then
  echo "=== density probe full (L4, budget 控制断点续跑) ==="
  $PY -u run_density_probe.py --mode full --budget-min 150 --skip-report
fi
echo "=== density probe job done ==="
