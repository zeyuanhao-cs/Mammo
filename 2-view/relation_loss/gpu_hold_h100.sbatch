#!/bin/bash
#SBATCH --account=massey04900
#SBATCH --job-name=gpu_hold_h100
#SBATCH --partition=genoa
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/zhao/Mammography/Mammo/2-view/logs/gpu_hold_%j.log

# 占位作业: 持有 H100 allocation 6 小时。
# 所有任务通过 srun --jobid $SLURM_JOB_ID --overlap 在此 allocation 内执行,
# 单个任务失败不影响 allocation。
sleep 21590
