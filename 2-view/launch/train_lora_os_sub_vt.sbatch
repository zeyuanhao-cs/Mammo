#!/bin/bash
#SBATCH --job-name=lora_vt
#SBATCH --partition=genoa
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mail-type=NONE

# 解冻视觉塔的 os_sub 复刻训练 (Qwen3.5-4B LoRA, freeze_vision_tower: false)
# 输出: /nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt

set -x
cd /home/zhao/Mammography/Mammo

module purge
module load NeSI
source /nesi/project/massey04900/Mammography/venv/bin/activate

LOG=/home/zhao/Mammography/Mammo/2-view/logs/train_lora_vt_${SLURM_JOB_ID}.log

echo "=== node: $(hostname) gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ===" | tee -a "$LOG"

# 依据集群 dill 补丁环境, 单进程预处理已在原 os_sub 训练验证过;
# preprocessing_num_workers=8 + dill 补丁在 os_sub 时可用, 此处保持一致
llamafactory-cli train 2-view/configs/train_lora_2view_os_sub_vt.yaml 2>&1 | tee -a "$LOG"

rc=$?
echo "=== exit code: $rc ===" | tee -a "$LOG"
exit $rc
