#!/bin/bash
# 独立 2-epoch 训练 (786432, output ...-vt-ep2)。只训练，评测由 night_chain.sh 在两次训练后统一跑。
# 用法: tmux new-session -d -s ep2 "bash scripts/bench/ep2_chain.sh"
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CFG=$PROJ/configs/train_lora_2view_os_sub_vt_ep2_local.yaml
OUT=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep2
LOG=$PROJ/logs/vlm/ep2_chain.log

cd "$PROJ" || exit 1
mkdir -p logs/vlm
ts() { date +"%H:%M:%S"; }

echo "[$(ts)] === EP2 CHAIN START (train only) ===" | tee -a "$LOG"
rc=0
if [ ! -f "$OUT/adapter_model.safetensors" ]; then
  echo "[$(ts)] training ep2 (2 epochs, 786432)" | tee -a "$LOG"
  llamafactory-cli train "$CFG" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
else
  echo "[$(ts)] skip training (adapter exists)" | tee -a "$LOG"
fi

echo "[$(ts)] EP2_TRAIN_DONE rc=$rc" | tee -a "$LOG"
echo "EP2_TRAIN_RC=$rc" > "$PROJ/logs/vlm/ep2_train_summary.txt"
