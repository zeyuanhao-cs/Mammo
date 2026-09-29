#!/bin/bash
# =============================================================================
# 第三轮全自动链路:
#   1. 训练 (train_balanced_2to1.json, 4657 条, 2:1 lesion:normal + 四类平衡)
#      - 输出覆盖 /hy-tmp/9_9/Qwen/qwen-4b/lora_lesion_1to3
#   2. 训练成功后推理 direct_test.json 前 500 条
# 全程无交互; 训练失败或 adapter 未更新则中止。
# =============================================================================
set -u
cd /hy-tmp/9_9/Mammo

FT=/hy-tmp/9_9/Mammo/vindr_finetune
ADAPTER_DIR=/hy-tmp/9_9/Qwen/qwen-4b/lora_lesion_1to3
ADAPTER="$ADAPTER_DIR/adapter_model.safetensors"

echo "[round3] ===== 步骤1: 训练 (4657 条, 2:1 + 四类平衡) ====="
START_TS=$(date +%s)
llamafactory-cli train vindr_finetune/trial.yaml > "$FT/train_round3.log" 2>&1
RC=$?
echo "[round3] $(date '+%F %T') 训练退出码: $RC (日志: train_round3.log)"

if [ $RC -ne 0 ] || [ ! -f "$ADAPTER" ]; then
  echo "[round3] 错误: 训练失败或 adapter 未保存, 中止后续推理"
  exit 1
fi

# mtime 校验: 确认 adapter 是本次训练新产物, 而非旧模型残留
MTIME=$(stat -c %Y "$ADAPTER")
if [ "$MTIME" -lt "$START_TS" ]; then
  echo "[round3] 错误: adapter 修改时间早于本次训练开始, 疑似旧模型残留, 中止"
  exit 1
fi
echo "[round3] 新 adapter 训练完成 (mtime 校验通过)"

echo "[round3] ===== 步骤2: 推理 direct_test.json 前 500 条 ====="
python3 -u vindr_finetune/infer_full_test.py --no-wait --limit 500 > "$FT/infer_round3.log" 2>&1
RC2=$?
echo "[round3] $(date '+%F %T') 推理退出码: $RC2 (日志: infer_round3.log)"

echo "[round3] ===== 全部完成 ====="
echo "[round3] 推理结果行数:"
wc -l "$FT/outputs/direct_full/predictions.jsonl" 2>/dev/null || echo "无输出文件, 请检查 infer_round3.log"
