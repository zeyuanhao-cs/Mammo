#!/bin/bash
# =============================================================================
# round3 看门狗: 用户把 output_dir 改为 /hy-tmp/9_9/Qwen/qwen-4b/lora/,
# 正在运行的 round3.sh 检查的是旧路径, 训练结束后会误判中止。
# 本脚本轮询训练进程 + 新路径 adapter, 训练完成后自动推理前 500 条。
# =============================================================================
set -u
cd /hy-tmp/9_9/Mammo

FT=/hy-tmp/9_9/Mammo/vindr_finetune
ADAPTER_DIR=/hy-tmp/9_9/Qwen/qwen-4b/lora
ADAPTER="$ADAPTER_DIR/adapter_model.safetensors"

echo "[watch3] 等待训练完成 (输出目录: $ADAPTER_DIR)"
while true; do
  if ! pgrep -f "llamafactory-cli train" > /dev/null 2>&1; then
    if [ -f "$ADAPTER" ]; then
      echo "[watch3] $(date '+%F %T') 训练完成且 adapter 已保存"
      break
    fi
    echo "[watch3] 训练进程退出但暂未找到 adapter, 30s 后复查..."
    sleep 30
    if [ ! -f "$ADAPTER" ]; then
      echo "[watch3] 错误: 训练结束但未保存 adapter, 中止 (查看 train_round3.log)"
      exit 1
    fi
    break
  fi
  echo "[watch3] $(date '+%F %T') 训练进行中..."
  sleep 300
done

echo "[watch3] ===== 推理 direct_test.json 前 500 条 ====="
python3 -u vindr_finetune/infer_full_test.py --no-wait --limit 500 > "$FT/infer_round3.log" 2>&1
echo "[watch3] $(date '+%F %T') 推理退出码: $?"

echo "[watch3] ===== 全部完成 ====="
wc -l "$FT/outputs/direct_full/predictions.jsonl" 2>/dev/null || echo "无输出文件, 请检查 infer_round3.log"
