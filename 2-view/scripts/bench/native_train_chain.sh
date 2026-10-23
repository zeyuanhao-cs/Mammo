#!/bin/bash
# 原分辨率 ep2（896x1536, image_max_pixels=1376256）训练链
#   --smoke : 10 步冒烟（第 5 步存 ckpt、第 8 步 kill、续跑到 10、16 条推理、删临时目录）
#   默认    : 正式训练（最多 3 次自动续跑）+ 训练完 b8 推理测试集 + 出报告
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
CLI=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/llamafactory-cli
CFG=$PROJ/configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml
OUT=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep2-native
STATUS=$PROJ/outputs/eval/native_status.md
LOG=$PROJ/logs/vlm/native_ep2.log
mkdir -p "$PROJ/logs/vlm" "$PROJ/outputs/eval" "$PROJ/outputs/scoring"

cd "$PROJ" || exit 1
ts() { date +"%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG" | tee -a "$STATUS"; }
[ -f "$STATUS" ] || echo "# 原分辨率 ep2 训练/评测状态" > "$STATUS"

# 环境安全检查：不允许 expandable_segments
if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  say "ERROR: PYTORCH_CUDA_ALLOC_CONF 已设置（${PYTORCH_CUDA_ALLOC_CONF}），本机不支持，退出"
  exit 2
fi

latest_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }

# ---------------- 冒烟 ----------------
if [ "${1:-}" = "--smoke" ]; then
  SMOKE_CFG=$PROJ/configs/_smoke_native_ep2.yaml
  SMOKE_OUT=/home/z/zhao/Mammography/model/_smoke_native_ep2
  "$PY" - <<PYEOF
src = "$CFG"
dst = "$SMOKE_CFG"
s = open(src, encoding="utf-8").read()
s = s.replace("output_dir: $OUT", "output_dir: $SMOKE_OUT")
s = s.replace("save_steps: 200", "save_steps: 5")
s = s.replace("save_total_limit: 2", "save_total_limit: 1")
s = s.replace("logging_steps: 10", "logging_steps: 1\nmax_steps: 10")
open(dst, "w", encoding="utf-8").write(s)
PYEOF
  rm -rf "$SMOKE_OUT"; : > "$LOG"
  say "冒烟开始：10 步，第 5 步存 ckpt，第 8 步 kill"
  bash scripts/bench/gpu_sampler.sh "$PROJ/logs/vlm/smoke_gpu.txt" & SPID=$!
  "$CLI" train "$SMOKE_CFG" >> "$LOG" 2>&1 & TPID=$!
  for i in $(seq 1 300); do
    STEP=$(tr '\r' '\n' < "$LOG" | grep -oE "[0-9]+/10 \[[0-9:]+<" | tail -1 | grep -oE "^[0-9]+" || true)
    [ -n "${STEP:-}" ] && [ "$STEP" -ge 8 ] && break
    sleep 5
  done
  say "第 ${STEP:-?} 步：kill 训练，随后自动续跑"
  kill $TPID 2>/dev/null; pkill -f "$SMOKE_CFG" 2>/dev/null; sleep 6
  CK=$(latest_ckpt "$SMOKE_OUT"); say "可用 checkpoint-${CK:-none}"
  "$CLI" train "$SMOKE_CFG" >> "$LOG" 2>&1 & TPID2=$!
  for i in $(seq 1 360); do
    tr '\r' '\n' < "$LOG" | grep -q "train_runtime" && break
    sleep 5
  done
  sleep 3; kill $SPID 2>/dev/null
  PEAK=$(sort -n "$PROJ/logs/vlm/smoke_gpu.txt" 2>/dev/null | tail -1)
  say "冒烟训练结束；nvidia-smi 峰值=${PEAK:-?} MiB"
  grep -oE "PEAK_ALLOCATED=[0-9.]+ GiB[^|]*" "$LOG" | tail -1 | tee -a "$STATUS"
  grep -c "expandable_segments" "$LOG" | tee -a "$STATUS"
  say "用冒烟 checkpoint 推理 16 条（image_max_pixels=1376256）"
  "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$SMOKE_OUT" \
    --data data/splits/test_local.json --limit 16 --batch-size 8 --image-max-pixels 1376256 \
    --out-suffix _smoke_native_16 \
    --out-pred outputs/predictions/predictions_qwen3.5-4b_smoke_native_16.jsonl \
    --out-probs outputs/scoring/smoke_native_16_with_probs.jsonl --tag smoke-native 2>&1 | tail -3 | tee -a "$LOG"
  N=$(wc -l < outputs/scoring/smoke_native_16_with_probs.jsonl 2>/dev/null || echo 0)
  say "推理产出 ${N}/16 条"
  if [ "$N" -ge 16 ]; then say "冒烟 PASS"; else say "冒烟 FAIL（推理 ${N} 行）"; exit 3; fi
  head -c 300 outputs/predictions/predictions_qwen3.5-4b_smoke_native_16.jsonl | tee -a "$LOG"
  rm -rf "$SMOKE_OUT" "$SMOKE_CFG"
  say "临时目录已删除"
  exit 0
fi

# ---------------- 正式训练 ----------------
say "正式训练开始（原分辨率 896x1536, 2 epoch, 每 200 步存, keep 2）"
bash scripts/bench/gpu_sampler.sh "$PROJ/logs/vlm/native_ep2_gpu.txt" & SPID=$!
OK=0
for n in 1 2 3; do
  CK=$(latest_ckpt "$OUT")
  if [ -n "${CK:-}" ]; then say "第 $n 次尝试：从 checkpoint-$CK 续跑"; else say "第 $n 次尝试：从头开始"; fi
  "$CLI" train "$CFG" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then say "第 $n 次尝试成功"; OK=1; break; fi
  say "第 $n 次尝试失败 rc=$rc；尾部日志："
  tail -5 "$LOG" | while read -r line; do say "  $line"; done
  sleep 30
done
kill $SPID 2>/dev/null
PEAK=$(sort -n "$PROJ/logs/vlm/native_ep2_gpu.txt" 2>/dev/null | tail -1)
say "训练阶段结束 ok=$OK；nvidia-smi 峰值=${PEAK:-?} MiB"
grep -oE "PEAK_ALLOCATED=[0-9.]+ GiB[^|]*" "$LOG" | tail -1 | tee -a "$STATUS" || true

# ---------------- 推理 ----------------
PROBS=$PROJ/outputs/scoring/vt-ep2-native_test_b8_with_probs.jsonl
PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep2-native_test_b8_2000.jsonl
if [ "$(wc -l < "$PROBS" 2>/dev/null || echo 0)" -lt 2000 ]; then
  say "b8 推理开始（2000 条，image_max_pixels=1376256）"
  "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$OUT" \
    --data data/splits/test_local.json --limit 0 --batch-size 8 --image-max-pixels 1376256 \
    --out-suffix _vt-ep2-native_test_b8 --out-pred "$PRED" --out-probs "$PROBS" \
    --tag vt-ep2-native 2>&1 | tail -3 | tee -a "$LOG"
fi
say "推理结束：$(wc -l < "$PROBS" 2>/dev/null || echo 0)/2000"

if [ "$(wc -l < "$PROBS" 2>/dev/null || echo 0)" -ge 2000 ]; then
  "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" \
    --out "$PROJ/outputs/eval/metrics_probs_vt-ep2-native_test_b8.json" \
    --tag vt-ep2-native --n-boot 1000 2>&1 | tail -3 | tee -a "$LOG"
  say "生成报告 ep2_native_report.md"
  "$PY" scripts/bench/ep2_native_report.py 2>&1 | tail -5 | tee -a "$LOG"
  say "报告完成"
else
  say "推理不完整，跳过报告"
fi
