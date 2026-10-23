#!/bin/bash
# 原分辨率 ep2 冒烟：10 步、第 5 步存 ckpt、第 8 步 kill、自动续跑到 10、再用 checkpoint-10 推理 16 条
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
CLI=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/llamafactory-cli
CFG=$PROJ/configs/_smoke_native_ep2.yaml
OUT=/home/z/zhao/Mammography/model/_smoke_native_ep2
LOG=$PROJ/chain_v1/logs/smoke_native.log
STATUS=$PROJ/outputs/eval/native_status.md
cd "$PROJ" || exit 1
mkdir -p chain_v1/logs outputs/eval outputs/scoring
ts() { date +"%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG" | tee -a "$STATUS"; }

# 从正式配置派生冒烟配置
"$PY" - <<'PYEOF'
src = "/home/z/zhao/Mammography/Mammo/2-view/configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml"
dst = "/home/z/zhao/Mammography/Mammo/2-view/configs/_smoke_native_ep2.yaml"
s = open(src, encoding="utf-8").read()
s = s.replace("output_dir: /home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep2-native",
              "output_dir: /home/z/zhao/Mammography/model/_smoke_native_ep2")
s = s.replace("save_steps: 200", "save_steps: 5")
s = s.replace("save_total_limit: 2", "save_total_limit: 1")
s = s.replace("logging_steps: 10", "logging_steps: 1\nmax_steps: 10")
open(dst, "w", encoding="utf-8").write(s)
print("smoke config written")
PYEOF

rm -rf "$OUT"
: > "$LOG"
grep -q "native_status" "$STATUS" 2>/dev/null || echo "# 原分辨率 ep2 训练/评测状态" > "$STATUS"

say "冒烟开始（10 步，第 5 步存 ckpt，第 8 步 kill 后自动续跑）"
bash scripts/bench/gpu_sampler.sh "$PROJ/chain_v1/logs/smoke_gpu.txt" &
SPID=$!

"$CLI" train "$CFG" >> "$LOG" 2>&1 &
TPID=$!
say "train pid=$TPID"

# 等到第 8 步
for i in $(seq 1 240); do
  STEP=$(tr '\r' '\n' < "$LOG" | grep -oE "[0-9]+/10 \[[0-9:]+<" | tail -1 | grep -oE "^[0-9]+" || true)
  if [ -n "${STEP:-}" ] && [ "$STEP" -ge 8 ]; then break; fi
  sleep 5
done
say "到达第 ${STEP:-?} 步，kill 训练进程"
pkill -f "$CFG" 2>/dev/null; kill $TPID 2>/dev/null
sleep 5
CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
say "已存 checkpoint-${CKPT:-none}；开始续跑"

"$CLI" train "$CFG" >> "$LOG" 2>&1 &
TPID2=$!
for i in $(seq 1 240); do
  if tr '\r' '\n' < "$LOG" | grep -q "train_runtime"; then break; fi
  sleep 5
done
sleep 3
kill $SPID 2>/dev/null
PEAK=$(sort -n "$PROJ/chain_v1/logs/smoke_gpu.txt" 2>/dev/null | tail -1)
say "续跑结束；nvidia-smi 峰值=${PEAK} MiB"
grep -oE "PEAK_ALLOCATED=[0-9.]+ GiB[^\"]*" "$LOG" | tail -1 | tee -a "$STATUS"

say "用 checkpoint 推理 16 条（--image-max-pixels 1376256）"
"$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$OUT" \
  --data data/splits/test_local.json --limit 16 --batch-size 8 --image-max-pixels 1376256 \
  --out-suffix _smoke_native_16 \
  --out-pred outputs/predictions/predictions_qwen3.5-4b_smoke_native_16.jsonl \
  --out-probs outputs/scoring/smoke_native_16_with_probs.jsonl \
  --tag smoke-native 2>&1 | tail -3 | tee -a "$LOG"
N=$(wc -l < outputs/scoring/smoke_native_16_with_probs.jsonl 2>/dev/null || echo 0)
say "推理产出 ${N}/16 条"
head -c 400 outputs/predictions/predictions_qwen3.5-4b_smoke_native_16.jsonl 2>/dev/null | tee -a "$LOG"

if [ "$N" -ge 16 ]; then
  say "冒烟 PASS（步数/续跑/推理三项）"
else
  say "冒烟 FAIL（推理行数 ${N}）"
fi
