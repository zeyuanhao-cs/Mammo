#!/bin/bash
# 总控：等冒烟结束 → 三项自检 → 加 tokenized_path → 正式训练(每100步存, keep2, 最多3次续跑)
#       → b8 推理(2000, image_max_pixels=1376256) → summarize → 报告
# 任何一项失败：写 outputs/eval/native_status.md 并停止。
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
CLI=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/llamafactory-cli
CFG=$PROJ/configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml
OUT=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep2-native
TOK=$PROJ/data/tokenized/ep2_native
STATUS=$PROJ/outputs/eval/native_status.md
LOG=$PROJ/logs/vlm/native_ep2.log
MLOG=$PROJ/logs/vlm/native_master.log
PROBS=$PROJ/outputs/scoring/vt-ep2-native_test_b8_with_probs.jsonl
PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep2-native_test_b8_2000.jsonl
METRICS=$PROJ/outputs/eval/metrics_probs_vt-ep2-native_test_b8.json

cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs/vlm" "$PROJ/outputs/eval" "$PROJ/outputs/scoring" "$PROJ/outputs/predictions" "$TOK"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$MLOG" | tee -a "$STATUS"; }
die() { say "STOP：$*"; exit 1; }

# 产物校验：adapter 存在 且 trainer_state.global_step == 878
check_trained() {
  [ -f "$OUT/adapter_model.safetensors" ] || return 1
  GS=$("$PY" - "$OUT" <<'PYEOF2'
import json, sys
try:
    print(json.load(open(sys.argv[1] + "/trainer_state.json")).get("global_step"))
except Exception:
    print("err")
PYEOF2
)
  [ "${GS:-err}" = "878" ]
}

if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  die "PYTORCH_CUDA_ALLOC_CONF 已设置（${PYTORCH_CUDA_ALLOC_CONF}），本机不支持 expandable_segments"
fi

say "总控启动：等待冒烟测试（smoke_native）结束"

# ---------- A. 等冒烟进程结束 ----------
for i in $(seq 1 720); do
  if ! pgrep -f "native_train_chain.sh --smoke" >/dev/null && ! tmux has-session -t smoke_native 2>/dev/null; then
    break
  fi
  sleep 10
done
say "冒烟进程已结束"

# ---------- B. 三项自检 ----------
say "运行冒烟三项检查"
if "$PY" scripts/bench/check_smoke.py >> "$MLOG" 2>&1; then
  RES=$(tail -1 "$MLOG")
  say "冒烟自检 PASS：${RES}"
else
  REASON=$("$PY" scripts/bench/check_smoke.py 2>&1 | tail -2 | tr '\n' ' ')
  die "冒烟自检未通过：${REASON}"
fi

# 预处理是否在续跑时重跑（用日志里 training 块之间的 tokenizer 行判断）
TOK_N=$(tr '\r' '\n' < "$LOG" | grep -c "Running tokenizer on dataset")
say "冒烟日志中 tokenizer 渲染次数=${TOK_N}"

# ---------- C. 加 tokenized_path（单独给原分辨率，放在项目 data 下） ----------
if ! grep -q "^tokenized_path:" "$CFG"; then
  "$PY" - <<PYEOF
p = "$CFG"
s = open(p, encoding="utf-8").read()
s = s.replace("### dataset", "### dataset\ntokenized_path: $TOK   # 原分辨率专用缓存（每视角 token 数与 786432 不同，不能共用）")
open(p, "w", encoding="utf-8").write(s)
PYEOF
  say "已写入 tokenized_path=$TOK"
else
  say "tokenized_path 已存在，跳过"
fi

# 正式训练：每 100 步存、keep 2
"$PY" - <<PYEOF
p = "$CFG"
s = open(p, encoding="utf-8").read()
s = s.replace("save_steps: 200", "save_steps: 100")
open(p, "w", encoding="utf-8").write(s)
PYEOF

# ---------- D. 正式训练（最多 3 次） ----------
say "正式训练开始（896x1536, 2 epoch, save 每 100 步, keep 2）"
bash scripts/bench/gpu_sampler.sh "$PROJ/logs/vlm/native_ep2_gpu.txt" & SPID=$!
OK=0
for n in 1 2 3; do
  CK=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  if [ -n "${CK:-}" ]; then say "第 ${n}/3 次尝试：从 checkpoint-${CK} 续跑"; else say "第 ${n}/3 次尝试：从头开始"; fi
  "$CLI" train "$CFG" >> "$LOG" 2>&1
  rc=$?
  GS=$("$PY" - "$OUT" <<'PYEOF2'
import json, sys
try:
    print(json.load(open(sys.argv[1] + "/trainer_state.json")).get("global_step"))
except Exception:
    print("err")
PYEOF2
)
  if [ "$rc" -eq 0 ] && check_trained; then
    say "第 ${n}/3 次尝试成功（rc=0, global_step=${GS}, adapter 存在）"; OK=1; break
  fi
  ERR=$(tail -3 "$LOG" | tr '\n' ' ' | cut -c1-300)
  if [ "$rc" -eq 0 ]; then
    say "第 ${n}/3 次尝试 rc=0 但产物不合格（global_step=${GS}，adapter=$([ -f "$OUT/adapter_model.safetensors" ] && echo yes || echo no)）：${ERR}"
  else
    say "第 ${n}/3 次尝试失败 rc=${rc}（global_step=${GS}）：${ERR}"
  fi
  sleep 30
done
kill $SPID 2>/dev/null
PEAK=$(sort -n "$PROJ/logs/vlm/native_ep2_gpu.txt" 2>/dev/null | tail -1)
say "训练阶段结束 ok=${OK}；nvidia-smi 峰值=${PEAK:-?} MiB"
[ "$OK" -eq 1 ] || die "正式训练 3 次尝试均失败（产物/步数校验未过），最后错误：$(tail -3 "$LOG" | tr '\n' ' ' | cut -c1-300)"
check_trained || die "训练结束但产物校验失败（adapter 或 global_step≠878），不进入推理"
say "产物校验通过：adapter 存在，global_step=878"

# ---------- E. 推理 ----------
if [ "$(wc -l < "$PROBS" 2>/dev/null || echo 0)" -lt 2000 ]; then
  say "b8 推理开始（2000 条，image_max_pixels=1376256）"
  "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$OUT" \
    --data data/splits/test_local.json --limit 0 --batch-size 8 --image-max-pixels 1376256 \
    --out-suffix _vt-ep2-native_test_b8 --out-pred "$PRED" --out-probs "$PROBS" \
    --tag vt-ep2-native >> "$MLOG" 2>&1
fi
NL=$(wc -l < "$PROBS" 2>/dev/null || echo 0)
say "推理结束：${NL}/2000"
[ "$NL" -ge 2000 ] || die "推理不完整（${NL}/2000）"

"$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
  --tag vt-ep2-native --n-boot 1000 >> "$MLOG" 2>&1
say "metrics 已生成：$(basename "$METRICS")"

# ---------- F. 报告 ----------
say "生成 ep2_native_report.md"
if "$PY" scripts/bench/lesion_size_bins.py >> "$MLOG" 2>&1 && \
   "$PY" scripts/bench/ep2_native_report.py >> "$MLOG" 2>&1; then
  say "报告完成：outputs/eval/ep2_native_report.md"
else
  die "报告生成失败，见 $MLOG"
fi

say "全部完成"
