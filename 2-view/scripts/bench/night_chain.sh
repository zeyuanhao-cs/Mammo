#!/bin/bash
# 夜间串行链 v2：每个阶段自查，失败即停并写 outputs/eval/overnight_status.md
#   A  ep2 训练检查 (rc/adapter/global_step=878/no nan)
#   G  ep2-e2 验证集 gate (1600 条, parse_fail<=16, density AUC>=0.92, birads>=0.80, findings>=0.78)
#   B  原生分辨率训练 (最多 3 次; 3 次都失败则跳过原生评测但继续 ep2 评测)
#   C1 ep2 其余评测 (e1 val/test, e2 test)  行数检查 1600/2000
#   C2 原生评测 (val/test, batch 8->4->2)   行数检查 1600/2000
#   D  仅在输入齐全时生成 resolution_report.md
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
export PATH=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin:$PATH
LLAMAFACTORY_CLI=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/llamafactory-cli
BASE=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt
EP2=$BASE-ep2
NATIVE=$BASE-ep2-native
CFG=$PROJ/configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml
CHAIN=$PROJ/logs/vlm/night_chain.log
TLOG=$PROJ/logs/vlm/native_train.log
GPU=$PROJ/logs/vlm/native_train_gpu.txt
BL=$PROJ/logs/vlm/native_eval_batch.txt
EP2LOG=$PROJ/logs/vlm/ep2_chain.log
STATUS=$PROJ/outputs/eval/overnight_status.md
VAL=$PROJ/data/splits/val_raw_local.json
TEST=$PROJ/data/splits/test_local.json
EXPECT_STEPS=878

cd "$PROJ" || exit 1
mkdir -p logs/vlm outputs/eval outputs/scoring outputs/predictions
ts() { date +"%H:%M:%S"; }
now() { date +"%Y-%m-%d %H:%M:%S"; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
say() { echo "[$(ts)] $*" | tee -a "$CHAIN"; }
st() { printf '%s\n' "$*" >> "$STATUS"; }
stamp_line() { st "- \`$(now)\` $*"; }

[ -f "$STATUS" ] || printf '# 夜间链运行状态（自动生成）\n\n' > "$STATUS"
st ""; st "---"; st "# 运行开始 $(now)"

phase_start() { st ""; st "## $1 开始 $(now)"; say "=== $1 开始 ==="; }
phase_ok()    { st "- ✅ $1 结束 $(now)"; say "=== $1 通过 ==="; }
phase_bad()   { st "- ❌ $1 $(now)：$2"; say "!!! $1 失败：$2"; }
stop_chain()  { st ""; st "## ⛔ 链已停止：$1"; st "- 停止时间 $(now)"; st "- GPU 已空闲，等待人工处理"; say "STOP: $1"; exit 1; }

# ---------- 训练检查 ----------
trainer_step() {
  "$PY" - "$1" <<'EOF' 2>/dev/null || echo err
import json, sys
try:
    print(json.load(open(sys.argv[1] + "/trainer_state.json")).get("global_step"))
except Exception:
    print("err")
EOF
}

has_nan() {  # $1=log file
  [ -f "$1" ] || return 1
  tr '\r' '\n' < "$1" | grep -qiE "'loss': *'(nan|-?inf)'" && return 0
  tr '\r' '\n' < "$1" | grep -qiE "^ *loss.*(nan|inf)" && return 0
  return 1
}

check_train() {  # $1=label $2=outdir $3=log $4=rc
  local label=$1 out=$2 logf=$3 rc=$4 reason=""
  if [ "$rc" -ne 0 ]; then
    phase_bad "$label" "训练进程非 0 退出 rc=$rc"
    return 1
  fi
  if [ ! -f "$out/adapter_model.safetensors" ]; then
    phase_bad "$label" "缺少 $out/adapter_model.safetensors"
    return 1
  fi
  local gs
  gs=$(trainer_step "$out")
  if [ "$gs" != "$EXPECT_STEPS" ]; then
    phase_bad "$label" "trainer_state.global_step=$gs（期望 $EXPECT_STEPS）"
    return 1
  fi
  if has_nan "$logf"; then
    phase_bad "$label" "训练日志出现 nan/inf loss"
    return 1
  fi
  phase_ok "$label"
  return 0
}

# ---------- 评测检查 ----------
AUC() {  # $1=metrics json  $2=key(表达式)
  "$PY" - "$1" "$2" <<'EOF' 2>/dev/null || echo err
import json, sys
m = json.load(open(sys.argv[1]))
k = sys.argv[2]
v = {"density": m["density_4class"]["ovr_macro_auc"],
     "birads": m["birads_ovr_macro_auc"],
     "findings": m["findings_macro_auroc"]}[k]
print(f"{v:.6f}")
EOF
}

parse_fails() {  # $1=pred jsonl
  [ -f "$1" ] || { echo 999999; return; }
  "$PY" - "$1" <<'EOF' 2>/dev/null || echo 999999
import json, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        if json.loads(line).get("prediction_json") is None:
            n += 1
    except Exception:
        n += 1
print(n)
EOF
}

# ================= Phase A: ep2 训练 =================
phase_start "A. ep2 训练 (786432, 2 epoch)"
while ! grep -q "EP2_TRAIN_DONE" "$EP2LOG" 2>/dev/null; do
  if ! pgrep -f "ep2_chain.sh" >/dev/null && ! tmux has-session -t ep2 2>/dev/null; then
    phase_bad "A. ep2 训练" "进程消失且无 EP2_TRAIN_DONE 标记"
    stop_chain "ep2 训练异常终止（无完成标记）"
  fi
  sleep 60
done
EP2_RC=$(grep -o "EP2_TRAIN_DONE rc=[0-9]*" "$EP2LOG" | tail -1 | cut -d= -f2)
stamp_line "ep2 训练进程返回 rc=$EP2_RC"
check_train "A. ep2 训练" "$EP2" "$EP2LOG" "${EP2_RC:-1}" || stop_chain "ep2 训练检查未通过"

# ================= Phase G: ep2-e2 验证集 gate =================
phase_start "G. ep2-e2 验证集 gate (b8, 1600 条)"
G_PROBS=$PROJ/outputs/scoring/vt-ep2-e2_val_b8_with_probs.jsonl
G_PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep2-e2_val_b8.jsonl
G_METRICS=$PROJ/outputs/eval/metrics_probs_vt-ep2-e2_val_b8.json
if [ "$(nlines "$G_PROBS")" -lt 1600 ]; then
  say "G: 推理 ep2-e2/val"
  "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$EP2/checkpoint-878" \
    --data "$VAL" --limit 0 --batch-size 8 --out-suffix "_vt-ep2-e2_val_b8" \
    --out-pred "$G_PRED" --out-probs "$G_PROBS" --tag "gate-ep2-e2-val" 2>&1 | tail -3 | tee -a "$CHAIN"
fi
if [ ! -f "$G_METRICS" ]; then
  "$PY" scripts/eval/summarize_probs.py --probs "$G_PROBS" --out "$G_METRICS" \
    --tag "vt-ep2-e2-val" --n-boot 1000 2>&1 | tail -3 | tee -a "$CHAIN"
fi

G_LINES=$(nlines "$G_PROBS")
G_PF=$(parse_fails "$G_PRED")
G_D=$(AUC "$G_METRICS" density); G_B=$(AUC "$G_METRICS" birads); G_F=$(AUC "$G_METRICS" findings)
stamp_line "G 结果：lines=$G_LINES/1600, parse_fail=$G_PF, density=$G_D, birads=$G_B, findings=$G_F"
G_REASON=""
[ "$G_LINES" -ge 1600 ] || G_REASON="验证集只完成 $G_LINES/1600 条"
[ "$G_REASON" = "" ] && [ "$G_PF" -gt 16 ] && G_REASON="解析失败 $G_PF > 16"
if [ "$G_REASON" = "" ]; then
  "$PY" - "$G_D" "$G_B" "$G_F" <<'EOF' >/dev/null 2>&1 || G_REASON="AUC 未达阈值"
import sys
d, b, f = map(float, sys.argv[1:4])
sys.exit(0 if (d >= 0.92 and b >= 0.80 and f >= 0.78) else 1)
EOF
  [ "$G_REASON" != "" ] && G_REASON="AUC 未达阈值（density=$G_D≥0.92? birads=$G_B≥0.80? findings=$G_F≥0.78?）"
fi
if [ "$G_REASON" != "" ]; then
  phase_bad "G. ep2-e2 val gate" "$G_REASON"
  stop_chain "ep2 训练质量未达标：$G_REASON"
fi
phase_ok "G. ep2-e2 val gate"

# ================= Phase B: 原生分辨率训练 =================
phase_start "B. 原生分辨率训练 (1386240, 2 epoch, 最多 3 次)"
bash scripts/bench/gpu_sampler.sh "$GPU" &
SPID=$!
NATIVE_OK=0
for n in 1 2 3; do
  say "原生训练 尝试 $n/3"
  : > "$TLOG"
  "$LLAMAFACTORY_CLI" train "$CFG" >> "$TLOG" 2>&1
  rc=$?
  stamp_line "原生训练 尝试 $n 返回 rc=$rc"
  if check_train "B. 原生训练 尝试 $n" "$NATIVE" "$TLOG" "$rc"; then NATIVE_OK=1; break; fi
  tail -50 "$TLOG" > "logs/vlm/native_fail_${n}.log"
  st "  - 失败日志：\`logs/vlm/native_fail_${n}.log\`"
  sleep 30
done
kill $SPID 2>/dev/null
sleep 1
PEAK=$(sort -n "$GPU" 2>/dev/null | tail -1)
printf "NATIVE_TRAIN_OK=%s\nNATIVE_PEAK_SMI_MIB=%s\n" "$NATIVE_OK" "$PEAK" > logs/vlm/native_train_summary.txt
stamp_line "原生训练结束：ok=$NATIVE_OK, nvidia-smi 峰值=${PEAK} MiB"
if [ "$NATIVE_OK" -eq 1 ]; then phase_ok "B. 原生训练"; else phase_bad "B. 原生训练" "3 次尝试均未通过检查，跳过原生评测"; fi

# ================= Phase C1: ep2 其余评测 =================
phase_start "C1. ep2 评测 (e1 val/test, e2 test)"
MISSING=""
for pair in 439:e1 878:e2; do
  CKPT=${pair%%:*}
  TAG=${pair##*:}
  for split in val test; do
    if [ "$split" = "val" ]; then DATA=$VAL; N=1600; else DATA=$TEST; N=2000; fi
    PROBS=$PROJ/outputs/scoring/vt-ep2-${TAG}_${split}_b8_with_probs.jsonl
    PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep2-${TAG}_${split}_b8.jsonl
    METRICS=$PROJ/outputs/eval/metrics_probs_vt-ep2-${TAG}_${split}_b8.json
    if [ "$(nlines "$PROBS")" -lt "$N" ]; then
      say "ep2 infer $TAG/$split"
      "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$EP2/checkpoint-$CKPT" \
        --data "$DATA" --limit 0 --batch-size 8 --out-suffix "_vt-ep2-${TAG}_${split}_b8" \
        --out-pred "$PRED" --out-probs "$PROBS" --tag "vt-ep2-${TAG}-${split}" 2>&1 | tail -3 | tee -a "$CHAIN"
    fi
    if [ ! -f "$METRICS" ] && [ "$(nlines "$PROBS")" -ge "$N" ]; then
      "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
        --tag "vt-ep2-${TAG}-${split}" --n-boot 1000 2>&1 | tail -3 | tee -a "$CHAIN"
    fi
    L=$(nlines "$PROBS")
    if [ "$L" -lt "$N" ] || [ ! -f "$METRICS" ]; then
      MISSING="$MISSING ep2-${TAG}/${split}($L/$N)"
    fi
    stamp_line "ep2 ${TAG}/${split}: probs=$L/$N metrics=$([ -f "$METRICS" ] && echo ok || echo missing)"
  done
done
if [ "$MISSING" != "" ]; then phase_bad "C1. ep2 评测" "不完整：$MISSING"; else phase_ok "C1. ep2 评测"; fi
[ "$MISSING" = "" ] && { "$PY" scripts/bench/ep2_vs_ep3.py >/dev/null 2>&1 && st "- \`$(now)\` 已生成 outputs/eval/ep2_vs_ep3.md"; }

# ================= Phase C2: 原生评测 =================
CMISS="$MISSING"
if [ "$NATIVE_OK" -eq 1 ]; then
  phase_start "C2. 原生评测 (val/test, 1386240, batch 8->4->2)"
  : > "$BL"
  for split in val test; do
    if [ "$split" = "val" ]; then DATA=$VAL; N=1600; else DATA=$TEST; N=2000; fi
    PROBS=$PROJ/outputs/scoring/vt-ep2native_${split}_b8_with_probs.jsonl
    PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep2native_${split}_b8.jsonl
    METRICS=$PROJ/outputs/eval/metrics_probs_vt-ep2native_${split}_b8.json
    if [ "$(nlines "$PROBS")" -lt "$N" ]; then
      for bs in 8 4 2; do
        say "native infer $split batch=$bs"
        "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$NATIVE" \
          --data "$DATA" --limit 0 --batch-size "$bs" --image-max-pixels 1386240 \
          --out-suffix "_vt-ep2native_${split}_b8" \
          --out-pred "$PRED" --out-probs "$PROBS" --tag "ep2native-${split}" 2>&1 | tail -3 | tee -a "$CHAIN"
        rc=${PIPESTATUS[0]}
        echo "$split,batch=$bs,rc=$rc" >> "$BL"
        [ "$rc" -eq 0 ] && break
        tail -20 "$CHAIN" > "logs/vlm/native_eval_fail_${split}_${bs}.log"
      done
    fi
    if [ ! -f "$METRICS" ] && [ "$(nlines "$PROBS")" -ge "$N" ]; then
      "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
        --tag "ep2native-${split}" --n-boot 1000 2>&1 | tail -3 | tee -a "$CHAIN"
    fi
    L=$(nlines "$PROBS")
    if [ "$L" -lt "$N" ] || [ ! -f "$METRICS" ]; then CMISS="$CMISS native/${split}($L/$N)"; fi
    stamp_line "native ${split}: probs=$L/$N metrics=$([ -f "$METRICS" ] && echo ok || echo missing)"
  done
  [ "$CMISS" = "" ] && phase_ok "C2. 原生评测" || phase_bad "C2. 原生评测" "不完整：$CMISS"
else
  stamp_line "C2 跳过（原生训练未通过）"
fi

# ================= Phase D: 报告 =================
phase_start "D. 报告"
have_all=1
for f in \
  "$PROJ/outputs/eval/metrics_overfit_val_e3_b8.json" \
  "$PROJ/outputs/scoring/overfit_val_e3_b8_with_probs.jsonl" \
  "$PROJ/outputs/eval/metrics_probs_vt-ep3-e3_test_b8.json" \
  "$PROJ/outputs/scoring/vt-ep3-e3_test_b8_with_probs.jsonl" \
  "$PROJ/outputs/eval/metrics_probs_vt-ep2-e2_val_b8.json" \
  "$PROJ/outputs/scoring/vt-ep2-e2_val_b8_with_probs.jsonl" \
  "$PROJ/outputs/eval/metrics_probs_vt-ep2-e2_test_b8.json" \
  "$PROJ/outputs/scoring/vt-ep2-e2_test_b8_with_probs.jsonl" ; do
  [ -f "$f" ] || { have_all=0; CMISS="$CMISS $(basename "$f")"; }
done
if [ "$NATIVE_OK" -eq 1 ]; then
  for f in \
    "$PROJ/outputs/eval/metrics_probs_vt-ep2native_val_b8.json" \
    "$PROJ/outputs/scoring/vt-ep2native_val_b8_with_probs.jsonl" \
    "$PROJ/outputs/eval/metrics_probs_vt-ep2native_test_b8.json" \
    "$PROJ/outputs/scoring/vt-ep2native_test_b8_with_probs.jsonl" ; do
    [ -f "$f" ] || { have_all=0; CMISS="$CMISS $(basename "$f")"; }
  done
else
  have_all=0
  CMISS="$CMISS native(训练未通过)"
fi

if [ "$have_all" -eq 1 ]; then
  "$PY" scripts/bench/resolution_report.py 2>&1 | tail -5 | tee -a "$CHAIN"
  if [ -f "$PROJ/outputs/eval/resolution_report.md" ]; then
    phase_ok "D. 报告"; st "- 产物：\`outputs/eval/resolution_report.md\`"; st ""; st "## ✅ 全部完成 $(now)"
  else
    phase_bad "D. 报告" "脚本运行但未生成文件"; stop_chain "报告生成失败"
  fi
else
  phase_bad "D. 报告" "输入不齐全，未生成报告：$CMISS"
  st ""; st "## ⛔ 链结束（未生成报告）$(now)"; st "- 缺失：$CMISS"; st "- GPU 已空闲，等待人工处理"
fi
say "NIGHT_ALL_DONE"
