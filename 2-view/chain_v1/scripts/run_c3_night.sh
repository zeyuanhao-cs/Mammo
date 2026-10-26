#!/usr/bin/env bash
# C3 夜间编排（新设计）：门控 → 端点核对 → C3full_s（与 B0 同起点同超参 + axis）vs B0
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
MDIR=/home/z/zhao/Mammography/model
STATUS=$CV1/outputs/pilot_status.md
C3A=$MDIR/chain_v1_C3a
C3F=$MDIR/chain_v1_C3full_s
say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { say "STOP: $*"; exit 1; }
rows_of() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# ---- 保险：GPU 阶段前确认无残留进程 + GPU 空闲 ----
gpu_guard() {
  pkill -f "eval_soft.py" 2>/dev/null; pkill -f "run_train.py" 2>/dev/null; sleep 3
  local waited=0 free
  while [ $waited -lt 30 ]; do
    if [ "$(pgrep -f 'eval_soft.py|run_train.py' | wc -l)" -eq 0 ]; then
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "${free:-0}" -ge 20480 ] && { say "GPU 就绪（空闲 ${free} MiB）"; return 0; }
    fi
    sleep 60; waited=$((waited + 1))
  done
  say "GPU 守护 30 分钟未就绪（free=${free:-?}），仍继续尝试"
  return 0
}

scan_args() {  # $1 adapter $2 mixer $3 out $4 points
  local ad=$1 mx=$2 out=$3 pts=$4 marg=""
  [ -n "$mx" ] && marg="--mixer $mx"
  gpu_guard
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$ad" $marg --soft-mode axis --mode scan \
    --force-field mass --scan-values "$pts" \
    --ids-file "$OUTD/ids2_scan20.json" --out "$out" > "$LOGD/$(basename "$out" .jsonl).log" 2>&1
}

hard_eval() {  # $1 adapter $2 out $3 tag
  [ "$(rows_of "$2")" -ge 2000 ] && { say "评测[$3] 已有 $(rows_of "$2") 行，跳过"; return 0; }
  gpu_guard
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$1" --mode hard \
    --data "$PROJ/data/splits/test_local.json" --out "$2" > "$LOGD/eval_$3.log" 2>&1
}

soft_eval() {  # $1 adapter $2 mixer $3 out $4 tag
  [ "$(rows_of "$3")" -ge 2000 ] && { say "评测[$4] 已有 $(rows_of "$3") 行，跳过"; return 0; }
  gpu_guard
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$1" --mixer "$2" --soft-mode axis --mode soft \
    --data "$PROJ/data/splits/test_local.json" --out "$3" > "$LOGD/eval_$4.log" 2>&1
}

train_c3full() {
  [ -f "$C3F/adapter_model.safetensors" ] && { say "[C3full_s] 已有 adapter，跳过"; return 0; }
  gpu_guard
  say "[C3full_s] 训练开始（与 B0 同起点同超参 + axis；439 步）"
  "$PY" "$CV1/train/run_train.py" --epochs 1.0 --seed 42 --save-steps 50 \
    --soft-mode axis --alpha 0.05 --lam 0.05 --rho-max 1 --rho-warmup-frac 0 --no-reuse-embeds \
    --max-steps 439 --resume auto --output-dir "$C3F" > "$LOGD/train_C3full_s.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && die "[C3full_s] 训练 rc=$rc（见 logs/train_C3full_s.log）"
  say "[C3full_s] 训练完成"
}

# ---------------- 0) 记录 B0 的原始命令 ----------------
say "B0 原始训练命令（复现依据，来自 run_pilot.sh train_stage2 + 日志核对）：
  python chain_v1/train/run_train.py --no-soft --epochs 1.0 --seed 42 --save-steps 50 --output-dir \$MDIR/chain_v1_pilot_B0 --resume auto
  （无 --init-adapter → 从 Qwen3.5-4B 基础模型 + 新 LoRA 起步；data=data/llamafactory/train_llamafactory_os_sub_local.json (7019)；
    lr=5e-5 cosine warmup=0.05；bs=1 grad_accum=16；bf16；grad_ckpt；seed/data_seed=42；max_steps=439（由 epochs=1.0 推导，training_args.bin 记录 max_steps=-1））"

# ---------------- 1) 门控结果 ----------------
if [ ! -f "$OUTD/c3a_soft400.jsonl" ] || [ "$(rows_of "$OUTD/c3a_soft400.jsonl")" -ne 400 ]; then
  die "c3a_soft400.jsonl 行数异常（$(rows_of "$OUTD/c3a_soft400.jsonl") ≠ 400）"
fi
"$PY" "$CV1/scripts/c3a_full_gate.py" > "$LOGD/c3a_full_gate.log" 2>&1
g_rc=$?
say "C3a 400 例门控 rc=$g_rc（0=PASS）"

# ---------------- 2) 端点核对结果 ----------------
if ! grep -q ALL_DONE /tmp/c3a_endpoint_run.log 2>/dev/null; then
  say "端点核对尚未完成，等待..."
  while ! grep -q ALL_DONE /tmp/c3a_endpoint_run.log 2>/dev/null; do sleep 20; done
fi
worst=$("$PY" - <<'EOF'
import re
p = "/home/z/zhao/Mammography/Mammo/2-view/chain_v1/logs/c3a_endpoint_compare.log"
try:
    nums = [float(x) for x in re.findall(r"\|Δ\|=([0-9.]+)", open(p).read())]
    print(f"{max(nums):.4f}" if nums else "nan")
except Exception:
    print("nan")
EOF
)
say "端点核对：max|Δ|=$worst（阈值 0.01）"
ep_bad=0
if [ "$worst" = "nan" ] || "$PY" -c "import sys;sys.exit(0 if float('$worst')>0.01 else 1)"; then ep_bad=1; fi

# ---------------- 3) 分支 ----------------
if [ $g_rc -ne 0 ] || [ $ep_bad -eq 1 ]; then
  say "==== 门控或端点未过：只评 B0 测试集，然后停下 ===="
  hard_eval "$MDIR/chain_v1_pilot_B0" "$OUTD/b0_test_hard.jsonl" b0-test || say "B0 测试集评测失败（继续收尾）"
  "$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report_gatefail.log" 2>&1 || say "报告生成失败"
  say "已停：见 outputs/c3_report.md（仅 B0）"
  exit 0
fi

say "==== 分支：C3full_s（axis，同 B0 起点/超参）vs B0 ===="
train_c3full

soft_eval "$C3F" "$C3F/soft_mixer.pt" "$OUTD/c3full_test_soft.jsonl" c3full-test-soft || die "C3full soft 评测失败"
hard_eval "$MDIR/chain_v1_pilot_B0" "$OUTD/b0_test_hard.jsonl" b0-test || die "B0 测试集评测失败"
"$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report_v1.log" 2>&1 || die "第一版报告生成失败"
say "第一版报告已出（C3full soft vs B0）：outputs/c3_report.md"

hard_eval "$C3F" "$OUTD/c3full_test_hard.jsonl" c3full-test-hard || die "C3full hard 评测失败"
"$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report_v1b.log" 2>&1 || die "报告更新（hard）失败"
say "报告已更新：新增 C3full(hard) 列与 C3full(hard)−B0 bootstrap"

if [ "$(rows_of "$OUTD/c3full_scan.jsonl")" -ne 360 ]; then
  scan_args "$C3F" "$C3F/soft_mixer.pt" "$OUTD/c3full_scan.jsonl" "0,0.1,0.2,0.3,0.4,0.45,0.5,0.7,1" || die "C3full 扫描失败"
fi
"$PY" "$CV1/scripts/scan_w.py" --mixer "$C3F/soft_mixer.pt" --scan-jsonl "$OUTD/c3full_scan.jsonl" \
  --points 0,0.1,0.2,0.3,0.4,0.45,0.5,0.7,1 --out /tmp/c3full_rs.json \
  > "$LOGD/c3full_scanw.log" 2>&1 || say "C3full w(p) 导出失败（继续）"
"$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report.log" 2>&1 || say "最终报告刷新失败"
say "全部完成：outputs/c3_report.md（含 9 点扫描与 w(p)）"
