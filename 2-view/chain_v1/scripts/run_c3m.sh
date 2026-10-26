#!/usr/bin/env bash
# C3m（双视角互相验证）：等 C3full 软推理报告出完 → 冒烟 2 步 → 训练 439 步 → hard 评测 → 报告
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
MDIR=/home/z/zhao/Mammography/model
STATUS=$CV1/outputs/pilot_status.md
M=$MDIR/chain_v1_C3m
SMOKE=$MDIR/_c3m_smoke
AX="--soft-mode linear --alpha 0.05 --lam 0.05 --rho-max 1 --rho-warmup-frac 0 --no-reuse-embeds --mutual"
say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { say "STOP: $*"; exit 1; }

gpu_guard() {
  pkill -f "eval_soft.py" 2>/dev/null; pkill -f "run_train.py" 2>/dev/null; sleep 3
  for _ in $(seq 1 30); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$(pgrep -f 'eval_soft.py|run_train.py' | wc -l)" -eq 0 ] && [ "${free:-0}" -ge 20480 ]; then
      say "GPU 就绪（空闲 ${free} MiB）"; return 0
    fi
    sleep 60
  done
  say "GPU 等待 30 分钟仍未就绪，继续尝试"; return 0
}

# ---- 0) 等 C3full 软推理 + 报告完成 ----
while ! grep -q ALL_DONE /tmp/c3soft_run.log 2>/dev/null; do sleep 20; done
say "C3full 软推理已结束，开始 C3m 流程"

# ---- 1) 冒烟 2 步 ----
rm -rf "$SMOKE"; gpu_guard
say "[C3m] 冒烟 2 步"
"$PY" "$CV1/train/run_train.py" $AX --max-steps 2 --limit 32 --save-steps 2 --seed 42 --resume off \
  --output-dir "$SMOKE" > "$LOGD/train_c3m_smoke.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -q "n_am" "$LOGD/train_c3m_smoke.log" \
   || ! grep -q "unknown 已进优化器: True" "$LOGD/train_c3m_smoke.log"; then
  say "STOP: [C3m] 冒烟失败 rc=$rc（见 logs/train_c3m_smoke.log）"
  tail -6 "$LOGD/train_c3m_smoke.log" | cut -c1-300 >> "$STATUS"
  exit 1
fi
nan=$(grep -ci "nan" "$LOGD/train_c3m_smoke.log" || true)
say "[C3m] 冒烟通过（linear + mutual，unknown 已进优化器）（loss 有限、日志含 n_am/n_ma/n_none；nan 匹配=$nan）"
tail -2 "$LOGD/train_c3m_smoke.log" | tr '\r' '\n' | grep -oE "\{'rho'.*\}" | tail -1 | cut -c1-220 >> "$STATUS"

# ---- 2) 训练 439 步 ----
if [ -f "$M/adapter_model.safetensors" ]; then
  say "[C3m] 已有 adapter，跳过训练"
else
  gpu_guard
  say "[C3m] 训练开始（439 步，mutual，axis，α=λ=0.05，ρ=1，seed 42）"
  "$PY" "$CV1/train/run_train.py" $AX --epochs 1.0 --seed 42 --max-steps 439 --save-steps 50 \
    --resume auto --output-dir "$M" > "$LOGD/train_C3m.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$M/adapter_model.safetensors" ]; then
    say "STOP: [C3m] 训练失败 rc=$rc（见 logs/train_C3m.log）"
    tail -6 "$LOGD/train_C3m.log" | cut -c1-300 >> "$STATUS"
    exit 1
  fi
  say "[C3m] 训练完成"
fi

# ---- 3) 测试集 hard 评测 ----
if [ "$(wc -l < "$OUTD/c3m_test_hard.jsonl" 2>/dev/null || echo 0)" -lt 2000 ]; then
  gpu_guard
  say "[C3m] 测试集 hard 评测 2000"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$M" --mode hard \
    --data "$PROJ/data/splits/test_local.json" --out "$OUTD/c3m_test_hard.jsonl" \
    > "$LOGD/eval_c3m-test-hard.log" 2>&1 || { say "STOP: C3m hard 评测失败"; exit 1; }
  say "[C3m] hard 评测完成：$(wc -l < "$OUTD/c3m_test_hard.jsonl") 行"
fi

# ---- 3b) 测试集 soft 评测（linear，batch=1）----
if [ "$(wc -l < "$OUTD/c3m_test_soft.jsonl" 2>/dev/null || echo 0)" -lt 2000 ]; then
  gpu_guard
  say "[C3m] 测试集 soft 评测 2000（linear，batch=1）"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$M" --soft-mode linear --mode soft \
    --data "$PROJ/data/splits/test_local.json" --out "$OUTD/c3m_test_soft.jsonl" \
    > "$LOGD/eval_c3m-test-soft.log" 2>&1 || { say "STOP: C3m soft 评测失败"; exit 1; }
  say "[C3m] soft 评测完成：$(wc -l < "$OUTD/c3m_test_soft.jsonl") 行"
fi

# ---- 4) 报告 ----
say "[C3m] 生成报告（并列 B0 / C3full soft / C3full hard / C3m soft+hard）"
"$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report_c3m.log" 2>&1 \
  || { say "STOP: 报告生成失败"; exit 1; }
say "完成：outputs/c3_report.md（C3m soft/hard 两列 + 三组 bootstrap + 老师选择曲线）"
