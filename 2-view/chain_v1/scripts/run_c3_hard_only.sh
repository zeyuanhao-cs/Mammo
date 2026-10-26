#!/usr/bin/env bash
# 等 B0 测试集评测完 → 训练 C3full_s（axis，与 B0 同起点/超参）→ C3full hard 评测 → 报告（不含软推理/扫描）
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
MDIR=/home/z/zhao/Mammography/model
STATUS=$CV1/outputs/pilot_status.md
C3F=$MDIR/chain_v1_C3full_s
say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

# ---- 1) 等 B0 评测跑完（2000 行且无评测进程）----
while :; do
  n=$(wc -l < "$OUTD/b0_test_hard.jsonl" 2>/dev/null || echo 0)
  np=$(pgrep -f "eval_soft.py" | wc -l)
  [ "$n" -ge 2000 ] && [ "$np" -eq 0 ] && break
  sleep 30
done
say "B0 测试集评测完成（$(wc -l < "$OUTD/b0_test_hard.jsonl") 行），开始 C3full_s 流程"

# ---- 2) GPU 守护 ----
pkill -f "eval_soft.py" 2>/dev/null; pkill -f "run_train.py" 2>/dev/null; sleep 3
for i in $(seq 1 30); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$(pgrep -f 'eval_soft.py|run_train.py' | wc -l)" -eq 0 ] && [ "${free:-0}" -ge 20480 ]; then
    say "GPU 就绪（空闲 ${free} MiB）"; break
  fi
  sleep 60
done

# ---- 3) 训练 C3full_s（同 B0 起点/超参 + axis）----
if [ -f "$C3F/adapter_model.safetensors" ]; then
  say "[C3full_s] 已有 adapter，跳过训练"
else
  say "[C3full_s] 训练开始（基础模型 + 新 LoRA，1 epoch=439 步，axis α=λ=0.05 ρ=1，seed 42）"
  "$PY" "$CV1/train/run_train.py" --epochs 1.0 --seed 42 --save-steps 50 \
    --soft-mode axis --alpha 0.05 --lam 0.05 --rho-max 1 --rho-warmup-frac 0 --no-reuse-embeds \
    --max-steps 439 --resume auto --output-dir "$C3F" > "$LOGD/train_C3full_s.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$C3F/adapter_model.safetensors" ]; then
    say "STOP: [C3full_s] 训练失败 rc=$rc（见 logs/train_C3full_s.log）"
    tail -5 "$LOGD/train_C3full_s.log" | cut -c1-300 >> "$STATUS"
    exit 1
  fi
  say "[C3full_s] 训练完成"
fi

# ---- 4) C3full hard 测试集评测（不经过软注入）----
if [ "$(wc -l < "$OUTD/c3full_test_hard.jsonl" 2>/dev/null || echo 0)" -lt 2000 ]; then
  say "[C3full hard] 测试集 2000 评测开始"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$C3F" --mode hard \
    --data "$PROJ/data/splits/test_local.json" --out "$OUTD/c3full_test_hard.jsonl" \
    > "$LOGD/eval_c3full-test-hard.log" 2>&1 || { say "STOP: C3full hard 评测失败"; exit 1; }
  say "[C3full hard] 完成：$(wc -l < "$OUTD/c3full_test_hard.jsonl") 行"
fi

# ---- 5) 第一版报告（C3full(hard) vs B0）----
say "生成报告（C3full(hard) − B0 bootstrap + 硬矛盾/分段/漏判子集/混淆矩阵）"
"$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report_hard.log" 2>&1 \
  || { say "STOP: 报告生成失败"; exit 1; }
say "完成：outputs/c3_report.md（C3full hard vs B0；软推理与 9 点扫描按用户要求暂不做）"
