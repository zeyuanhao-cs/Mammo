#!/usr/bin/env bash
# C3m+T（mutual + 保留 GT 老师项）：等 C3m 全流程结束 → 冒烟 2 步（核对 kl_T 与 kl_mutual 都在）
#   → 训练 439 步（save 50，支持断点续训）→ 测试集 hard 评测 2000 → 报告
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
MDIR=/home/z/zhao/Mammography/model
STATUS=$CV1/outputs/pilot_status.md
M=$MDIR/chain_v1_C3mT
SMOKE=$MDIR/_c3mt_smoke
AX="--soft-mode linear --alpha 0.05 --lam 0.05 --rho-max 1 --rho-warmup-frac 0 --no-reuse-embeds --mutual --keep-gt-teacher"
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

# ---- 0) 等 C3m 全流程结束 ----
while :; do
  n=$(wc -l < "$OUTD/c3m_test_hard.jsonl" 2>/dev/null || echo 0)
  busy=$(pgrep -f "run_c3m.sh|run_train.py|eval_soft.py" | wc -l)
  [ "$n" -ge 2000 ] && [ "$busy" -eq 0 ] && break
  sleep 30
done
say "C3m 完成（$(wc -l < "$OUTD/c3m_test_hard.jsonl") 行），开始 C3m+T"

# ---- 1) 冒烟 2 步（核对 kl_T 与 kl_mutual 同时存在）----
rm -rf "$SMOKE"; gpu_guard
say "[C3m+T] 冒烟 2 步"
"$PY" "$CV1/train/run_train.py" $AX --max-steps 2 --limit 32 --save-steps 2 --seed 42 --resume off \
  --output-dir "$SMOKE" > "$LOGD/train_c3mt_smoke.log" 2>&1
rc=$?
if [ $rc -ne 0 ] \
   || ! grep -q "unknown 已进优化器: True" "$LOGD/train_c3mt_smoke.log"; then
  say "STOP: [C3m+T] 冒烟失败 rc=$rc"; tail -6 "$LOGD/train_c3mt_smoke.log" | cut -c1-300 >> "$STATUS"; exit 1
fi
"$PY" - "$LOGD/train_c3mt_smoke.log" <<'EOF' >> "$STATUS" 2>&1
import ast, re, sys
p = sys.argv[1]
recs = []
for chunk in open(p, encoding="utf-8", errors="ignore").read().replace("\r", "\n").split("\n"):
    if "'kl'" not in chunk:
        continue
    m = re.search(r"\{'rho'.*\}", chunk)
    if m:
        try:
            recs.append(ast.literal_eval(m.group(0)))
        except Exception:
            pass
if not recs:
    print("[C3m+T] 冒烟日志里没有指标记录"); sys.exit(1)
r = recs[-1]
need = ["loss", "ce1", "ce2", "ce_m", "kl", "kl_mutual", "n_am", "n_ma", "n_none"]
missing = [k for k in need if k not in r]
if missing:
    print(f"[C3m+T] 冒烟日志缺字段: {missing}"); sys.exit(1)
f = {k: float(r[k]) for k in ("loss", "ce1", "ce2", "ce_m", "kl", "kl_mutual")}
exp = f["ce1"] + 0.05 * (f["ce2"] + f["ce_m"]) + 0.05 * (f["kl"] + f["kl_mutual"])
ok = abs(exp - f["loss"]) < 2e-3
print(f"[C3m+T] 冒烟指标: loss={f['loss']:.4f} ce1={f['ce1']:.4f} ce2={f['ce2']:.4f} "
      f"ce_m={f['ce_m']:.4f} kl_T={f['kl']:.4f} kl_mutual={f['kl_mutual']:.4f} "
      f"n_am/n_ma/n_none={r['n_am']}/{r['n_ma']}/{r['n_none']} | loss 复算={exp:.4f} {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
EOF
smoke_rc=$?
[ $smoke_rc -ne 0 ] && die "[C3m+T] 冒烟校验失败（见 pilot_status.md 上一条）"
say "[C3m+T] 冒烟通过：kl_T 与 kl_mutual 同时存在于损失"

# ---- 2) 训练 439 步（save 50，resume auto）----
if [ -f "$M/adapter_model.safetensors" ]; then
  say "[C3m+T] 已有 adapter，跳过训练"
else
  gpu_guard
  say "[C3m+T] 训练开始（439 步，mutual+keep-gt-teacher，linear，save 50）"
  "$PY" "$CV1/train/run_train.py" $AX --epochs 1.0 --seed 42 --max-steps 439 --save-steps 50 \
    --resume auto --output-dir "$M" > "$LOGD/train_C3mT.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$M/adapter_model.safetensors" ]; then
    say "STOP: [C3m+T] 训练失败 rc=$rc"; tail -6 "$LOGD/train_C3mT.log" | cut -c1-300 >> "$STATUS"; exit 1
  fi
  say "[C3m+T] 训练完成"
fi

# ---- 3) 测试集 hard 评测 ----
if [ "$(wc -l < "$OUTD/c3mt_test_hard.jsonl" 2>/dev/null || echo 0)" -lt 2000 ]; then
  gpu_guard
  say "[C3m+T] 测试集 hard 评测 2000"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$M" --mode hard \
    --data "$PROJ/data/splits/test_local.json" --out "$OUTD/c3mt_test_hard.jsonl" \
    > "$LOGD/eval_c3mt-test-hard.log" 2>&1 || { say "STOP: C3m+T hard 评测失败"; exit 1; }
fi

# ---- 3b) 测试集 soft 评测（linear，batch=1）----
if [ "$(wc -l < "$OUTD/c3mt_test_soft.jsonl" 2>/dev/null || echo 0)" -lt 2000 ]; then
  gpu_guard
  say "[C3m+T] 测试集 soft 评测 2000（linear，batch=1）"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$M" --soft-mode linear --mode soft \
    --data "$PROJ/data/splits/test_local.json" --out "$OUTD/c3mt_test_soft.jsonl" \
    > "$LOGD/eval_c3mt-test-soft.log" 2>&1 || { say "STOP: C3m+T soft 评测失败"; exit 1; }
fi

# ---- 4) 报告 ----
say "[C3m+T] 生成报告（并列 B0 / C3full / C3m / C3m+T）"
"$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report_c3mt.log" 2>&1 \
  || { say "STOP: 报告生成失败"; exit 1; }
say "完成：outputs/c3_report.md（含 C3m+T−C3m、C3m+T−C3full(hard) bootstrap）"
