#!/usr/bin/env bash
# 全量 C3：从头 1 epoch（439 步，warmup 0.2），全量评测 + pilot_v2_report.md
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
MDIR=/home/z/zhao/Mammography/model
STATUS=$CV1/outputs/pilot_status.md
C3=$MDIR/chain_v1_C3
STEPS=439
say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

try_train() {
  local attempt=1
  while [ $attempt -le 3 ]; do
    say "[C3] 训练尝试 $attempt/3（1 epoch=$STEPS 步，warmup 0.2）"
    "$PY" "$CV1/train/run_train.py" --epochs 1.0 --seed 42 --save-steps 50 \
      --soft-mode logit_mlp --mixer-init small --soft-gate uncert \
      --alpha 0.05 --lam 0.05 --rho-max 0.5 --rho-warmup-frac 0.2 --no-reuse-embeds \
      --output-dir "$C3" --resume auto > "$LOGD/pilot_C3_train.log" 2>&1
    rc=$?
    gs=$("$PY" -c "import json;print(json.load(open('$C3/trainer_state.json'))['global_step'])" 2>/dev/null || echo 0)
    if [ $rc -eq 0 ] && [ "$gs" = "$STEPS" ] && [ -f "$C3/adapter_model.safetensors" ] && [ -f "$C3/soft_mixer.pt" ]; then
      say "[C3] 训练成功（尝试 $attempt, global_step=$gs）"; return 0
    fi
    say "[C3] 尝试 $attempt 失败 rc=$rc global_step=$gs"
    { echo "[$(date '+%F %T')] [C3] 失败尾部:"
      tail -5 "$LOGD/pilot_C3_train.log" | cut -c1-300
      echo "  nvidia-smi: $(nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader | head -1)"
    } >> "$STATUS"
    [ $attempt -lt 3 ] && { say "[C3] 等 5 分钟后重试"; sleep 300; }
    attempt=$((attempt + 1))
  done
  say "[C3] 3 次均失败，停下"; exit 1
}

try_train

say "[C3] 全量 1600 软模式评测（batch=1）"
"$PY" "$CV1/infer/eval_soft.py" --adapter "$C3" --mixer "$C3/soft_mixer.pt" \
  --soft-mode logit_mlp --mode soft --out "$OUTD/c3_soft1600.jsonl" \
  > "$LOGD/pilot_c3_soft1600.log" 2>&1 || say "[C3] 全量评测失败（继续出报告）"

say "[C3] 7 点扫描（40 例）"
"$PY" "$CV1/infer/eval_soft.py" --adapter "$C3" --mixer "$C3/soft_mixer.pt" \
  --soft-mode logit_mlp --mode scan --force-field mass \
  --scan-values 0,0.1,0.3,0.5,0.7,0.9,1.0 \
  --ids-file "$OUTD/ids2_scan20.json" --out "$OUTD/c3_scan.jsonl" \
  > "$LOGD/pilot_c3_scan.log" 2>&1 || say "[C3] 扫描失败（继续）"

say "[C3] 生成 pilot_v2_report.md"
"$PY" "$CV1/scripts/pilot_v2_metrics.py" > "$LOGD/pilot_v2_metrics.log" 2>&1 \
  || say "[C3] pilot_v2_metrics 失败"
say "[C3] 完成"
