#!/usr/bin/env bash
# C3 快速验证：C3q(50 步) 评测 400 例 + 7 点扫描 → 门控 → 通过则自动跑全量 C3
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
MDIR=/home/z/zhao/Mammography/model
STATUS=$CV1/outputs/pilot_status.md
C3Q=$MDIR/chain_v1_C3q
C3=$MDIR/chain_v1_C3
say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

# 1) 训练（若已有产物则跳过；训练通常由外部启动）
if [ ! -f "$C3Q/soft_mixer.pt" ]; then
  say "C3q: 训练 50 步（init=B0, logit_mlp, uncert, alpha=lam=0.05）"
  "$PY" "$CV1/train/run_train.py" --init-adapter "$MDIR/chain_v1_pilot_B0" \
    --max-steps 50 --seed 42 --save-steps 50 --resume off \
    --soft-mode logit_mlp --mixer-init small --soft-gate uncert \
    --alpha 0.05 --lam 0.05 --rho-max 0.5 --rho-warmup-frac 0 --no-reuse-embeds \
    --output-dir "$C3Q" > "$LOGD/pilot_c3q_train.log" 2>&1 \
    || { say "C3q 训练 FAIL"; exit 1; }
  say "C3q: 训练完成"
fi

# 2) 400 例软模式评测
say "C3q: 400 例软模式评测"
"$PY" "$CV1/infer/eval_soft.py" --adapter "$C3Q" --mixer "$C3Q/soft_mixer.pt" \
  --soft-mode logit_mlp --mode soft --ids-file "$OUTD/ids2_ablation.json" \
  --out "$OUTD/c3q_soft400.jsonl" > "$LOGD/pilot_c3q_soft400.log" 2>&1 \
  || { say "C3q 400 例评测 FAIL（停下）"; exit 1; }

# 3) 7 点扫描（40 例）
say "C3q: 7 点扫描（40 例）"
"$PY" "$CV1/infer/eval_soft.py" --adapter "$C3Q" --mixer "$C3Q/soft_mixer.pt" \
  --soft-mode logit_mlp --mode scan --force-field mass \
  --scan-values 0,0.1,0.3,0.5,0.7,0.9,1.0 \
  --ids-file "$OUTD/ids2_scan20.json" --out "$OUTD/c3q_scan.jsonl" \
  > "$LOGD/pilot_c3q_scan.log" 2>&1 || { say "C3q 扫描 FAIL（停下）"; exit 1; }

# 4) 门控
say "C3q: 门控判定"
"$PY" "$CV1/scripts/c3_gate.py" > "$LOGD/pilot_c3_gate.log" 2>&1
rc=$?
tail -5 "$OUTD/c3q_summary.md" >> "$STATUS" 2>/dev/null
if [ $rc -ne 0 ]; then
  say "C3q: GATE=STOP，停下等用户（见 outputs/pilot/c3q_summary.md）"
  exit 0
fi
say "C3q: GATE=PASS → 自动开始全量 C3"

# 5) 全量 C3（从头 1 epoch，warmup 0.2）
tmux new-session -d -s c3full "bash $CV1/scripts/run_c3_full.sh >> $LOGD/c3_full_master.out 2>&1"
say "C3: 全量训练已在 tmux c3full 启动"
