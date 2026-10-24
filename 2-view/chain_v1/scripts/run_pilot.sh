#!/usr/bin/env bash
# chain_v1 阶段一(最快版) + 阶段二(全量版, 时间压缩版) 全自动总控。tmux 内运行。
# 阶段二顺序: B0 → C1 → B0(eval_soft 硬)/C1(linear soft) 主要指标 → C1 小扫描(GT各20, 7点)
#            → 第一版报告 → C2 → C2 指标 → 最终报告（含 C2 与 C1−C2 bootstrap 差值）
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
MDIR=/home/z/zhao/Mammography/model
START_ADAPTER=$MDIR/Qwen3.5-4B-lora-2view-os-sub-vt/checkpoint-439
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
STATUS=$CV1/outputs/pilot_status.md
TOTAL_EPOCH_STEPS=439
mkdir -p "$OUTD" "$LOGD"
export PYTHONUNBUFFERED=1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$STATUS"; }
done_rows() { [ -s "$1" ] && [ "$(wc -l < "$1")" -eq "$2" ]; }  # 产物跳过判断

# 等待 GPU 释放：每分钟查一次空闲显存 >= 20 GB，最多 30 分钟
wait_gpu() {
  local need=20480 waited=0 free
  while [ $waited -lt 30 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$need" ]; then
      say "GPU 就绪：空闲 ${free} MiB"; return 0
    fi
    sleep 60; waited=$((waited + 1))
  done
  say "等待 GPU 30 分钟仍不足（空闲 ${free:-?} MiB），继续尝试"
  return 0
}

# 心跳：每 10 分钟往 pilot_status.md 写一行（阶段 / 最新步数 / GPU 显存）
(
  while true; do
    sleep 600
    st=$(tail -1 "$STATUS" 2>/dev/null | cut -c1-100)
    lg=$(ls -t "$LOGD"/pilot_*_train.log 2>/dev/null | head -1)
    prog=$(tr '\r' '\n' < "$lg" 2>/dev/null | grep -oE "[0-9]+/[0-9]+ \[" | tail -1)
    gpu=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | head -1)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HEARTBEAT 最近事件=[${st}] 进度=[${prog}] GPU(used/free)=[${gpu}]" >> "$STATUS"
  done
) &

if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  echo "FATAL: PYTORCH_CUDA_ALLOC_CONF 必须为空"; exit 1
fi
say "==== pilot(时间压缩版) 开始（PID $$）===="

# ---------- P0.5 --init-adapter 2 步测通 ----------
if [ ! -f "$OUTD/.init_ok" ]; then
  say "P0.5: --init-adapter 2 步测通开始"
  "$PY" "$CV1/train/run_train.py" \
    --init-adapter "$START_ADAPTER" --max-steps 2 --limit 32 --save-steps 2 --resume off \
    --soft-mode logit_mlp --mixer-init small --alpha 1 --lam 1 --rho-max 0.5 --rho-warmup-frac 0 \
    --no-reuse-embeds --output-dir "$MDIR/_pilot_init_test" \
    > "$LOGD/pilot_init_test.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$MDIR/_pilot_init_test/adapter_model.safetensors" ]; then
    say "P0.5: FAIL rc=$rc，停下"; exit 1
  fi
  grep -E "\[init-adapter\]|\[mixer-scale\]" "$LOGD/pilot_init_test.log" | while read -r l; do say "P0.5: $l"; done
  touch "$OUTD/.init_ok"
  say "P0.5: PASS"
fi

train_and_watch() {  # $1 log  $2 总步数  $3 组名
  local logf=$1 total=$2 gname=$3
  local last_mark=0
  while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
    if grep -qiE "loss.*(nan|inf)" "$logf" 2>/dev/null; then
      say "[$gname] FAIL: NaN/Inf，停下等我"; exit 1
    fi
    local cur
    cur=$(grep -oE "[0-9]+/$total" "$logf" 2>/dev/null | tail -1 | cut -d/ -f1)
    [ -z "$cur" ] && continue
    if [ $((cur / 50)) -gt $last_mark ] 2>/dev/null; then
      last_mark=$((cur / 50))
      local loss_line
      loss_line=$(grep -oE "\{'loss'[^}]*\}" "$logf" | tail -1)
      say "[$gname] 进度 $cur/$total  $loss_line"
    fi
  done
  wait $TRAIN_PID
  return $?
}

check_products() {  # $1 outdir  $2 期望步数  $3 需要mixer(yes/no)
  local out=$1 want=$2 needmix=$3
  [ -f "$out/adapter_model.safetensors" ] || { say "产物检查 FAIL: $out 缺 adapter"; return 1; }
  if [ "$needmix" = "yes" ]; then
    [ -f "$out/soft_mixer.pt" ] || { say "产物检查 FAIL: $out 缺 soft_mixer.pt"; return 1; }
  fi
  local gs
  gs=$("$PY" -c "import json;print(json.load(open('$out/trainer_state.json'))['global_step'])" 2>/dev/null)
  [ "$gs" = "$want" ] || { say "产物检查 FAIL: $out global_step=$gs != $want"; return 1; }
  say "产物检查 PASS: $out (global_step=$gs)"
  return 0
}

# ---------- 阶段一 ----------
if [ ! -f "$OUTD/.stage1_done" ]; then
  if check_products "$MDIR/chain_v1_QC1" 50 yes; then
    say "QC1 产物已齐（global_step=50），跳过训练"
  else
    say "==== 阶段一: QC1 训练开始（50 步, seed 42）===="
    "$PY" "$CV1/train/run_train.py" \
      --init-adapter "$START_ADAPTER" --max-steps 50 --seed 42 --save-steps 50 --resume off \
      --soft-mode logit_mlp --mixer-init small --alpha 1 --lam 1 --rho-max 0.5 --rho-warmup-frac 0 \
      --no-reuse-embeds --output-dir "$MDIR/chain_v1_QC1" \
      > "$LOGD/pilot_qc1_train.log" 2>&1 &
    TRAIN_PID=$!
    train_and_watch "$LOGD/pilot_qc1_train.log" 50 QC1
    rc=$?
    [ $rc -ne 0 ] && { say "阶段一: QC1 训练 rc=$rc，停下"; exit 1; }
    check_products "$MDIR/chain_v1_QC1" 50 yes || { say "阶段一: 停下"; exit 1; }
    grep -E "\[mixer-scale\]" "$LOGD/pilot_qc1_train.log" | while read -r l; do say "QC1: $l"; done
  fi

  if done_rows "$OUTD/start_hard.jsonl" 1600; then
    say "评测 1/5 已有 1600 行，跳过"
  else
    say "阶段一评测 1/5: 起点 hard 全验证集 1600（b8 管线，只用来挑子集）"
    "$PY" "$PROJ/scripts/vlm/infer_2view_batched.py" --adapter "$START_ADAPTER" \
      --data "$PROJ/data/splits/val_raw_local.json" --batch-size 8 \
      --out-suffix _pilot_start --out-probs "$OUTD/start_hard.jsonl" --tag pilot-start \
      > "$LOGD/pilot_start_hard.log" 2>&1 || { say "阶段一评测 1/5 FAIL"; exit 1; }
    say "阶段一评测 1/5 done: $(wc -l < "$OUTD/start_hard.jsonl") 行"
  fi

  say "阶段一: 挑批次"
  "$PY" "$CV1/scripts/make_ids.py" --hard-jsonl "$OUTD/start_hard.jsonl" \
    --out-prefix "$OUTD/ids" --n-soft 200 --n-scan-pos 50 --n-scan-neg 50

  if done_rows "$OUTD/start_hard200.jsonl" 200; then
    say "评测 1.5/5 已有 200 行，跳过"
  else
    say "阶段一评测 1.5/5: 起点 hard 子集（eval_soft，与 QC1 同管线同口径，机制指标用）"
    "$PY" "$CV1/infer/eval_soft.py" --adapter "$START_ADAPTER" --mode hard \
      --ids-file "$OUTD/ids_soft.json" --out "$OUTD/start_hard200.jsonl" \
      > "$LOGD/pilot_start_hard200.log" 2>&1 || { say "阶段一评测 1.5/5 FAIL"; exit 1; }
    say "阶段一评测 1.5/5 done: $(wc -l < "$OUTD/start_hard200.jsonl") 行"
  fi

  if done_rows "$OUTD/qc1_soft.jsonl" 200; then
    say "评测 2/5 已有 200 行，跳过"
  else
    say "阶段一评测 2/5: QC1 soft（≤200）"
    "$PY" "$CV1/infer/eval_soft.py" --adapter "$MDIR/chain_v1_QC1" \
      --mixer "$MDIR/chain_v1_QC1/soft_mixer.pt" --mode soft \
      --ids-file "$OUTD/ids_soft.json" --out "$OUTD/qc1_soft.jsonl" \
      > "$LOGD/pilot_qc1_soft.log" 2>&1 || { say "阶段一评测 2/5 FAIL"; exit 1; }
  fi

  if done_rows "$OUTD/qc1_swap.jsonl" 200; then
    say "评测 3/5 已有 200 行，跳过"
  else
    say "阶段一评测 3/5: QC1 swap_hard（软换硬；batch=1 与 soft 同口径）"
    "$PY" "$CV1/infer/eval_soft.py" --adapter "$MDIR/chain_v1_QC1" \
      --soft-mode logit_mlp --mode swap_hard \
      --ids-file "$OUTD/ids_soft.json" --out "$OUTD/qc1_swap.jsonl" \
      > "$LOGD/pilot_qc1_swap.log" 2>&1 || { say "阶段一评测 3/5 FAIL"; exit 1; }
  fi

  if done_rows "$OUTD/qc1_scan.jsonl" 1100; then
    say "评测 4/5 已有 1100 行，跳过"
  else
    say "阶段一评测 4/5: QC1 scan（50+50）"
    "$PY" "$CV1/infer/eval_soft.py" --adapter "$MDIR/chain_v1_QC1" \
      --mixer "$MDIR/chain_v1_QC1/soft_mixer.pt" --mode scan \
      --ids-file "$OUTD/ids_scan.json" --out "$OUTD/qc1_scan.jsonl" \
      > "$LOGD/pilot_qc1_scan.log" 2>&1 || { say "阶段一评测 4/5 FAIL"; exit 1; }
  fi

  # （起点 scan 对照原为 5/5，已挪到阶段二末 2.6b，见下）

  # ---- 阶段一附加: 确定样本上软注入的扰动（GT BI-RADS=1 且起点四征象 p<0.05）----
  say "阶段一评测 6/6: 确定样本软注入扰动（100 例，QC1 软 vs 软换硬，batch=1）"
  "$PY" "$CV1/scripts/make_cert_ids.py" --hard-jsonl "$OUTD/start_hard.jsonl" \
    --out "$OUTD/ids_cert.json" --n 100 || { say "确定样本挑选 FAIL"; exit 1; }
  cert_n=$("$PY" -c "import json;print(len(json.load(open('$OUTD/ids_cert.json'))))")
  if done_rows "$OUTD/qc1_cert_soft.jsonl" "$cert_n"; then
    say "确定样本 soft 已有 $cert_n 行，跳过"
  else
    "$PY" "$CV1/infer/eval_soft.py" --adapter "$MDIR/chain_v1_QC1" \
      --mixer "$MDIR/chain_v1_QC1/soft_mixer.pt" --mode soft \
      --ids-file "$OUTD/ids_cert.json" --out "$OUTD/qc1_cert_soft.jsonl" \
      > "$LOGD/pilot_cert_soft.log" 2>&1 || { say "确定样本 soft FAIL"; exit 1; }
  fi
  if done_rows "$OUTD/qc1_cert_swap.jsonl" "$cert_n"; then
    say "确定样本 swap 已有 $cert_n 行，跳过"
  else
    "$PY" "$CV1/infer/eval_soft.py" --adapter "$MDIR/chain_v1_QC1" \
      --soft-mode logit_mlp --mode swap_hard \
      --ids-file "$OUTD/ids_cert.json" --out "$OUTD/qc1_cert_swap.jsonl" \
      > "$LOGD/pilot_cert_swap.log" 2>&1 || { say "确定样本 swap FAIL"; exit 1; }
  fi

  say "阶段一: 分析 + quick_report + GATE"
  set +e
  "$PY" "$CV1/scripts/quick_analyze.py" > "$LOGD/pilot_quick_analyze.log" 2>&1
  gate_rc=$?
  set -e
  "$PY" "$CV1/scripts/cert_perturb.py" > "$LOGD/pilot_cert_report.log" 2>&1 \
    || say "cert_perturb 失败（不阻断 GATE）"
  tail -3 "$CV1/outputs/pilot/quick_report.md" >> "$STATUS" 2>/dev/null
  if [ $gate_rc -ne 0 ]; then
    say "阶段一: GATE=STOP（rc=$gate_rc），停下等用户"
    exit 0
  fi
  say "阶段一: GATE=PASS，进入阶段二"
  touch "$OUTD/.stage1_done"
fi

# ---------- 阶段二 ----------
# C1/C2 用 linear 注入（不用 mixer）：QC1 的 logit_mlp 扫描方向反向，归因于 mixer（见阶段一结论）
SOFT_ARGS='--soft-mode linear --alpha 1 --lam 1 --rho-max 0.5 --rho-warmup-frac 0.2 --no-reuse-embeds'
B0=$MDIR/chain_v1_pilot_B0
C1=$MDIR/chain_v1_pilot_C1
C2=$MDIR/chain_v1_pilot_C2

train_stage2() {  # $1 组名  $2 额外参数  $3 需要mixer(yes/no)
  local g=$1 extra=$2 needmix=$3
  local out="$MDIR/chain_v1_pilot_$g"
  if check_products "$out" $TOTAL_EPOCH_STEPS "$needmix" 2>/dev/null; then
    say "[$g] 产物已齐，跳过训练"; return 0
  fi
  local attempt=1
  while [ $attempt -le 3 ]; do
    say "[$g] 训练尝试 $attempt/3（1 epoch=$TOTAL_EPOCH_STEPS 步, seed 42, resume=auto）"
    "$PY" "$CV1/train/run_train.py" $extra --epochs 1.0 --seed 42 --save-steps 50 \
      --output-dir "$out" --resume auto > "$LOGD/pilot_${g}_train.log" 2>&1 &
    TRAIN_PID=$!
    train_and_watch "$LOGD/pilot_${g}_train.log" $TOTAL_EPOCH_STEPS "$g"
    rc=$?
    if [ $rc -eq 0 ] && check_products "$out" $TOTAL_EPOCH_STEPS "$needmix"; then
      say "[$g] 训练成功（尝试 $attempt）"; return 0
    fi
    say "[$g] 尝试 $attempt 失败 rc=$rc"
    { echo "[$(date '+%F %T')] [$g] 失败尾部（rc=$rc）:"
      tail -5 "$LOGD/pilot_${g}_train.log" 2>/dev/null | cut -c1-300
      echo "  nvidia-smi: $(nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader | head -1)"
    } >> "$STATUS"
    if [ $attempt -lt 3 ]; then
      say "[$g] 等待 5 分钟后重试"
      sleep 300
      wait_gpu
    fi
    attempt=$((attempt + 1))
  done
  say "[$g] 3 次尝试均失败 → 跳过该组，继续后续步骤"; echo "- [$g] 训练 3 次失败，已跳过" >> "$OUTD/eval_skips.txt"
  return 1
}

run_eval() {  # $1 tag  $2 adapter  $3 mixer  $4 mode  $5 idsfile  $6 out  $7 softmode  $8 batch  $9 额外参数
  local tag=$1 adapter=$2 mixer=$3 mode=$4 idsf=$5 out=$6 softmode=${7:-logit_mlp} batch=${8:-1} extra9=${9:-}
  local expect=1600
  [ -n "$idsf" ] && expect=$("$PY" -c "import json;print(len(json.load(open('$idsf'))))")
  local have=0
  [ -f "$out" ] && have=$(wc -l < "$out")
  if [ "$have" -ge "$expect" ]; then say "评测[$tag] 已有 $have 行，跳过"; return 0; fi
  say "评测[$tag] 开始（$mode, 期望 $expect, batch=$batch）"
  local mixarg="" idarg="" batcharg=""
  [ -n "$mixer" ] && mixarg="--mixer $mixer"
  [ -n "$idsf" ] && idarg="--ids-file $idsf"
  [ "$batch" -gt 1 ] && batcharg="--batch-size $batch"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$adapter" $mixarg --mode "$mode" \
    --soft-mode "$softmode" $idarg $batcharg $extra9 --out "$out" \
    > "$LOGD/pilot_eval_${tag}.log" 2>&1 \
    || { say "评测[$tag] FAIL（跳过，继续后续）"; echo "- 评测[$tag] 失败（见 logs/pilot_eval_${tag}.log）" >> "$OUTD/eval_skips.txt"; return 0; }
  say "评测[$tag] done: $(wc -l < "$out") 行"
}

# ---- 2.1 训练 B0 → C1 ----
if [ ! -f "$OUTD/.stage2_bc_train_done" ]; then
  say "==== 阶段二: B0 / C1 训练（串行）===="
  train_stage2 B0 "--no-soft --resume off" no || say "[B0] 训练失败，跳过该组继续"
  train_stage2 C1 "$SOFT_ARGS --resume off" no || say "[C1] 训练失败，跳过该组继续"
  touch "$OUTD/.stage2_bc_train_done"
  say "阶段二: B0/C1 训练完成"
fi

# ---- 2.2+2.3 B0 全量 eval_soft 硬模式 + C1 soft 全量 + 第一版报告 ----
# （对齐检查结论：已知 b8 批量差异，约 2% 翻转，主比较统一用 eval_soft，见 pilot_status.md。
#   B0 的 b8 全量只在最后作附表参考，见 2.8。）
if [ ! -f "$OUTD/.stage2_bc_eval_done" ]; then
  run_eval b0-hard "$B0" "" hard "" "$OUTD/b0_hard.jsonl"

  if [ -s "$OUTD/b0_hard.jsonl" ]; then
    "$PY" "$CV1/scripts/make_ids.py" --hard-jsonl "$OUTD/b0_hard.jsonl" \
      --out-prefix "$OUTD/ids2" --n-soft 100000 --n-scan-pos 50 --n-scan-neg 50 \
      --n-ablation 400
  else
    say "缺 b0_hard.jsonl，跳过 make_ids（后续评测将按缺项跳过）"
  fi

  t0=$(date +%s)
  run_eval c1-soft "$C1" "" soft "" "$OUTD/c1_soft.jsonl" linear
  say "C1 soft 1600 条用时 $(( ($(date +%s) - t0) / 60 )) 分钟（batch=1）"

  # ---- C1 小规模概率扫描（GT>=3 / GT<=2 各 20，7 个网格点）----
  say "C1 小规模扫描: GT>=3/GT<=2 各 20，p∈{0,0.1,0.3,0.5,0.7,0.9,1.0}"
  "$PY" "$CV1/scripts/make_scan20.py" || { say "scan20 选批次 FAIL（跳过）"; echo "- scan20 选批次失败" >> "$OUTD/eval_skips.txt"; }
  if [ -f "$OUTD/ids2_scan20.json" ]; then
    run_eval c1-scan "$C1" "" scan "$OUTD/ids2_scan20.json" "$OUTD/c1_scan.jsonl" linear 1 \
      "--force-field mass --scan-values 0,0.1,0.3,0.5,0.7,0.9,1.0"
  fi

  say "写第一版 pilot_v1_report.md（B0/C1 主要指标）"
  "$PY" "$CV1/scripts/pilot_metrics.py" > "$LOGD/pilot_metrics_bc.log" 2>&1 \
    || { say "pilot_metrics（B0/C1 版）FAIL（继续）"; echo "- 第一版 pilot_metrics 失败" >> "$OUTD/eval_skips.txt"; }
  say "第一版 pilot_v1_report.md 已生成"
  touch "$OUTD/.stage2_bc_eval_done"
fi

# ---- 2.4 训练 C2 + 评测 ----
if [ ! -f "$OUTD/.stage2_c2_done" ]; then
  say "==== 阶段二: C2 训练 ===="
  train_stage2 C2 "$SOFT_ARGS --detach-p --resume off" no || say "[C2] 训练失败，跳过该组继续"
  run_eval c2-soft "$C2" "" soft "" "$OUTD/c2_soft.jsonl" linear
  touch "$OUTD/.stage2_c2_done"
  say "阶段二: C2 训练+评测完成"
fi

# ---- 2.7 最终报告（含 C2、C1−C2 bootstrap 差值）----
say "==== 汇总指标 + 差值 bootstrap + 最终报告 ===="
"$PY" "$CV1/scripts/pilot_metrics.py" > "$LOGD/pilot_metrics.log" 2>&1 \
  || { say "pilot_metrics FAIL（继续，保留已有产物）"; echo "- 最终 pilot_metrics 失败" >> "$OUTD/eval_skips.txt"; }
say "==== pilot_v1_report.md 完成，A40 停下等用户 ===="
