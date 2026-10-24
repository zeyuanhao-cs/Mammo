#!/usr/bin/env bash
# chain_v1 阶段一(最快版) + 阶段二(全量版, 时间压缩版) 全自动总控。tmux 内运行。
# 阶段二顺序: GATE → 训 B0 → 训 C1 → B0(eval_soft 硬)/C1(soft) 主要指标 → 第一版报告
#            → 2.3b 探针重跑 → 训 C2 → 消融(400) → 扫描(50+50) → 遮挡 → 最终报告 → B0 b8 附表
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
SOFT_ARGS='--soft-mode logit_mlp --mixer-init small --alpha 1 --lam 1 --rho-max 0.5 --rho-warmup-frac 0.2 --no-reuse-embeds'
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
    "$PY" "$CV1/train/run_train.py" $extra --epochs 1.0 --seed 42 --save-steps 100 \
      --output-dir "$out" --resume auto > "$LOGD/pilot_${g}_train.log" 2>&1 &
    TRAIN_PID=$!
    train_and_watch "$LOGD/pilot_${g}_train.log" $TOTAL_EPOCH_STEPS "$g"
    rc=$?
    if [ $rc -eq 0 ] && check_products "$out" $TOTAL_EPOCH_STEPS "$needmix"; then
      say "[$g] 训练成功（尝试 $attempt）"; return 0
    fi
    say "[$g] 尝试 $attempt 失败 rc=$rc"
    attempt=$((attempt + 1))
  done
  say "[$g] 3 次尝试均失败，停下等我"; exit 1
}

run_eval() {  # $1 tag  $2 adapter  $3 mixer  $4 mode  $5 idsfile  $6 out  $7 softmode  $8 batch(默认1，全口径统一)
  local tag=$1 adapter=$2 mixer=$3 mode=$4 idsf=$5 out=$6 softmode=${7:-logit_mlp} batch=${8:-1}
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
    --soft-mode "$softmode" $idarg $batcharg --out "$out" \
    > "$LOGD/pilot_eval_${tag}.log" 2>&1 \
    || { say "评测[$tag] FAIL"; exit 1; }
  say "评测[$tag] done: $(wc -l < "$out") 行"
}

# ---- 2.1 训练 B0 → C1 ----
if [ ! -f "$OUTD/.stage2_bc_train_done" ]; then
  say "==== 阶段二: B0 / C1 训练（串行）===="
  train_stage2 B0 "--no-soft --resume off" no || exit 1
  train_stage2 C1 "$SOFT_ARGS --resume off" yes || exit 1
  touch "$OUTD/.stage2_bc_train_done"
  say "阶段二: B0/C1 训练完成"
fi

# ---- 2.2+2.3 B0 全量 eval_soft 硬模式 + C1 soft 全量 + 第一版报告 ----
# （对齐检查结论：已知 b8 批量差异，约 2% 翻转，主比较统一用 eval_soft，见 pilot_status.md。
#   B0 的 b8 全量只在最后作附表参考，见 2.8。）
if [ ! -f "$OUTD/.stage2_bc_eval_done" ]; then
  run_eval b0-hard "$B0" "" hard "" "$OUTD/b0_hard.jsonl"

  "$PY" "$CV1/scripts/make_ids.py" --hard-jsonl "$OUTD/b0_hard.jsonl" \
    --out-prefix "$OUTD/ids2" --n-soft 100000 --n-scan-pos 50 --n-scan-neg 50 \
    --n-ablation 400

  t0=$(date +%s)
  run_eval c1-soft "$C1" "$C1/soft_mixer.pt" soft "" "$OUTD/c1_soft.jsonl"
  say "C1 soft 1600 条用时 $(( ($(date +%s) - t0) / 60 )) 分钟（batch=1）"

  say "写第一版 pilot_v1_report.md（B0/C1 主要指标）"
  "$PY" "$CV1/scripts/pilot_metrics.py" > "$LOGD/pilot_metrics_bc.log" 2>&1 \
    || { say "pilot_metrics（B0/C1 版）FAIL"; exit 1; }
  say "第一版 pilot_v1_report.md 已生成"
  touch "$OUTD/.stage2_bc_eval_done"
fi

# ---- 2.3b 探针重跑（position_ids 修复版）+ 新旧对比 ----
# 旧探针 forward_birads 的 prefill 不传 position_ids，走了 soft_decode 修掉的同款 bug；
# 放在第一版报告之后，不挡 B0/C1。新旧结果并排追加进最终报告（见 2.7）。
if [ ! -f "$OUTD/.probe_rerun_done" ]; then
  say "2.3b: soft_evidence_probe 重跑（position_ids 修复版，同 adapter 同数据）"
  PBA=$MDIR/Qwen3.5-4B-lora-2view-os-sub-vt
  "$PY" "$CV1/scripts/soft_evidence_probe.py" --adapter "$PBA" \
    --data "$PROJ/data/splits/test_local.json" \
    --probs "$PROJ/outputs/scoring/os_sub_vt_test_b8_with_probs.jsonl" \
    --out "$CV1/outputs/soft_evidence_test_fixed.jsonl" --split test --limit 0 \
    --sens-out "$CV1/outputs/soft_evidence_sensitivity_fixed.json" --sens-n 20 \
    > "$LOGD/probe_rerun_test.log" 2>&1 || { say "2.3b: test 重跑 FAIL"; exit 1; }
  "$PY" "$CV1/scripts/soft_evidence_probe.py" --adapter "$PBA" \
    --data "$PROJ/data/splits/val_raw_local.json" \
    --probs "$PROJ/outputs/scoring/os_sub_vt_val_b8_with_probs.jsonl" \
    --out "$CV1/outputs/soft_evidence_val_fixed.jsonl" --split val --limit 0 \
    > "$LOGD/probe_rerun_val.log" 2>&1 || { say "2.3b: val 重跑 FAIL"; exit 1; }
  "$PY" "$CV1/scripts/probe_compare.py" > "$LOGD/probe_compare.log" 2>&1 \
    || { say "2.3b: probe_compare FAIL"; exit 1; }
  say "2.3b: 完成，新旧对比见 $OUTD/probe_compare.md"
  touch "$OUTD/.probe_rerun_done"
fi

# ---- 2.4 训练 C2 + 评测 ----
if [ ! -f "$OUTD/.stage2_c2_done" ]; then
  say "==== 阶段二: C2 训练 ===="
  train_stage2 C2 "$SOFT_ARGS --detach-p --resume off" yes || exit 1
  run_eval c2-soft "$C2" "$C2/soft_mixer.pt" soft "" "$OUTD/c2_soft.jsonl"
  touch "$OUTD/.stage2_c2_done"
  say "阶段二: C2 训练+评测完成"
fi

# ---- 2.5 消融（400 乳房 study 抽样）+ 扫描 ----
if [ ! -f "$OUTD/.stage2_abl_done" ]; then
  say "消融评测开始（teacher/swap 用 400 乳房，study 抽样 seed 42，各组同批）"
  run_eval b0-teacher "$B0" "" teacher "$OUTD/ids2_ablation.json" "$OUTD/b0_teacher.jsonl"
  run_eval c1-teacher "$C1" "$C1/soft_mixer.pt" teacher "$OUTD/ids2_ablation.json" "$OUTD/c1_teacher.jsonl"
  run_eval c2-teacher "$C2" "$C2/soft_mixer.pt" teacher "$OUTD/ids2_ablation.json" "$OUTD/c2_teacher.jsonl"
  run_eval c1-swap "$C1" "" swap_hard "$OUTD/ids2_ablation.json" "$OUTD/c1_swap.jsonl"
  run_eval c2-swap "$C2" "" swap_hard "$OUTD/ids2_ablation.json" "$OUTD/c2_swap.jsonl"
  say "概率扫描开始（GT>=3/GT<=2 各 50）"
  run_eval c1-scan "$C1" "$C1/soft_mixer.pt" scan "$OUTD/ids2_scan.json" "$OUTD/c1_scan.jsonl" logit_mlp 1
  run_eval c2-scan "$C2" "$C2/soft_mixer.pt" scan "$OUTD/ids2_scan.json" "$OUTD/c2_scan.jsonl" logit_mlp 1
  touch "$OUTD/.stage2_abl_done"
  say "消融+扫描完成"
fi

# ---- 2.6 遮挡 ----
if [ ! -f "$OUTD/.occl_done" ]; then
  say "==== 遮挡机制分析开始（B0 hard / C1 soft × 4 conditions × 192）===="
  "$PY" "$CV1/scripts/make_occl_jsonl.py" --out-prefix "$OUTD/occl" || { say "遮挡清单构建 FAIL"; exit 1; }
  for cond in orig lesion ctrl1 ctrl2; do
    run_eval "occl-b0-$cond" "$B0" "" hard "" "$OUTD/occl_b0_$cond.jsonl"
  done
  for cond in orig lesion ctrl1 ctrl2; do
    run_eval "occl-c1-$cond" "$C1" "$C1/soft_mixer.pt" soft "" "$OUTD/occl_c1_$cond.jsonl"
  done
  "$PY" "$CV1/scripts/pilot_occlusion.py" || { say "遮挡分析 FAIL"; exit 1; }
  touch "$OUTD/.occl_done"
fi

# ---- 2.6b 起点 scan 对照（冻结代码，从断点续跑；规模不变 100×11）----
if [ ! -f "$OUTD/.start_scan_done" ]; then
  say "2.6b: 起点 scan 对照（frozen_stage1/eval_soft.py，linear 注入）"
  for f in infer/soft_decode.py train/soft_mixer.py; do
    cmp -s "$CV1/$f" "$CV1/frozen_stage1/$(basename "$f")" \
      || { say "2.6b: $f 与冻结版本不一致，停下（如需更新请重新冻结）"; exit 1; }
  done
  "$PY" "$CV1/frozen_stage1/eval_soft.py" --adapter "$START_ADAPTER" \
    --soft-mode linear --mode scan \
    --ids-file "$OUTD/ids_scan.json" --out "$OUTD/start_scan.jsonl" \
    > "$LOGD/pilot_start_scan.log" 2>&1 || { say "2.6b 起点 scan FAIL"; exit 1; }
  n=$(wc -l < "$OUTD/start_scan.jsonl")
  [ "$n" -eq 1100 ] || { say "2.6b 起点 scan 行数不足（$n != 1100），停下"; exit 1; }
  say "2.6b done: $n 行"
  touch "$OUTD/.start_scan_done"
fi

# ---- 2.7 最终报告 ----
say "==== 汇总指标 + 差值 bootstrap + 最终报告 ===="
"$PY" "$CV1/scripts/pilot_metrics.py" > "$LOGD/pilot_metrics.log" 2>&1 \
  || { say "pilot_metrics FAIL"; exit 1; }
[ -f "$OUTD/probe_compare.md" ] && cat "$OUTD/probe_compare.md" >> "$CV1/outputs/pilot_v1_report.md"
say "==== pilot_v1_report.md 完成，A40 停下等用户 ===="

# ---- 2.8 附表: B0 b8 全量（仅参考，不进主指标；跑在报告之后，不影响主比较口径）----
if [ ! -f "$OUTD/.b0_b8_ref_done" ]; then
  say "附表: B0 b8 全量 1600（仅参考）"
  "$PY" "$PROJ/scripts/vlm/infer_2view_batched.py" --adapter "$B0" \
    --data "$PROJ/data/splits/val_raw_local.json" --batch-size 8 \
    --out-suffix _pilot_b0 \
    --out-probs "$OUTD/b0_b8_probs.jsonl" --tag pilot-b0 \
    > "$LOGD/pilot_eval_b0-b8.log" 2>&1 || { say "附表[B0-b8-1600] FAIL"; exit 1; }
  say "附表[B0-b8-1600] done: $(wc -l < "$OUTD/b0_b8_probs.jsonl") 行"
  touch "$OUTD/.b0_b8_ref_done"
fi
