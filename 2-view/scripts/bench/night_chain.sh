#!/bin/bash
# 夜间串行链（顺序：两次训练 -> ep2 评测 -> 原分辨率评测 -> 报告）
#   A: 等 ep2 训练结束 (EP2_TRAIN_DONE)
#   B: 原生分辨率 2 epoch 训练（最多 3 次重启, expandable_segments）
#   C1: ep2 评测 e1/e2 × val+test (b8, 786432) + ep2_vs_ep3.md
#   C2: 原生评测 val+test (b8, 1386240, batch 8->4->2)
#   D: resolution_report.md
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
BASE=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt
EP2=$BASE-ep2
NATIVE=$BASE-ep2-native
CFG=$PROJ/configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml
CHAIN=$PROJ/logs/vlm/night_chain.log
TLOG=$PROJ/logs/vlm/native_train.log
GPU=$PROJ/logs/vlm/native_train_gpu.txt
BL=$PROJ/logs/vlm/native_eval_batch.txt
EP2LOG=$PROJ/logs/vlm/ep2_chain.log
MTLOG=$PROJ/logs/vlm/memtest_native.log
VAL=$PROJ/data/splits/val_raw_local.json
TEST=$PROJ/data/splits/test_local.json

cd "$PROJ" || exit 1
mkdir -p logs/vlm outputs/eval outputs/scoring outputs/predictions
ts() { date +"%H:%M:%S"; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

echo "[$(ts)] ===== NIGHT CHAIN START (train,train,eval,eval,report) =====" | tee -a "$CHAIN"

# ---------------- Phase A: 等 ep2 训练结束 ----------------
while ! grep -q "EP2_TRAIN_DONE" "$EP2LOG" 2>/dev/null; do
  if ! pgrep -f "ep2_chain.sh" >/dev/null && ! tmux has-session -t ep2 2>/dev/null; then
    if grep -q "TRAIN_RC=" "$MTLOG" 2>/dev/null; then
      [ -f "$PROJ/logs/vlm/.ep2_seen" ] || date +%s > "$PROJ/logs/vlm/.ep2_seen"
      SEEN=$(cat "$PROJ/logs/vlm/.ep2_seen" 2>/dev/null || echo 0)
      NOW=$(date +%s)
      if [ $((NOW - SEEN)) -ge 180 ]; then
        if grep -q "EP2 CHAIN START" "$EP2LOG" 2>/dev/null; then
          echo "[$(ts)] ep2 训练已结束但无 DONE 标记 -> 继续后续步骤" | tee -a "$CHAIN"
          break
        fi
        echo "[$(ts)] ep2 未启动(>180s) -> 现在启动 ep2 训练" | tee -a "$CHAIN"
        tmux new-session -d -s ep2 "bash $PROJ/scripts/bench/ep2_chain.sh"
        sleep 60
      fi
    fi
  fi
  sleep 60
done
echo "[$(ts)] ep2 训练阶段结束，进入原生分辨率训练" | tee -a "$CHAIN"

# ---------------- Phase B: 原生分辨率训练（最多 3 次） ----------------
bash scripts/bench/gpu_sampler.sh "$GPU" &
SPID=$!
NOK=0
for n in 1 2 3; do
  echo "[$(ts)] native train attempt $n" | tee -a "$CHAIN"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    llamafactory-cli train "$CFG" >> "$TLOG" 2>&1
  rc=$?
  echo "[$(ts)] attempt $n rc=$rc" | tee -a "$CHAIN"
  if [ "$rc" -eq 0 ]; then NOK=1; break; fi
  tail -50 "$TLOG" > "logs/vlm/native_fail_${n}.log"
  echo "[$(ts)] fail log -> logs/vlm/native_fail_${n}.log" | tee -a "$CHAIN"
  sleep 30
done
kill $SPID 2>/dev/null
sleep 1
PEAK=$(sort -n "$GPU" 2>/dev/null | tail -1)
echo "[$(ts)] native train done, ok=$NOK, peak_gpu_smi_mib=$PEAK" | tee -a "$CHAIN"
printf "NATIVE_TRAIN_OK=%s\nNATIVE_PEAK_SMI_MIB=%s\n" "$NOK" "$PEAK" > logs/vlm/native_train_summary.txt

# ---------------- Phase C1: ep2 评测 (e1/e2 × val+test, b8, 786432) ----------------
for pair in 439:e1 878:e2; do
  CKPT=${pair%%:*}
  TAG=${pair##*:}
  for split in val test; do
    if [ "$split" = "val" ]; then DATA=$VAL; N=1600; else DATA=$TEST; N=2000; fi
    PROBS=$PROJ/outputs/scoring/vt-ep2-${TAG}_${split}_b8_with_probs.jsonl
    PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep2-${TAG}_${split}_b8.jsonl
    METRICS=$PROJ/outputs/eval/metrics_probs_vt-ep2-${TAG}_${split}_b8.json
    if [ "$(nlines "$PROBS")" -lt "$N" ]; then
      echo "[$(ts)] ep2 infer $TAG/$split" | tee -a "$CHAIN"
      "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$EP2/checkpoint-$CKPT" \
        --data "$DATA" --limit 0 --batch-size 8 --out-suffix "_vt-ep2-${TAG}_${split}_b8" \
        --out-pred "$PRED" --out-probs "$PROBS" --tag "vt-ep2-${TAG}-${split}" 2>&1 | tail -3 | tee -a "$CHAIN"
    fi
    if [ ! -f "$METRICS" ]; then
      echo "[$(ts)] ep2 summarize $TAG/$split" | tee -a "$CHAIN"
      "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
        --tag "vt-ep2-${TAG}-${split}" --n-boot 1000 2>&1 | tail -3 | tee -a "$CHAIN"
    fi
  done
done
echo "[$(ts)] ep2 eval done -> ep2_vs_ep3.md" | tee -a "$CHAIN"
"$PY" scripts/bench/ep2_vs_ep3.py 2>&1 | tail -5 | tee -a "$CHAIN"

# ---------------- Phase C2: 原生分辨率评测 (val+test, 1386240, batch 8->4->2) ----------------
: > "$BL"
for split in val test; do
  if [ "$split" = "val" ]; then DATA=$VAL; N=1600; else DATA=$TEST; N=2000; fi
  PROBS=$PROJ/outputs/scoring/vt-ep2native_${split}_b8_with_probs.jsonl
  PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep2native_${split}_b8.jsonl
  METRICS=$PROJ/outputs/eval/metrics_probs_vt-ep2native_${split}_b8.json

  if [ "$(nlines "$PROBS")" -lt "$N" ]; then
    for bs in 8 4 2; do
      echo "[$(ts)] native infer $split batch=$bs" | tee -a "$CHAIN"
      "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$NATIVE" \
        --data "$DATA" --limit 0 --batch-size "$bs" --image-max-pixels 1386240 \
        --out-suffix "_vt-ep2native_${split}_b8" \
        --out-pred "$PRED" --out-probs "$PROBS" --tag "ep2native-${split}" 2>&1 | tail -3 | tee -a "$CHAIN"
      rc=${PIPESTATUS[0]}
      if [ "$rc" -eq 0 ]; then
        echo "$split,batch=$bs,rc=0" >> "$BL"
        break
      fi
      echo "$split,batch=$bs,rc=$rc (fallback)" >> "$BL"
      echo "[$(ts)] $split batch=$bs 失败 rc=$rc -> 降档" | tee -a "$CHAIN"
      tail -20 "$CHAIN" > "logs/vlm/native_eval_fail_${split}_${bs}.log"
    done
  else
    echo "$split,skipped(probs complete)" >> "$BL"
  fi

  if [ ! -f "$METRICS" ]; then
    echo "[$(ts)] native summarize $split" | tee -a "$CHAIN"
    "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
      --tag "ep2native-${split}" --n-boot 1000 2>&1 | tail -3 | tee -a "$CHAIN"
  fi
done

# ---------------- Phase D: 报告 ----------------
echo "[$(ts)] === resolution report ===" | tee -a "$CHAIN"
"$PY" scripts/bench/resolution_report.py 2>&1 | tail -20 | tee -a "$CHAIN"
echo "[$(ts)] NIGHT_ALL_DONE" | tee -a "$CHAIN"
