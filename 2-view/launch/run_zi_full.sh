#!/usr/bin/env bash
# =============================================================================
# zero / inject 前缀干预 全量 (双顺序模型, prompt_order_a)
#   1) run_probe.py --zi --mode full   2000 条 x {greedy, zero, inject}
#   2) analyze_prefix_zi.py            子集分析 + study 配对 bootstrap -> md
#
#   产物: outputs/prefix_probe_dualorder_A/
#           probs_{greedy,zero,inject}_test.jsonl
#           prefix_zi_report.md
#   日志: logs/vlm/prefix_zi_full.log, logs/vlm/prefix_zi_analyze.log
#
# 可续跑: probs_*.jsonl 按 id 续写, 中断后重跑本脚本即可。
# 用法: bash 2-view/launch/run_zi_full.sh
# =============================================================================
set -u
ROOT=/home/z/zhao/Mammography
PROJ=$ROOT/Mammo/2-view
PY=$ROOT/miniconda3/envs/mammo/bin/python
LOGDIR=$PROJ/logs/vlm
OUT=$PROJ/outputs/prefix_probe_dualorder_A

export HF_HOME=$ROOT/.hf_home
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
cd "$PROJ"
mkdir -p "$LOGDIR" "$OUT"

echo "=== [zi-full] start $(date '+%F %T') ==="
"$PY" scripts/prefix_probe/run_probe.py --zi --mode full \
    --adapter "$ROOT/model/Qwen3.5-4B-lora-2view-dualorder" \
    --model-path "$ROOT/model/Qwen3.5-4B" \
    --prompt-file dataset_build/prompt_order_a.txt \
    > "$LOGDIR/prefix_zi_full.log" 2>&1
rc=$?
echo "[zi-full] exit=$rc  $(date '+%F %T')"
if [ $rc -ne 0 ]; then
    echo "!!! [zi-full] 失败, 不跑分析"
    exit $rc
fi

"$PY" scripts/prefix_probe/analyze_prefix_zi.py --out-root "$OUT" \
    > "$LOGDIR/prefix_zi_analyze.log" 2>&1
rc=$?
echo "[zi-analyze] exit=$rc  $(date '+%F %T')"
echo "=== ALL DONE $(date '+%F %T') ==="
