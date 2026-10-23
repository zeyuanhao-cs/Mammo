#!/usr/bin/env python3
"""norule prompt 消融 (os_sub, test 2000, 只推理, 不训练).

删掉 prompt 里的 findings->BI-RADS 一致性规则后, 检查 BI-RADS 是否仍被
findings 前缀决定 (zero / inject 干预, 子集定义同 analyze_prefix_zi.py).

阶段 (--phase): infer | greedy | prefix | analyze | all
  infer  : infer_2view.py 逻辑 (PROMPT_FILE 打补丁 -> prompt_norule.txt)
           -> outputs/predictions/predictions_qwen3.5-4b_os_sub_norule_2000.jsonl
  greedy : score_2view.py 逻辑 (同补丁, 逐 token 一致)
           -> outputs/prefix_probe_norule/probs_greedy_test.jsonl
  prefix : run_probe.run_stage_prefix 的 zero + inject (instruction=prompt_norule)
           -> outputs/prefix_probe_norule/probs_{zero,inject}find_test.jsonl
  analyze: analyze_prefix_zi.py (OUT_ROOT 打补丁 -> prefix_probe_norule, CPU)

所有 jsonl 断点续跑; 原 outputs/prefix_probe/ 一律不动.
"""
import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/prefix_probe
SCRIPTS_DIR = THIS_DIR.parent                       # 2-view/scripts
PROJ_DIR = SCRIPTS_DIR.parent                       # 2-view
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(THIS_DIR))

PROMPT_NORULE = Path("/home/zhao/Mammography/prompt_norule.txt")
OUT_ROOT = PROJ_DIR / "outputs" / "prefix_probe_norule"
PRED_FILE = PROJ_DIR / "outputs/predictions/predictions_qwen3.5-4b_os_sub_norule_2000.jsonl"

import infer_2view as I2V   # noqa: E402
import score_2view as S2V   # noqa: E402
import run_probe as RP      # noqa: E402


def phase_infer(limit: int):
    """贪心推理 (infer_2view.py 原逻辑, prompt 换成 norule)."""
    I2V.PROMPT_FILE = PROMPT_NORULE
    sys.argv = ["infer_2view.py", "--model", "qwen3.5-4b",
                "--adapter", RP.DEFAULT_ADAPTER, "--limit", str(limit),
                "--out-suffix", "_os_sub_norule", "--data", RP.TEST_DATA]
    I2V.main()


def phase_greedy(limit: int):
    """贪心预测重打分 (score_2view.py 原逻辑, prompt 换成 norule)."""
    S2V.PROMPT_FILE = PROMPT_NORULE
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["score_2view.py", "--adapter", RP.DEFAULT_ADAPTER,
                "--pred-file", str(PRED_FILE),
                "--out", str(OUT_ROOT / "probs_greedy_test.jsonl"),
                "--limit", str(limit), "--tag", "norule_greedy",
                "--data", RP.TEST_DATA]
    S2V.main()


def phase_prefix(limit: int, budget_min: float, t_start: float):
    """zero / inject 前缀打概率 (run_probe 原路径, instruction=prompt_norule)."""
    sys.path.insert(0, str(SCRIPTS_DIR / "ablation"))
    from run_ablation import load_model
    I2V.PROMPT_FILE = PROMPT_NORULE
    instruction = I2V.load_instruction()
    bundle = load_model(RP.DEFAULT_ADAPTER)
    test_recs, test_order = RP.load_json_recs(RP.TEST_DATA)
    preds = RP.load_jsonl_preds(str(PRED_FILE))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for mode in ("zero", "inject"):
        RP.run_stage_prefix(bundle, instruction, test_recs, test_order, preds,
                            mode, OUT_ROOT, limit, t_start, budget_min)


def phase_analyze():
    """子集分析 (analyze_prefix_zi.py 原逻辑, OUT_ROOT 换成 prefix_probe_norule)."""
    import analyze_prefix_zi as AZ
    AZ.OUT_ROOT = OUT_ROOT
    sys.argv = ["analyze_prefix_zi.py"]
    AZ.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["infer", "greedy", "prefix", "analyze", "all"])
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 条 (0=全部, smoke 用)")
    ap.add_argument("--budget-min", type=float, default=1e9,
                    help="prefix 阶段的作业级预算 (分钟, 从驱动启动起算)")
    ap.add_argument("--skip-analyze", action="store_true")
    args = ap.parse_args()

    t_start = time.perf_counter()
    if args.phase in ("infer", "all"):
        print(f"\n===== [phase infer] limit={args.limit} =====", flush=True)
        phase_infer(args.limit)
    if args.phase in ("greedy", "all"):
        print(f"\n===== [phase greedy] limit={args.limit} =====", flush=True)
        phase_greedy(args.limit)
    if args.phase in ("prefix", "all"):
        print(f"\n===== [phase prefix] limit={args.limit} =====", flush=True)
        phase_prefix(args.limit, args.budget_min, t_start)
    if args.phase in ("analyze", "all") and not args.skip_analyze:
        print("\n===== [phase analyze] =====", flush=True)
        phase_analyze()
    print(f"\n===== run_norule done ({(time.perf_counter() - t_start) / 60:.1f} min) =====",
          flush=True)


if __name__ == "__main__":
    main()
