#!/usr/bin/env python3
"""
VinDr-Mammo VLM 推理 - CoT Prompt (分步推理后输出 JSON)

用法:  python vindr_infer/run_cot.py
流程:  1) build_mllm.py --prompt cot    注入 prompt -> mllm_infer.json (覆盖式)
       2) llamafactory-cli train infer.yaml  批量推理 (自带 tqdm 进度条, 实时流式输出)
       3) 解析 predict_results.json, 输出统计
日志:  outputs/cot/infer.log  (完整推理输出, 含进度条记录)
环境:  到 4090 机器上只需改顶部路径常量 (CONDA_ENV/模型/数据路径)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

# ===================== 配置 (跨机器迁移时改这里) =====================
LLAMAFACTORY_DIR = Path("/Users/haozeyuan/Desktop/phd/vlm/9_9/LlamaFactory")
CONDA_ENV = "netest"                       # conda 环境名 (4090 上改为对应环境)
PROMPT = "cot"                             # 本脚本对应 prompt
CONFIG = str(LLAMAFACTORY_DIR / "vindr_infer" / "infer.yaml")
BUILD = str(LLAMAFACTORY_DIR / "vindr_infer" / "build_mllm.py")
OUTPUT = str(LLAMAFACTORY_DIR / "vindr_infer" / "outputs" / PROMPT)
CLI = str(Path.home() / "anaconda3" / "envs" / CONDA_ENV / "bin" / "llamafactory-cli")
PY = str(Path.home() / "anaconda3" / "envs" / CONDA_ENV / "bin" / "python")
# ====================================================================


def run_build():
    """[1/3] 注入 prompt 到 mllm_infer.json"""
    print(f"\n[1/3] 注入 prompt ({PROMPT}) -> mllm_infer.json")
    subprocess.run([PY, BUILD, "--prompt", PROMPT], check=True)


def run_infer():
    """[2/3] llamafactory-cli 批量推理 (流式输出 tqdm 进度条 + 写日志)"""
    print(f"\n[2/3] llamafactory-cli train (批量推理, 进度条实时显示)")
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    log_path = Path(OUTPUT) / "infer.log"
    cmd = [CLI, "train", CONFIG,
           f"output_dir={OUTPUT}", "overwrite_output_dir=true"]
    # tqdm 在管道下也输出进度 (TQDM_MININTERVAL=1 每秒刷新, FORCE=1 强制显示)
    env = {**__import__("os").environ,
           "TQDM_MININTERVAL": "1", "TQDM_DISABLE": "0"}
    n = 0  # 已输出行数 (粗略进度)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=str(LLAMAFACTORY_DIR),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            print(line, end="")          # 终端实时显示 (含 tqdm 进度条)
            f.write(line)                # 日志记录进度
            n += 1
        rc = proc.wait()
    if rc != 0:
        print(f"\n[ERROR] 推理失败 rc={rc}, 完整日志: {log_path}")
        sys.exit(1)
    print(f"  日志已记录: {log_path} ({n} 行)")


def show_stats():
    """[3/3] 解析预测结果, 输出统计"""
    print(f"\n[3/3] 结果统计")
    rf = Path(OUTPUT) / "predict_results.json"
    if rf.exists():
        results = json.load(open(rf))
        print(f"  预测结果文件: {rf}")
        print(f"  预测条数: {len(results)}")
        if results:
            print(f"  第1条预览: {str(results[0])[:120]}...")
    else:
        print(f"  [提示] {rf} 不存在, 检查 {OUTPUT}/ 目录内容")
        print(f"  目录内容: {list(Path(OUTPUT).iterdir())}")


def main():
    t0 = time.time()
    print("=" * 55)
    print(f"VinDr-Mammo 推理 - {PROMPT.upper()} Prompt (step-by-step)")
    print(f"模型: Qwen3.5-4B | 数据: test_raw.json (4000条) | 输出: {OUTPUT}")
    print("=" * 55)
    run_build()
    run_infer()
    show_stats()
    print(f"\n[完成] 总耗时 {time.time() - t0:.0f}s | 结果目录: {OUTPUT}/")


if __name__ == "__main__":
    main()
