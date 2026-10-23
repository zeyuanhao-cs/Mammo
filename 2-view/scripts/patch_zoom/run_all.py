#!/usr/bin/env python3
"""patch-zoom 全流程驱动（单个 GPU 作业内执行）.

阶段:
  1) 冒烟: 每个尺度 20 步训练 + 16 块 test 推理, 实测显存/每步耗时/推理速度
  2) 预算闸门: 估算总时长 > --gate-min 则写 gate.json 并退出(不开始正式训练)
  3) 正式: 按 x1 -> x2 -> x074 依次 "训练(2 epoch, save_steps=50) -> test 推理",
     每组完成后写 $OUT/done/<scale>.done; 重提时跳过已完成组,
     未完成组从 output_dir 下最近的 checkpoint 续训 resume_from_checkpoint=true
  4) 评测 + 报告: 三组 probs 齐全时跑 eval_patch.py 并生成 patch_zoom_report.md
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

SCALES = ["x1", "x2", "x074"]
CONFIGS = Path("/home/zhao/Mammography/Mammo/2-view/configs")
ENGINE = Path("/home/zhao/Mammography/Mammo")
TRAIN_CURVE_RE = re.compile(r"(\d+)/(\d+)\s*\[.*?<?\s*([\d.]+)\s*(s/it|it/s)")
LOAD_RE = re.compile(r"\[model\] loaded in ([\d.]+)s")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_cfg(scale):
    return (CONFIGS / f"patch_zoom_{scale}.yaml").read_text(encoding="utf-8")


def make_yaml(scale, out_dir, *, smoke=False, resume=False, max_steps=None,
              logging_steps="10", save_steps="50"):
    lines = []
    for line in read_cfg(scale).splitlines():
        key = line.split(":", 1)[0].strip() if ":" in line else ""
        if key == "output_dir":
            lines.append(f"output_dir: {out_dir}")
        elif key == "save_steps":
            lines.append(f"save_steps: {save_steps}")
        elif key == "logging_steps":
            lines.append(f"logging_steps: {logging_steps}")
        elif key == "resume_from_checkpoint":
            continue
        else:
            lines.append(line)
    if max_steps:
        lines.append(f"max_steps: {max_steps}")
    lines.append(f"resume_from_checkpoint: {'true' if resume else 'false'}")
    path = Path(f"/tmp/pz_{scale}_{'smoke' if smoke else 'train'}.yaml")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class MemMonitor(threading.Thread):
    """采样 nvidia-smi 显存占用, 记录峰值 (MiB)."""

    def __init__(self, interval=3.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=15)
        return self.peak


def run_train(scale, yaml_path, log_path, monitor=False):
    mon = MemMonitor() if monitor else None
    if mon:
        mon.start()
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.call(["llamafactory-cli", "train", str(yaml_path)],
                             stdout=f, stderr=subprocess.STDOUT, cwd=str(ENGINE))
    wall = time.perf_counter() - t0
    peak = mon.stop() if mon else None
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    per_step, steps = None, None
    for m in TRAIN_CURVE_RE.finditer(text):
        steps = int(m.group(2))
        v = float(m.group(3))
        per_step = v if m.group(4) == "s/it" else (1.0 / v if v else None)
    return {"rc": rc, "wall_s": wall, "s_per_step": per_step, "n_steps": steps,
            "peak_mem_mib": peak}


def run_infer(scale, adapter, out_probs, log_path, limit=0, batch=8):
    cmd = ["python", "-u", str(ENGINE / "2-view/scripts/patch_zoom/infer_patch.py"),
           "--adapter", str(adapter), "--scale", scale, "--out-probs", str(out_probs),
           "--batch-size", str(batch)]
    if limit:
        cmd += ["--limit", str(limit)]
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ENGINE))
    wall = time.perf_counter() - t0
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    m = LOAD_RE.search(text)
    load = float(m.group(1)) if m else 0.0
    return {"rc": rc, "wall_s": wall, "load_s": load}


def latest_ckpt(out_dir):
    cks = [p for p in glob.glob(str(Path(out_dir) / "checkpoint-*")) if Path(p).is_dir()]
    if not cks:
        return None
    return max(cks, key=lambda p: int(Path(p).name.split("-")[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/zhao/Mammography/Mammo/2-view/outputs/patch_zoom")
    ap.add_argument("--model-root", default="/nesi/project/massey04900/Mammography/model")
    ap.add_argument("--logdir", default=str(ENGINE / "2-view/logs"))
    ap.add_argument("--gate-min", type=float, default=105.0)
    ap.add_argument("--budget-min", type=float, default=115.0)
    ap.add_argument("--smoke-n", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--stages", default="all", choices=["all", "smoke", "train", "eval"])
    args = ap.parse_args()

    out = Path(args.out)
    logdir = Path(args.logdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "done").mkdir(exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    smoke_json = out / "smoke_metrics.json"

    t_start = time.perf_counter()

    # ---------------- 1. smoke ----------------
    if args.stages in ("all", "smoke"):
        metrics = {}
        if smoke_json.exists():
            metrics = json.loads(smoke_json.read_text())
        for scale in SCALES:
            if scale in metrics and metrics[scale].get("ok"):
                continue
            log(f"--- smoke {scale}: 20 steps + {args.smoke_n} patches infer")
            tmp = Path(args.model_root) / f"pz_smoke_{os.environ.get('SLURM_JOB_ID', 'local')}_{scale}"
            y = make_yaml(scale, tmp, smoke=True, max_steps=20,
                          logging_steps="1", save_steps="10000")
            tr = run_train(scale, y, logdir / f"patch_zoom_smoke_train_{scale}.log", monitor=True)
            inf = {"rc": -1, "wall_s": 0.0, "load_s": 0.0}
            if tr["rc"] == 0 and (tmp / "adapter_config.json").exists():
                inf = run_infer(scale, tmp, f"/tmp/pz_smoke_{scale}_probs.jsonl",
                                logdir / f"patch_zoom_smoke_infer_{scale}.log",
                                limit=args.smoke_n)
            shutil.rmtree(tmp, ignore_errors=True)
            metrics[scale] = {"ok": tr["rc"] == 0 and inf["rc"] == 0,
                              "train": tr, "infer": inf}
            smoke_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
            log(f"    s/step={tr['s_per_step']} infer_wall={inf['wall_s']:.1f}s "
                f"load={inf['load_s']:.1f}s rc=({tr['rc']},{inf['rc']})")

        # ---------------- 2. gate ----------------
        proj = {}
        pre_est = 90.0  # 预处理保守估计(缓存命中时更短)
        n = int(round(args.epochs * 125))  # 1995 样本 / 等效 batch16 -> 125 步/epoch
        for scale in SCALES:
            m = metrics.get(scale, {})
            if not m.get("ok"):
                proj[scale] = None
                continue
            s = m["train"]["s_per_step"]
            inf_rate = max(m["infer"]["wall_s"] - m["infer"]["load_s"], 0.0) / args.smoke_n
            proj[scale] = {"train_s": n * s, "infer_s": 672 * inf_rate,
                           "total_s": pre_est + n * s + 672 * inf_rate}
        failed = [s for s in SCALES if not proj.get(s)
                  or not metrics.get(s, {}).get("ok")
                  or not metrics[s]["train"].get("s_per_step")]
        total = sum(v["total_s"] for v in proj.values() if v)
        gate = {"projected_total_min": total / 60.0, "gate_min": args.gate_min,
                "n_steps": n, "per_scale": proj, "smoke_failed": failed,
                "pass": bool(not failed and total <= args.gate_min * 60)}
        (out / "gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False))
        log(f"=== projection: {total / 60.0:.1f} min (gate {args.gate_min:.0f} min) "
            f"pass={gate['pass']}")
        if failed:
            log(f"!!! 冒烟失败的尺度: {failed}, 不开始正式训练")
            return 2
        if not gate["pass"]:
            log("!!! 预估总时长超过闸门, 不开始正式训练, 请人工确认后再提交")
            return 2

    if args.stages in ("smoke",):
        return 0

    # ---------------- 3. train -> infer ----------------
    for scale in SCALES:
        marker = out / "done" / f"{scale}.done"
        if marker.exists():
            log(f"=== {scale} skipped (marker: {marker.read_text().strip()})")
            continue

        remaining = args.budget_min * 60 - (time.perf_counter() - t_start)
        if remaining < 20 * 60:
            log(f"!!! {scale} 剩余 {remaining / 60:.0f} min, 不足以开始新组, 退出等待重提")
            return 3

        adapter = Path(args.model_root) / f"Qwen3.5-4B-lora-patch-{scale}"
        ck = latest_ckpt(adapter)
        log(f"=== train {scale} (resume={ck or 'none'})")
        y = make_yaml(scale, adapter, resume=bool(ck))
        tr = run_train(scale, y, logdir / f"patch_zoom_train_{scale}.log", monitor=True)
        log(f"    train rc={tr['rc']} wall={tr['wall_s'] / 60:.1f} min steps={tr['n_steps']}")
        if tr["rc"] != 0 or not (adapter / "adapter_config.json").exists():
            log(f"!!! train {scale} failed, 停止")
            return 4

        log(f"=== infer {scale}")
        inf = run_infer(scale, adapter, out / f"{scale}_test_probs.jsonl",
                        logdir / f"patch_zoom_infer_{scale}.log")
        n_lines = sum(1 for _ in open(out / f"{scale}_test_probs.jsonl", encoding="utf-8")) \
            if (out / f"{scale}_test_probs.jsonl").exists() else 0
        log(f"    infer rc={inf['rc']} wall={inf['wall_s'] / 60:.1f} min lines={n_lines}")
        if inf["rc"] != 0 or n_lines < 672:
            log(f"!!! infer {scale} incomplete ({n_lines}/672), 停止; 重提时续跑")
            return 5
        marker.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} adapter={adapter}\n",
                          encoding="utf-8")

    # ---------------- 4. eval + report ----------------
    return run_eval(out, args)


def run_eval(out, args):
    probs = [out / f"{s}_test_probs.jsonl" for s in SCALES]
    if not all(p.exists() for p in probs):
        log("!!! probs 不齐, 暂不评测")
        return 6
    log("=== eval")
    cmd = ["python", "-u", str(ENGINE / "2-view/scripts/patch_zoom/eval_patch.py"),
           "--probs", *[str(p) for p in probs],
           "--scales", *SCALES,
           "--meta", "/nesi/project/massey04900/Mammography/patch_zoom/meta_test.jsonl",
           "--stats", "/nesi/project/massey04900/Mammography/patch_zoom/stats.json",
           "--out", str(out / "patch_zoom_eval.json"),
           "--md", str(out / "patch_zoom_eval_table.md"),
           "--report", str(ENGINE / "2-view/outputs/eval/patch_zoom_report.md"),
           "--smoke", str(out / "smoke_metrics.json")]
    with open(Path(args.logdir) / "patch_zoom_eval.log", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ENGINE))
    log(f"    eval rc={rc}")
    if rc == 0:
        print((out / "patch_zoom_eval_table.md").read_text(encoding="utf-8"), flush=True)
    return 0 if rc == 0 else 7


if __name__ == "__main__":
    raise SystemExit(main())
