#!/usr/bin/env python3
"""分辨率对比报告: e3(786432,3ep) / ep2(786432,2ep) / ep2-native(1386240,2ep)
 -> outputs/eval/resolution_report.md
"""
import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
PROJ = THIS.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

E = PROJ / "outputs" / "eval"
S = PROJ / "outputs" / "scoring"
OUT = E / "resolution_report.md"
N_BOOT, SEED = 1000, 42
FIND = C.FINDINGS

MODELS = [
    ("e3 (786432, 3ep)", (E / "metrics_overfit_val_e3_b8.json", S / "overfit_val_e3_b8_with_probs.jsonl"),
     (E / "metrics_probs_vt-ep3-e3_test_b8.json", S / "vt-ep3-e3_test_b8_with_probs.jsonl")),
    ("ep2 (786432, 2ep)", (E / "metrics_probs_vt-ep2-e2_val_b8.json", S / "vt-ep2-e2_val_b8_with_probs.jsonl"),
     (E / "metrics_probs_vt-ep2-e2_test_b8.json", S / "vt-ep2-e2_test_b8_with_probs.jsonl")),
    ("ep2-native (1386240, 2ep)", (E / "metrics_probs_vt-ep2native_val_b8.json",
                                   S / "vt-ep2native_val_b8_with_probs.jsonl"),
     (E / "metrics_probs_vt-ep2native_test_b8.json", S / "vt-ep2native_test_b8_with_probs.jsonl")),
]


def detect(path):
    if not Path(path).exists():
        return None, None
    recs = C.load_probs_jsonl(str(path))
    gt = sum(1 for r in recs if any(r["gt_findings"][k] for k in FIND))
    det = sum(1 for r in recs if any(r["gt_findings"][k] for k in FIND)
              and any(r["greedy_findings"].get(k, 0) == 1 for k in FIND))
    return det, gt


def nll(path):
    if not Path(path).exists():
        return None
    vals = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pn, gt = d["probs_norm"], d["ground_truth"]
        pr = [pn["breast_density"][gt["breast_density"]],
              pn["breast_birads"][str(gt["breast_birads"])]] + \
             [pn[f"findings.{k}"][str(int(gt["findings"][k]))] for k in FIND]
        vals.append(sum(-math.log(max(p, 1e-12)) for p in pr) / len(pr))
    return sum(vals) / len(vals) if vals else None


def auc_of(recs, key):
    if key == "density":
        return C.ovr_macro_auc([r["gt_density"] for r in recs], [r["p_density"] for r in recs], C.DENSITY_CLASSES)
    if key == "birads":
        return C.ovr_macro_auc([r["gt_birads"] for r in recs],
                               [{i: r["p_birads"][str(i)] for i in C.BIRADS_CLASSES} for r in recs],
                               C.BIRADS_CLASSES)
    if key == "findings":
        v = [C.auc([r["gt_findings"][k] for r in recs], [r["p_findings"][k] for r in recs]) for k in FIND]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None
    if key.startswith("f:"):
        k = key[2:]
        return C.auc([r["gt_findings"][k] for r in recs], [r["p_findings"][k] for r in recs])
    raise KeyError(key)


KEYS = ["density", "birads", "findings"] + [f"f:{k}" for k in FIND]


def paired(recs_a, recs_b, keys=KEYS, n_boot=N_BOOT, seed=SEED):
    A = {r["id"]: r for r in recs_a}
    B = {r["id"]: r for r in recs_b}
    ids = [i for i in A if i in B]
    groups = defaultdict(list)
    for i in ids:
        groups[A[i]["study"]].append(i)
    studies = sorted(groups)
    acc = {k: [] for k in keys}
    rng = random.Random(seed)
    for _ in range(n_boot):
        sid = []
        for _ in range(len(studies)):
            sid.extend(groups[studies[rng.randrange(len(studies))]])
        ra, rb = [A[i] for i in sid], [B[i] for i in sid]
        for k in keys:
            va, vb = auc_of(ra, k), auc_of(rb, k)
            if va is not None and vb is not None:
                acc[k].append(vb - va)
    out = {}
    for k, s in acc.items():
        if len(s) < 100:
            out[k] = (None, None)
        else:
            s.sort()
            out[k] = (s[int(0.025 * len(s))], s[min(int(0.975 * len(s)), len(s) - 1)])
    return out


def f6(v):
    return "—" if v is None else f"{v:.6f}"


def native_stats():
    log = PROJ / "logs" / "vlm" / "native_train.log"
    chain = PROJ / "logs" / "vlm" / "night_chain.log"
    summ = PROJ / "logs" / "vlm" / "native_train_summary.txt"
    peak = None
    ok = None
    if summ.exists():
        for line in open(summ, encoding="utf-8"):
            if line.startswith("NATIVE_TRAIN_OK="):
                ok = line.strip().split("=")[1]
            if line.startswith("NATIVE_PEAK_SMI_MIB="):
                peak = line.strip().split("=")[1]
    sit = []
    if log.exists():
        txt = log.read_text(errors="ignore").replace("\r", "\n")
        sit = [float(x) for x in re.findall(r"([0-9.]+)s/it", txt)]
    med = st.median(sit[-20:]) if sit else None
    restarts = 0
    if chain.exists():
        restarts = len(re.findall(r"attempt \d rc=[1-9]", chain.read_text(errors="ignore")))
    return ok, peak, med, restarts, len(sit)


def main():
    lines = ["# 分辨率 / epoch 数对比（b8 流水线）", ""]
    for split_name, idx in (("验证集 val (1600)", 1), ("测试集 test (2000)", 2)):
        lines += [f"## {split_name}", "",
                  "| model | density AUC | BI-RADS AUC | findings AUROC | mass AUC | calc AUC | asymmetry AUC | "
                  "arch.dist AUC | 病灶检出数 | 取值 NLL |", "|" + "---|" * 10]
        for name, vpair, tpair in MODELS:
            mpath, ppath = vpair if idx == 1 else tpair
            if not Path(mpath).exists():
                lines.append(f"| {name} | (metrics 缺失) " + "| " * 9 + "|")
                continue
            m = json.load(open(mpath, encoding="utf-8"))
            d, g = detect(ppath)
            cells = [f6(m["density_4class"]["ovr_macro_auc"]), f6(m["birads_ovr_macro_auc"]),
                     f6(m["findings_macro_auroc"])]
            for k in FIND:
                cells.append(f6(m["findings"][k]["auroc"]))
            cells.append("—" if d is None else f"{d}/{g}")
            cells.append(f6(nll(ppath)))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---------- 配对 bootstrap（test） ----------
    recs = {}
    for name, _v, t in MODELS:
        if Path(t[1]).exists():
            recs[name] = C.load_probs_jsonl(str(t[1]))
    lines += ["## 配对 bootstrap（study 级, 1000 次, seed 42, 测试集）", "",
              "差值为「后 - 前」，CI 不跨 0 标 `*`。", ""]
    pairs = [("ep2-native (1386240, 2ep)", "ep2 (786432, 2ep)"),
             ("ep2 (786432, 2ep)", "e3 (786432, 3ep)")]
    lines += ["| 比较 | field | ΔAUC | CI_low | CI_high |", "|---|---|---|---|---|"]
    for a, b in pairs:
        if a not in recs or b not in recs:
            lines.append(f"| {a} vs {b} | (数据缺失) | | | |")
            continue
        ci = paired(recs[b], recs[a])
        for k in KEYS:
            va, vb = auc_of(recs[a], k), auc_of(recs[b], k)
            if va is None or vb is None:
                continue
            lo, hi = ci[k]
            star = " *" if (lo is not None and (lo > 0 or hi < 0)) else ""
            lines.append(f"| {a} vs {b} | {k} | {va - vb:+.6f}{star} | {f6(lo)} | {f6(hi)} |")
    lines.append("")

    # ---------- 原生训练信息 ----------
    ok, peak, med, restarts, npts = native_stats()
    lines += ["## 原分辨率训练（ep2-native）", "",
              f"- 配置: `configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml` "
              f"(image_max_pixels=1386240, 2 epoch, save_strategy=steps/100, keep 2)",
              f"- 训练成功标记: `{ok}`；重启次数（非 0 退出）: **{restarts}**",
              f"- nvidia-smi 显存峰值: **{peak} MiB**",
              f"- 单步耗时（最后 {min(20, npts)} 个计时点中位数）: **{f6(med)} s/step**",
              f"- 2 epoch = 878 步，按此步速约 **{(med * 878 / 3600 if med else 0):.2f} h**",
              "- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已启用；失败日志 `logs/vlm/native_fail_<n>.log`",
              ""]

    # ---------- 逐条文件 ----------
    lines += ["## 逐条概率文件（with_probs.jsonl）", "", "| model | split | path |", "|---|---|---|"]
    for name, vpair, tpair in MODELS:
        for split, pair in (("val", vpair), ("test", tpair)):
            p = pair[1]
            rel = p.relative_to(PROJ) if Path(p).is_absolute() else p
            lines.append(f"| {name} | {split} | `{rel}` |")
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
