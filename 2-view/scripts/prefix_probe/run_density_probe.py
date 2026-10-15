#!/usr/bin/env python3
"""
密度前缀干预 (teacher-forced) —— os_sub adapter, test 2000 乳房, 只推理。

与 run_probe.py 的 prefix_gt/prefix_flip 同一框架:
  - 图像与 prompt 不变; 响应 JSON 里 breast_density 依次强制为 A/B/C/C/D... A/B/C/D,
    其余字段 (findings/birads) 固定为模型自己 greedy 生成的值
  - 一次 forward 读全部字段概率 (字段序 density -> findings -> birads, 强制 density
    位于 findings 之前, 因此 findings P(1) 是"给定强制密度"下的条件概率)
  - 每个条件一个 probs_forceden{A,B,C,D}_test.jsonl, 格式与 probs_greedy_test.jsonl 相同
  - 输出文件与 run_probe.py 同目录 (outputs/prefix_probe/), 不修改任何已有文件

报告 (每条件 + greedy 基线):
  - 平均 P(任一 finding=1)  (p_any = 1 - Π(1-p1), 用 probs_norm)
  - any-finding AUC (y=任一真值 finding=1, s=p_any)
  - 4 个 findings 各自 AUC (y=真值, s=P(1))
  - 各条件相对 greedy 的 Δ

用法:
  python run_density_probe.py --mode smoke            # 5 乳房 x 4 条件 + gate + 耗时预估
  python run_density_probe.py --mode full --budget-min N
  python run_density_probe.py --report-only           # 只读已有 jsonl 生成报告 (CPU)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/prefix_probe
sys.path.insert(0, str(THIS_DIR))

from run_probe import (                             # noqa: E402
    FINDINGS, SMOKE_N, TEST_DATA, TEST_PREDS, OUT_ROOT, S2V,
    auc_score, load_json_recs, load_jsonl_preds,
)

DENSITIES = ["A", "B", "C", "D"]
GREEDY_FILE = OUT_ROOT / "probs_greedy_test.jsonl"
GATE_TOL = 0.02
BOOT_N = 1000
SEED = 42


# ---------------- 构造强制密度后的 completion ----------------

def force_density(completion: str, dens: str) -> str:
    """把响应 JSON 里的 breast_density 值替换为 dens, 其余不动 (恰好一处)."""
    pat = re.compile(r'("breast_density"\s*:\s*")(A|B|C|D)(")')
    matches = list(pat.finditer(completion))
    assert len(matches) == 1, f"breast_density 出现 {len(matches)} 次: {completion[:120]!r}"
    m = matches[0]
    return completion[:m.start(2)] + dens + completion[m.end(2):]


# ---------------- GPU 阶段: 单条件打分 (断点续跑) ----------------

def run_condition(bundle, instruction, data_recs, order, preds, dens,
                  limit, t_start, budget_min):
    from tqdm.auto import tqdm
    jfile = OUT_ROOT / f"probs_forceden{dens}_test.jsonl"
    done = set()
    if jfile.exists():
        with open(jfile, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("probs") is not None:  # 空概率行不算完成, 需重跑
                            done.add(r["id"])
                    except Exception:
                        pass
    ids = order[:limit] if limit else order
    pending = [i for i in ids if i not in done and i in preds]
    print(f"[forced{dens}] total={len(ids)} done={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        return 0.0, 0.0

    t0 = time.perf_counter()
    with open(jfile, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, desc=f"forced{dens}", unit="case", dynamic_ncols=True)
        for sid in pbar:
            ts = time.perf_counter()
            rec = data_recs[sid]
            pred = preds[sid]
            completion = S2V.extract_json_text(pred.get("prediction_raw") or "")
            probs, probs_norm, argmax_match = None, None, None
            if not completion:
                raise RuntimeError(f"[fatal] {sid}: prediction_raw 中提取不到 completion JSON, 拒绝写入空行")
            try:
                comp2 = force_density(completion, dens)
                probs, probs_norm, argmax_match, _flags = S2V.score_one(
                    *bundle, instruction, rec, comp2)
            except Exception as e:
                raise RuntimeError(f"[fatal] {sid}: 概率提取失败: {e!r}") from e
            if probs is None:
                raise RuntimeError(f"[fatal] {sid}: 概率为空, 拒绝写入空行")
            row = {"id": sid, "ground_truth": pred.get("ground_truth"),
                   "greedy_prediction": pred.get("prediction_json"),
                   "forced_density": dens,
                   "argmax_match": argmax_match, "probs": probs, "probs_norm": probs_norm,
                   "seconds": round(time.perf_counter() - ts, 2)}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.set_postfix(avg=f"{(time.perf_counter()-t0)/max(pbar.n,1):.2f}s")
            left_min = budget_min - (time.perf_counter() - t_start) / 60
            est_min = (time.perf_counter() - t0) / max(pbar.n, 1) * (len(pending) - pbar.n) / 60
            if left_min < est_min + 10:
                print(f"\n[budget] 剩余预算 {left_min:.0f}min < 预估 {est_min:.0f}min+10min, "
                      f"提前停止 (断点续跑)", flush=True)
                break
    el = time.perf_counter() - t0
    print(f"[forced{dens}] new={len(pending)} elapsed={el/60:.1f} min "
          f"({el/len(pending):.2f}s/case)", flush=True)
    return el / max(len(pending), 1), el


def gate_check(limit_ids):
    """两项检查 (vs probs_greedy_test.jsonl, tol=GATE_TOL):
    (a) 前缀不变性: 强制 density 不改变 density 值之前的前缀, density 概率应一致;
    (b) 恒等检查: 强制密度 == 模型自己生成的密度时, completion 与 greedy 完全相同,
        征象/BI-RADS 概率应逐值一致。
    无有效比较样本时判 fail (防止空通过)。
    """
    greedy = {}
    with open(GREEDY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            greedy[r["id"]] = r.get("probs")
    worst_prefix, n_prefix = 0.0, 0
    worst_ident, n_ident = 0.0, 0
    for dens in DENSITIES:
        fpath = OUT_ROOT / f"probs_forceden{dens}_test.jsonl"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] not in limit_ids or not r.get("probs") or r["id"] not in greedy:
                    continue
                g = greedy[r["id"]]
                if not g or not g.get("breast_density"):
                    continue
                # (a) density 概率前缀不变性
                for c in g["breast_density"]:
                    if c in r["probs"]["breast_density"]:
                        worst_prefix = max(worst_prefix, abs(r["probs"]["breast_density"][c]
                                                             - g["breast_density"][c]))
                        n_prefix += 1
                # (b) 恒等检查: forced == greedy 自生成密度
                gp = r.get("greedy_prediction") or {}
                if gp.get("breast_density") == dens:
                    for fld in ("findings", "breast_birads"):
                        for k, v in (g.get(fld) or {}).items():
                            rv = (r["probs"].get(fld) or {}).get(k)
                            if isinstance(v, (int, float)) and isinstance(rv, (int, float)):
                                worst_ident = max(worst_ident, abs(rv - v))
                                n_ident += 1
    ok = (n_prefix > 0 and n_ident > 0
          and worst_prefix < GATE_TOL and worst_ident < GATE_TOL)
    print(f"[gate] (a) density 前缀不变: max_abs_diff={worst_prefix:.4f} n={n_prefix} "
          f"(tol={GATE_TOL})", flush=True)
    print(f"[gate] (b) forced==greedy 恒等: max_abs_diff={worst_ident:.4f} n={n_ident} "
          f"(tol={GATE_TOL})", flush=True)
    print(f"[gate] pass={ok}", flush=True)
    return {"prefix_max": worst_prefix, "prefix_n": n_prefix,
            "ident_max": worst_ident, "ident_n": n_ident, "pass": ok}


# ---------------- 报告 ----------------

def _read_rows(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("probs_norm"):
                    rows[r["id"]] = r
    return rows


def build_report():
    import numpy as np
    files = {"greedy": GREEDY_FILE}
    for d in DENSITIES:
        files[f"forced{d}"] = OUT_ROOT / f"probs_forceden{d}_test.jsonl"
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        print(f"[report] 缺少文件: {missing}, 先跑 GPU 阶段", flush=True)
        return
    tables = {k: _read_rows(p) for k, p in files.items()}
    common = set(tables["greedy"].keys())
    for k in DENSITIES:
        common &= set(tables[f"forced{k}"].keys())
    ids = sorted(common)
    print(f"[report] 共同 id 数: {len(ids)} "
          f"(各文件行数: { {k: len(v) for k, v in tables.items()} })", flush=True)

    gt_rows = {i: tables["greedy"][i]["ground_truth"] for i in ids}
    y_any = np.array([int(any(gt_rows[i]["findings"][fk] for fk in FINDINGS)) for i in ids])
    y_f = {fk: np.array([int(gt_rows[i]["findings"][fk]) for i in ids]) for fk in FINDINGS}

    def metrics(key):
        t = tables[key]
        p1 = {fk: np.array([t[i]["probs_norm"][f"findings.{fk}"]["1"] for i in ids])
              for fk in FINDINGS}
        p_any = np.ones(len(ids))
        for fk in FINDINGS:
            p_any *= (1.0 - p1[fk])
        p_any = 1.0 - p_any
        return {
            "mean_p_any": float(p_any.mean()),
            "any_auc": auc_score(y_any, p_any),
            "finding_auc": {fk: auc_score(y_f[fk], p1[fk]) for fk in FINDINGS},
        }

    rep = {"n": len(ids), "conditions": {}}
    for key in ["greedy"] + [f"forced{d}" for d in DENSITIES]:
        m = metrics(key)
        rep["conditions"][key] = m

    base = rep["conditions"]["greedy"]
    L = []
    L.append("# 密度前缀干预 (teacher-forced, os_sub, test %d 乳房)" % len(ids))
    L.append("")
    L.append("JSON 字段序 density -> findings -> birads; 强制 density 后 findings P(1) "
             "是前缀固定下的条件概率; 图像/prompt/其余字段与 greedy 完全一致。")
    L.append("")
    L.append("| 条件 | 平均P(任一finding=1) | any-AUC | " +
             " | ".join(f"{fk}" for fk in FINDINGS) + " |")
    L.append("|---|---|" + "---|" * (2 + len(FINDINGS) - 1 + 1 - 1))
    for key in ["greedy"] + [f"forced{d}" for d in DENSITIES]:
        m = rep["conditions"][key]
        L.append("| %s | %.4f | %.3f | %s |" % (
            key, m["mean_p_any"], m["any_auc"],
            " | ".join("%.3f" % m["finding_auc"][fk] for fk in FINDINGS)))
    L.append("")
    L.append("## 相对 greedy 的 Δ (AUC)")
    L.append("")
    L.append("| 条件 | Δany-AUC | " + " | ".join(f"Δ{fk}" for fk in FINDINGS) + " |")
    L.append("|---|---|" + "---|" * (1 + len(FINDINGS)))
    for d in DENSITIES:
        m = rep["conditions"][f"forced{d}"]
        L.append("| forced%s | %+.3f | %s |" % (
            d, m["any_auc"] - base["any_auc"],
            " | ".join("%+.3f" % (m["finding_auc"][fk] - base["finding_auc"][fk])
                       for fk in FINDINGS)))
    md = "\n".join(L) + "\n"
    (OUT_ROOT / "density_forcing_report.md").write_text(md, encoding="utf-8")
    json.dump(rep, open(OUT_ROOT / "density_forcing_report.json", "w"),
              ensure_ascii=False, indent=1)
    print(md, flush=True)
    print(f"[report] 已保存 density_forcing_report.md / .json", flush=True)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=False)
    ap.add_argument("--budget-min", type=float, default=125.0,
                    help="本次分配的可用分钟数 (full 模式预算控制)")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="不加载模型, 只读已有 jsonl 生成报告 (CPU)")
    args = ap.parse_args()
    if not args.mode and not args.report_only:
        ap.error("--mode 或 --report-only 必须指定一个")

    if args.report_only:
        build_report()
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    smoke = args.mode == "smoke"
    limit = SMOKE_N if smoke else 0
    budget = 1e9 if smoke else args.budget_min

    instruction = S2V.load_instruction()
    test_recs, test_order = load_json_recs(TEST_DATA)
    test_preds = load_jsonl_preds(TEST_PREDS)
    print(f"[data] test={len(test_order)} preds={len(test_preds)}", flush=True)

    sys.path.insert(0, str(THIS_DIR.parent / "ablation"))
    from run_ablation import load_model
    from run_probe import DEFAULT_ADAPTER
    bundle = load_model(DEFAULT_ADAPTER)

    timing = {}
    for dens in DENSITIES:
        spc, el = run_condition(bundle, instruction, test_recs, test_order, test_preds,
                                dens, limit, t_start, budget)
        timing[f"forced{dens}"] = {"sec_per_case": spc, "elapsed_min": el / 60}

    if smoke:
        gate = gate_check(test_order[:SMOKE_N])
        json.dump(gate, open(OUT_ROOT / "density_gate.json", "w"), indent=1)
        spcs = [timing[f"forced{d}"]["sec_per_case"] for d in DENSITIES]
        est = sum(spcs) / len(spcs) * 4 * len(test_order) / 60
        print(f"\n[SMOKE] sec/case: " +
              " ".join(f"{d}={timing['forced'+d]['sec_per_case']:.2f}" for d in DENSITIES), flush=True)
        print(f"[SMOKE] 全量预估: 4 条件 x {len(test_order)} 条 ≈ {est:.0f} min "
              f"(不含模型加载)", flush=True)
        json.dump({"timing": timing, "est_full_min": est},
                  open(OUT_ROOT / "density_smoke_report.json", "w"),
                  ensure_ascii=False, indent=1)
    else:
        gate_check([json.loads(l)["id"] for l in open(GREEDY_FILE, "r", encoding="utf-8")
                    if l.strip()])

    if not args.skip_report:
        build_report()
    print(f"[total] {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
