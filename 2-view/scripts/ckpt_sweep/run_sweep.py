#!/usr/bin/env python3
"""
os_sub checkpoint 扫描: 439(gate 对拍) -> 300 -> 400, 只在 val 1600 上推理。

设置与 score_2view.py / run_ablation.py 一致 (load_model 复用 run_ablation, 打分复用
S2V.score_one teacher-forcing)。前缀固定为最终模型(439)在 val 上的贪心 JSON
(predictions_qwen3.5-4b_os_sub_val_1600.jsonl 的 prediction_raw), 三个 checkpoint 共用。

每 checkpoint:
  - BI macro-OvR AUC (只含 val support>=10 的类)
  - 3v4 AUC 两种定义: A=P(4) 原始, B=P(4)/(P(3)+P(4)) 归一 (真值为 3/4 的子集)
  - BI-RADS NLL (probs_norm, 真值类) + 类先验基线 NLL (val 类频率)
  - 439 相对各 checkpoint 的 study 级配对 bootstrap (1000 次, seed=42) Δ

断点续跑: 每 checkpoint 一个 jsonl, 逐行 append, done id 跳过。
用法:
  python run_sweep.py --mode smoke
  python run_sweep.py --mode full --budget-min 60
  python run_sweep.py --mode report-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/ckpt_sweep
SCRIPTS_DIR = THIS_DIR.parent                       # 2-view/scripts
PROJ_DIR = SCRIPTS_DIR.parent                       # 2-view
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "vlm"))  # score_2view 已迁至 scripts/vlm
sys.path.insert(0, str(SCRIPTS_DIR / "ablation"))

import score_2view as S2V                           # noqa: E402
from run_ablation import load_model                 # noqa: E402

ADAPTER_ROOT = "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view-os-sub"
CKPTS = [  # (名字, adapter 路径) —— 按执行顺序
    ("439", ADAPTER_ROOT),                          # 根目录 final = checkpoint-439
    ("300", f"{ADAPTER_ROOT}/checkpoint-300"),
    ("400", f"{ADAPTER_ROOT}/checkpoint-400"),
]
VAL_DATA = str(PROJ_DIR / "data/splits/val_raw.json")
VAL_PREDS = str(PROJ_DIR / "outputs/predictions/predictions_qwen3.5-4b_os_sub_val_1600.jsonl")
REF_SCORED = str(PROJ_DIR / "outputs/scoring/os_sub_val_with_probs.jsonl")
OUT_ROOT = PROJ_DIR / "outputs" / "ckpt_sweep"

SMOKE_N = 5
SEED = 42
BOOT_N = 1000
GATE_TOL = 0.02


def load_val():
    with open(VAL_DATA, "r", encoding="utf-8") as f:
        raw = json.load(f)
    recs = {r["id"]: r for r in raw}
    order = [r["id"] for r in raw]
    preds = {}
    with open(VAL_PREDS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["id"]] = d
    return recs, order, preds


def load_jsonl(path):
    rows = {}
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        rows[d["id"]] = d
                    except Exception:
                        pass
    return rows


# ---------------- GPU 阶段 ----------------

def run_ckpt(name, adapter, recs, order, preds, limit, t_start, budget_min):
    """一个 checkpoint: 固定 439 贪心前缀, teacher-forcing 打概率, jsonl 断点续跑."""
    from tqdm.auto import tqdm
    jfile = OUT_ROOT / f"probs_ckpt{name}.jsonl"
    done = set()
    if jfile.exists():
        with open(jfile, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    ids = order[:limit] if limit else order
    pending = [i for i in ids if i not in done and i in preds]
    print(f"[ckpt{name}] adapter={adapter} total={len(ids)} done={len(done)} "
          f"pending={len(pending)}", flush=True)
    if not pending:
        return

    t_load = time.perf_counter()
    bundle = load_model(adapter)
    instruction = S2V.load_instruction()
    print(f"[ckpt{name}] model loaded {time.perf_counter()-t_load:.0f}s", flush=True)

    n_none = 0
    t0 = time.perf_counter()
    with open(jfile, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, desc=f"ckpt{name}", unit="case", dynamic_ncols=True)
        for sid in pbar:
            ts = time.perf_counter()
            rec = recs[sid]
            pred = preds[sid]
            completion = S2V.extract_json_text(pred.get("prediction_raw") or "")
            probs, probs_norm, argmax_match = None, None, None
            if completion:
                try:
                    probs, probs_norm, argmax_match, _f = S2V.score_one(
                        *bundle, instruction, rec, completion)
                except Exception as e:
                    print(f"\n[warn] {sid}: {e!r}", flush=True)
            if probs is None:
                n_none += 1
            row = {"id": sid, "ground_truth": pred.get("ground_truth"),
                   "argmax_match": argmax_match, "probs": probs, "probs_norm": probs_norm,
                   "seconds": round(time.perf_counter() - ts, 2)}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.set_postfix(avg=f"{(time.perf_counter()-t0)/max(pbar.n,1):.2f}s",
                             none=n_none)
            left_min = budget_min - (time.perf_counter() - t_start) / 60
            est_min = (time.perf_counter() - t0) / max(pbar.n, 1) * (len(pending) - pbar.n) / 60
            if pbar.n >= 10 and left_min < est_min + 3:
                print(f"\n[budget] 剩余 {left_min:.0f}min < 预估 {est_min:.0f}min+8min, "
                      f"提前停止 (断点续跑)", flush=True)
                break
    el = time.perf_counter() - t0
    print(f"[ckpt{name}] new={len(pending)} none={n_none} elapsed={el/60:.1f} min "
          f"({el/max(len(pending),1):.2f}s/case)", flush=True)
    del bundle
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def gate_check(limit_ids=None, tol=0.05, min_ids=50):
    """ckpt439 重打 vs os_sub_val_with_probs: 只对拍 BI-RADS 概率 (拦截);
    findings/density 差异只记录不拦截. min_ids: full 模式要求至少比这么多个 id."""
    mine = load_jsonl(OUT_ROOT / "probs_ckpt439.jsonl")
    ref = load_jsonl(REF_SCORED)
    ids = [i for i in (limit_ids or ref.keys()) if i in mine and i in ref]
    max_birads, n_birads = 0.0, 0
    max_other, n_other = 0.0, 0
    birads_diffs = []
    for i in ids:
        a, b = mine[i].get("probs_norm") or {}, ref[i].get("probs_norm") or {}
        for field in a:
            if not isinstance(a[field], dict) or field not in b or not isinstance(b[field], dict):
                continue
            for c in a[field]:
                if c in b[field] and a[field][c] is not None and b[field][c] is not None:
                    d = abs(float(a[field][c]) - float(b[field][c]))
                    if field == "breast_birads":
                        max_birads = max(max_birads, d)
                        birads_diffs.append(d)
                        n_birads += 1
                    else:
                        max_other = max(max_other, d)
                        n_other += 1
    import numpy as np
    mean_d = float(np.mean(birads_diffs)) if birads_diffs else 0.0
    med_d = float(np.median(birads_diffs)) if birads_diffs else 0.0
    ok = max_birads < tol and len(ids) >= min_ids
    print(f"[gate] ckpt439 vs ref: n_id={len(ids)} (要求>={min_ids}) "
          f"birads: n={n_birads} max_abs_diff={max_birads:.6f} mean={mean_d:.6f} "
          f"median={med_d:.6f} tol={tol} | "
          f"other(findings/density, 不拦截): n={n_other} max={max_other:.6f} pass={ok}", flush=True)
    return {"n_id": len(ids), "tol": tol, "min_ids": min_ids,
            "birads": {"n_values": n_birads, "max_abs_diff": max_birads,
                       "mean_abs_diff": mean_d, "median_abs_diff": med_d},
            "other_fields": {"n_values": n_other, "max_abs_diff": max_other},
            "pass": bool(ok)}


# ---------------- 指标 ----------------

def auc_score(y, s):
    import numpy as np
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = np.argsort(np.argsort(s))
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def one_metrics(rows):
    """rows: {id: row}; 返回单 checkpoint 指标 dict (None 概率的样本剔除)."""
    import numpy as np
    ids, ys, P = [], [], []
    for i, r in rows.items():
        pn = r.get("probs_norm") or {}
        pb = pn.get("breast_birads")
        gt = (r.get("ground_truth") or {}).get("breast_birads")
        if pb and gt is not None:
            ids.append(i)
            ys.append(int(gt))
            P.append([float(pb[str(c)]) for c in range(1, 6)])
    ys = np.asarray(ys)
    P = np.asarray(P)
    n = len(ids)
    out = {"n": n}
    if n == 0:
        return out
    # 类别 support
    supp = {c: int((ys == c).sum()) for c in range(1, 6)}
    out["class_support"] = supp
    # macro-OvR AUC (support>=10)
    aucs = []
    for c in range(1, 6):
        if supp[c] >= 10 and (ys == c).sum() < n:
            a = auc_score((ys == c).astype(int), P[:, c - 1])
            if a is not None:
                aucs.append(a)
    out["macro_ovr"] = float(np.mean(aucs)) if aucs else None
    # 3v4 两种定义
    m34 = ys == 3, ys == 4
    sel = m34[0] | m34[1]
    out["n_3v4"] = int(sel.sum())
    if sel.sum() >= 10:
        y34 = (ys[sel] == 4).astype(int)
        out["v3v4_A"] = auc_score(y34, P[sel, 3])                       # P(4) 原始
        denom = P[sel, 2] + P[sel, 3]
        ok = denom > 0
        out["v3v4_B"] = auc_score(y34[ok], (P[sel, 3] / np.where(ok, denom, 1))[ok]) \
            if ok.sum() >= 10 else None                                  # P4/(P3+P4) 归一
    else:
        out["v3v4_A"] = out["v3v4_B"] = None
    # NLL + 类先验基线
    eps = 1e-12
    ptrue = np.array([P[k, ys[k] - 1] for k in range(n)])
    out["nll"] = float(-np.log(np.clip(ptrue, eps, 1)).mean())
    prior = np.array([supp[c] / n for c in ys])
    out["nll_prior"] = float(-np.log(np.clip(prior, eps, 1)).mean())
    out["_ids"] = ids
    out["_ys"] = ys.tolist()
    out["_P"] = P.tolist()
    return out


def metric_from_arrays(ys, P):
    """bootstrap 内复算 (与 one_metrics 口径一致)."""
    import numpy as np
    ys = np.asarray(ys)
    P = np.asarray(P)
    supp = {c: int((ys == c).sum()) for c in range(1, 6)}
    aucs = []
    for c in range(1, 6):
        if supp[c] >= 10 and (ys == c).sum() < len(ys):
            a = auc_score((ys == c).astype(int), P[:, c - 1])
            if a is not None:
                aucs.append(a)
    res = {"macro_ovr": float(np.mean(aucs)) if aucs else None}
    sel = (ys == 3) | (ys == 4)
    if sel.sum() >= 10:
        y34 = (ys[sel] == 4).astype(int)
        res["v3v4_A"] = auc_score(y34, P[sel, 3])
        denom = P[sel, 2] + P[sel, 3]
        ok = denom > 0
        res["v3v4_B"] = auc_score(y34[ok], (P[sel, 3] / np.where(ok, denom, 1))[ok]) \
            if ok.sum() >= 10 else None
    else:
        res["v3v4_A"] = res["v3v4_B"] = None
    eps = 1e-12
    ptrue = np.array([P[k, ys[k] - 1] for k in range(len(ys))])
    res["nll"] = float(-np.log(np.clip(ptrue, eps, 1)).mean())
    prior = np.array([supp[c] / len(ys) for c in ys])
    res["nll_prior"] = float(-np.log(np.clip(prior, eps, 1)).mean())
    return res


def paired_bootstrap(m439, mck, seed=SEED, n_boot=BOOT_N):
    """study 级配对 bootstrap: 439 - ckpt 各指标 Δ 的 95% CI."""
    import numpy as np
    ids = [i for i in m439["_ids"] if i in set(mck["_ids"])]
    idx439 = {i: k for k, i in enumerate(m439["_ids"])}
    idxck = {i: k for k, i in enumerate(mck["_ids"])}
    ys = np.asarray([m439["_ys"][idx439[i]] for i in ids])
    P4 = np.asarray([m439["_P"][idx439[i]] for i in ids])
    Pc = np.asarray([mck["_P"][idxck[i]] for i in ids])
    study = np.asarray([i.rsplit("_", 1)[0] for i in ids])
    uniq = np.unique(study)
    sidx = {s: np.where(study == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    keys = ["macro_ovr", "v3v4_A", "v3v4_B", "nll"]
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([sidx[s] for s in pick])
        r4 = metric_from_arrays(ys[sel], P4[sel])
        rc = metric_from_arrays(ys[sel], Pc[sel])
        for k in keys:
            if r4[k] is not None and rc[k] is not None:
                boots[k].append(r4[k] - rc[k])
    ci = {}
    for k in keys:
        if boots[k]:
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            ci[k] = [float(lo), float(hi)]
        else:
            ci[k] = None
    return {"n_pair": len(ids), "boot_ci": ci, "n_boot": n_boot, "seed": seed}


# ---------------- 报告 ----------------

def build_report(smoke=False):
    per, pairs = {}, {}
    for name, _ in CKPTS:
        rows = load_jsonl(OUT_ROOT / f"probs_ckpt{name}.jsonl")
        if rows:
            per[name] = one_metrics(rows)
    if "439" in per:
        for name in per:
            if name != "439":
                pairs[name] = paired_bootstrap(per["439"], per[name])

    tag = " (smoke)" if smoke else ""
    L = [f"# os_sub checkpoint 扫描{tag} (val 1600, 前缀=439 贪心 JSON)\n",
         "说明: 三个 checkpoint (439=final / 300 / 400) 共用 439 的贪心预测 JSON 作为",
         "teacher-forcing 前缀, 在 BI-RADS 数字位置取 5 类候选 softmax (归一)。指标只在 val 上计算。",
         "study 级配对 bootstrap: 1000 次, seed=42; Δ = ckpt439 − 对照。\n",
         "## 各 checkpoint 指标\n",
         "| 指标 | ckpt439 | ckpt300 | ckpt400 |",
         "|---|---|---|---|"]
    def fmt(name, key, nd=4):
        v = per.get(name, {}).get(key)
        return "-" if v is None else f"{v:.{nd}f}"
    L.append(f"| n | {fmt('439','n',0)} | {fmt('300','n',0)} | {fmt('400','n',0)} |")
    L.append(f"| BI macro-OvR AUC | {fmt('439','macro_ovr')} | {fmt('300','macro_ovr')} | {fmt('400','macro_ovr')} |")
    L.append(f"| 3v4 AUC · A=P(4) | {fmt('439','v3v4_A')} | {fmt('300','v3v4_A')} | {fmt('400','v3v4_A')} |")
    L.append(f"| 3v4 AUC · B=P4/(P3+P4) | {fmt('439','v3v4_B')} | {fmt('300','v3v4_B')} | {fmt('400','v3v4_B')} |")
    L.append(f"| BI-RADS NLL | {fmt('439','nll')} | {fmt('300','nll')} | {fmt('400','nll')} |")
    L.append(f"| NLL 类先验基线 | {fmt('439','nll_prior')} | {fmt('300','nll_prior')} | {fmt('400','nll_prior')} |")

    if pairs:
        L += ["\n## 439 − checkpoint 配对 Δ (95% bootstrap CI)\n",
              "| 指标 | Δ(439−300) | Δ(439−400) |", "|---|---|---|"]
        for key, label in [("macro_ovr", "BI macro-OvR AUC"), ("v3v4_A", "3v4 AUC · A"),
                           ("v3v4_B", "3v4 AUC · B"), ("nll", "BI-RADS NLL")]:
            cells = []
            for cn in ("300", "400"):
                if cn in pairs and pairs[cn]["boot_ci"].get(key):
                    lo, hi = pairs[cn]["boot_ci"][key]
                    v4 = per["439"].get(key)
                    vc = per[cn].get(key)
                    d = (v4 - vc) if (v4 is not None and vc is not None) else None
                    cells.append(f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]" if d is not None else "-")
                else:
                    cells.append("-")
            L.append(f"| {label} | {cells[0]} | {cells[1]} |")

    # 架构噪声估计: 439 在 g01 (Blackwell) 重打 vs 10月8日 A100 参照文件 (同 439 权重)
    arch = None
    if "439" in per:
        ref_rows = load_jsonl(REF_SCORED)
        if ref_rows:
            m_ref = one_metrics(ref_rows)
            arch = {"note": "ckpt439 相同权重, g01(Blackwell) 重打 vs 10月8日 A100 参照",
                    "g01_rerun": {k: v for k, v in per["439"].items() if not k.startswith("_")},
                    "a100_ref": {k: v for k, v in m_ref.items() if not k.startswith("_")},
                    "diff": {k: (per["439"][k] - m_ref[k]) if per["439"].get(k) is not None
                             and m_ref.get(k) is not None else None
                             for k in ("macro_ovr", "v3v4_A", "v3v4_B", "nll")}}
            L += ["\n## 架构噪声估计 (ckpt439: g01 重打 vs 10月8日 A100 参照)\n",
                  "| 指标 | g01 重打 | A100 参照 | 差值 |", "|---|---|---|---|"]
            for k, lab in [("macro_ovr", "BI macro-OvR AUC"), ("v3v4_A", "3v4 AUC · A"),
                           ("v3v4_B", "3v4 AUC · B"), ("nll", "BI-RADS NLL")]:
                a, b = per["439"].get(k), m_ref.get(k)
                d = (a - b) if (a is not None and b is not None) else None
                L.append(f"| {lab} | {a if a is None else f'{a:.4f}'} | "
                         f"{b if b is None else f'{b:.4f}'} | "
                         f"{d if d is None else f'{d:+.4f}'} |")

    gate = None
    gfile = OUT_ROOT / "gate.json"
    if gfile.exists():
        with open(gfile, "r", encoding="utf-8") as f:
            gate = json.load(f).get("full")
    L += ["\n## 对拍 (ckpt439 重打 vs os_sub_val_with_probs.jsonl, 仅 BI-RADS 拦截)\n"]
    if gate:
        L.append(f"- n_id={gate['n_id']}, BI-RADS 比较值 {gate['birads']['n_values']} 个, "
                 f"max_abs_diff={gate['birads']['max_abs_diff']:.6f}, "
                 f"mean={gate['birads']['mean_abs_diff']:.6f}, "
                 f"median={gate['birads']['median_abs_diff']:.6f} (tol={gate['tol']}), "
                 f"pass={gate['pass']}; findings/density (不拦截): max={gate['other_fields']['max_abs_diff']:.6f}")
    else:
        L.append("- 未执行 (ckpt439 未完成)")

    text = "\n".join(L) + "\n"
    (OUT_ROOT / "summary_table.md").write_text(text, encoding="utf-8")

    rep = {"per_checkpoint": {k: {kk: vv for kk, vv in v.items()
                                  if not kk.startswith("_")} for k, v in per.items()},
           "paired_bootstrap": pairs,
           "arch_noise": arch,
           "gate": gate}
    (OUT_ROOT / "full_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] -> {OUT_ROOT/'summary_table.md'}, {OUT_ROOT/'full_report.json'}", flush=True)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "report-only"], required=True)
    ap.add_argument("--budget-min", type=float, default=1e9)
    ap.add_argument("--gate-first-ids", type=int, default=0,
                    help="full 模式下: 前N条完成后先做一次部分对拍(0=只在439全部完成后)")
    args = ap.parse_args()

    if args.mode == "report-only":
        # gate 重算 (纯 CPU, 只读 jsonl, 不做推理), 补 mean/median 后重新生成报告
        g = gate_check()
        with open(OUT_ROOT / "gate.json", "w", encoding="utf-8") as f:
            json.dump({"full": g}, f, ensure_ascii=False, indent=2)
        build_report()
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    recs, order, preds = load_val()
    limit = SMOKE_N if args.mode == "smoke" else 0
    t_start = time.perf_counter()

    if args.mode == "smoke":
        # 每个checkpoint 5条 + gate 部分对拍
        for name, adapter in CKPTS:
            run_ckpt(name, adapter, recs, order, preds, limit, t_start, args.budget_min)
        g = gate_check(limit_ids=order[:SMOKE_N], min_ids=1)
        with open(OUT_ROOT / "gate.json", "w", encoding="utf-8") as f:
            json.dump({"smoke": g}, f, ensure_ascii=False, indent=2)
        build_report(smoke=True)
        print("[smoke] done", flush=True)
        return

    # full: 顺序 439 -> 300 -> 400, 完成一个落盘一个; 439 后 gate
    for name, adapter in CKPTS:
        run_ckpt(name, adapter, recs, order, preds, limit, t_start, args.budget_min)
        if name == "439":
            rows439 = load_jsonl(OUT_ROOT / "probs_ckpt439.jsonl")
            if len(rows439) < len(order):
                print("[stop] ckpt439 未完成 (预算/中断), 先续跑 439 再跑 300/400", flush=True)
                return
            g = gate_check()
            with open(OUT_ROOT / "gate.json", "w", encoding="utf-8") as f:
                json.dump({"full": g}, f, ensure_ascii=False, indent=2)
            if not g["pass"]:
                print("[warn] gate 未通过, 只记录不拦截, 继续跑后续 checkpoint", flush=True)
    build_report()


if __name__ == "__main__":
    main()
