#!/usr/bin/env python3
"""
os_sub adapter (Qwen3.5-4B-lora-2view-os-sub) 前缀敏感性 + 隐状态探针 —— 只推理 + 线性探针。

与 score_2view.py / run_ablation.py 完全一致的设置:
  - ChatModel 加载 (template=qwen3_5, enable_thinking=False, sdpa, bf16,
    image_max_pixels=786432, image_min_pixels=512*512, adapter=-2view-os-sub)
  - 输入构建复用 HuggingfaceEngine._process_args 同一路径, teacher-forcing 单次 forward

阶段 (全部支持断点续跑, 按样本落盘):
  hidden_test  test 2000: 贪心前缀 teacher-forcing, 保存 BI-RADS 数字预测位置
               (prompt_len+tok-1, 即概率提取行) 的隐状态 [最后一层/倒数第8层/中间层]
               + 图像 token 最后一层均值; 同时重打概率与 os_sub_test_with_probs.jsonl 对拍
  hidden_val   val 1600: 同上 (探针特征)
  prefix_gt    test 2000: findings 换成真值 (density/birads 保持贪心), 只打概率
  prefix_flip  test 2000: findings 全部取反 (density/birads 保持贪心), 只打概率
  probe        val 隐状态训练 logistic regression (CV 选正则), test 报告 macro-OvR / 3v4 AUC
  report       汇总 summary_table.md + full_report.json

用法:
  python run_probe.py --mode smoke            # 每阶段 5 条 + 对拍 + 耗时预估
  python run_probe.py --mode full --budget-min 125
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/prefix_probe
SCRIPTS_DIR = THIS_DIR.parent                       # 2-view/scripts
PROJ_DIR = SCRIPTS_DIR.parent                       # 2-view
sys.path.insert(0, str(SCRIPTS_DIR))

import score_2view as S2V   # noqa: E402  (复用 load_instruction/extract_json_text/score_one/FIELD_SPECS)

DEFAULT_ADAPTER = "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view-os-sub"
TEST_DATA = "/nesi/project/massey04900/Mammography/test.json"
VAL_DATA = str(PROJ_DIR / "data/val_raw.json")
TEST_PREDS = str(PROJ_DIR / "outputs/predictions/predictions_qwen3.5-4b_os_sub_test_2000.jsonl")
VAL_PREDS = str(PROJ_DIR / "outputs/predictions/predictions_qwen3.5-4b_os_sub_val_1600.jsonl")
REF_SCORED = str(PROJ_DIR / "outputs/scoring/os_sub_test_with_probs.jsonl")
OUT_ROOT = PROJ_DIR / "outputs" / "prefix_probe"

SMOKE_N = 5
SEED = 42
GATE_TOL = 0.02
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
BOOT_N = 1000


# ---------------- data ----------------

def load_json_recs(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {r["id"]: r for r in raw}, [r["id"] for r in raw]


def load_jsonl_preds(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["id"]] = d
    return preds


def build_prefix(completion, mode, gt):
    """gt / flip: 只替换 findings 数字 (0/1), density 与 birads 保持贪心; 返回新文本."""
    assert mode in ("gt", "flip")
    text = completion
    for k in FINDINGS:
        m = re.search(rf'("{k}"\s*:\s*)([01])', text)
        assert m, f"completion 缺少 finding 字段 {k}"
        cur = m.group(2)
        if mode == "gt":
            new = str(int(bool(gt["findings"][k])))
        else:
            new = "1" if cur == "0" else "0"
        text = text[:m.start(2)] + new + text[m.end(2):]
    return text


# ---------------- forward: 概率 + 隐状态 ----------------

def forward_one(bundle, instruction, rec, completion, want_hidden, img_token_id=None):
    """复刻 S2V.score_one 的输入构建与概率提取; 可选返回 BI-RADS 数字预测位置隐状态
    和图像 token 最后一层均值. 返回 (probs, probs_norm, argmax_match, flags, hidden|None)."""
    import torch
    model, tokenizer, processor, template = bundle
    flags = {"multi_token": 0, "argmax_mismatch": 0, "field_missing": 0}
    images = list(rec["images"])

    messages = [{"role": "user", "content": instruction}]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired_messages = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, [], [], tokenizer, processor)
    prompt_len = len(prompt_ids)

    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    completion_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    input_ids = prompt_ids + completion_ids
    ids_t = torch.tensor([input_ids], device=model.device)
    attn_t = torch.ones_like(ids_t)

    mm_inputs = template.mm_plugin.get_mm_inputs(
        images=images, videos=[], audios=[], imglens=[len(images)], vidlens=[0], audlens=[0],
        batch_ids=[prompt_ids], processor=processor,
    )
    mm_kwargs = {}
    for key, value in mm_inputs.items():
        if isinstance(value, list) and isinstance(value[0], torch.Tensor):
            value = torch.stack(value)
        elif isinstance(value, list) and isinstance(value[0], list) and isinstance(value[0][0], torch.Tensor):
            value = torch.stack([torch.stack(v) for v in value])
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if torch.is_floating_point(value):
            value = value.to(model.dtype)
        if key == "second_per_grid_ts":
            mm_kwargs[key] = value.tolist()
        else:
            mm_kwargs[key] = value.to(model.device)

    if hasattr(processor, "create_mm_token_type_ids"):
        mtt = processor.create_mm_token_type_ids([input_ids])
        if not isinstance(mtt, torch.Tensor):
            mtt = torch.tensor(mtt)
        mm_kwargs["mm_token_type_ids"] = mtt.to(model.device)

    K = len(completion_ids) + 1
    with torch.inference_mode():
        try:
            out = model(input_ids=ids_t, attention_mask=attn_t, use_cache=False,
                        logits_to_keep=K, output_hidden_states=want_hidden, **mm_kwargs)
        except TypeError:
            try:
                out = model(input_ids=ids_t, attention_mask=attn_t, use_cache=False,
                            num_logits_to_keep=K, output_hidden_states=want_hidden, **mm_kwargs)
            except TypeError:
                out = model(input_ids=ids_t, attention_mask=attn_t, use_cache=False,
                            output_hidden_states=want_hidden, **mm_kwargs)
        logits = out.logits[0].float()
        hs = out.hidden_states if want_hidden else None

    if logits.shape[0] != len(input_ids):
        assert logits.shape[0] == K
        row_of = lambda j: j
    else:
        row_of = lambda j: prompt_len + j - 1

    probs, probs_norm, argmax_match = {}, {}, {}
    birads_pos = None
    for field, (regex, cands) in S2V.FIELD_SPECS.items():
        m = regex.search(completion)
        if not m:
            flags["field_missing"] += 1
            probs[field] = probs_norm[field] = argmax_match[field] = None
            continue
        value_start = m.start(1)
        tok_idx = S2V.find_token_index(offsets, value_start)
        if tok_idx is None:
            flags["field_missing"] += 1
            probs[field] = probs_norm[field] = argmax_match[field] = None
            continue

        prefix_ids = completion_ids[:tok_idx]
        context_str = completion[: offsets[tok_idx][0]]
        row = logits[row_of(tok_idx)]
        sm = torch.softmax(row, dim=-1)
        if int(row.argmax()) != completion_ids[tok_idx]:
            flags["argmax_mismatch"] += 1
            argmax_match[field] = False
        else:
            argmax_match[field] = True

        if field == "breast_birads":
            birads_pos = prompt_len + tok_idx - 1   # 预测数字的位置 (与概率行一致)

        p = {}
        for c in cands:
            full = tokenizer(context_str + c, add_special_tokens=False)["input_ids"]
            common = S2V.common_prefix_len(full, prefix_ids)
            cand_toks = full[common:]
            if len(cand_toks) != 1:
                flags["multi_token"] += 1
            p[c] = float(sm[cand_toks[0]])
        probs[field] = p
        z = sum(p.values())
        probs_norm[field] = {c: (v / z if z > 0 else 0.0) for c, v in p.items()} if z > 0 else None

    hidden = None
    if want_hidden and hs is not None and birads_pos is not None:
        import torch as _t
        L = len(hs)  # n_layers + 1 (含 embedding)
        if img_token_id is None:
            # 兜底: 图像 token 是 prompt 段中出现次数最多的 id (视觉 token 数远多于文本)
            vals, cnts = ids_t[0][:prompt_len].unique(return_counts=True)
            img_token_id = int(vals[cnts.argmax()])
        h_last = hs[-1][0, birads_pos]
        h_m8 = hs[-8][0, birads_pos]
        h_mid = hs[L // 2][0, birads_pos]
        img_mean = None
        if img_token_id is not None:
            mask = ids_t[0] == img_token_id
            if int(mask.sum()) > 0:
                img_mean = hs[-1][0][mask].mean(dim=0)
        hidden = {
            "h_last": h_last.detach().to(_t.float16).cpu().numpy(),
            "h_m8": h_m8.detach().to(_t.float16).cpu().numpy(),
            "h_mid": h_mid.detach().to(_t.float16).cpu().numpy(),
            "img_mean": (img_mean.detach().to(_t.float16).cpu().numpy()
                         if img_mean is not None else None),
            "birads_pos": birads_pos, "n_layers": L - 1,
        }
    return probs, probs_norm, argmax_match, flags, hidden


# ---------------- GPU 阶段驱动 ----------------

def run_stage_hidden(bundle, instruction, data_recs, order, preds, split, out_root,
                     limit, t_start, budget_min):
    """hidden_test / hidden_val: 逐样本保存 npz + 概率 jsonl (断点续跑)."""
    import numpy as np
    from tqdm.auto import tqdm
    model, tokenizer, _, _ = bundle
    img_token_id = None
    for tok in ["<|image_pad|>", "<|image_token_pad|>", "<image_pad|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            img_token_id = tid
            break

    hdir = out_root / "hidden" / split
    hdir.mkdir(parents=True, exist_ok=True)
    jfile = out_root / f"probs_greedy_{split}.jsonl"
    done = set()
    if jfile.exists():
        with open(jfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("hidden_ok"):
                        done.add(r["id"])
                except Exception:
                    pass
    ids = order[:limit] if limit else order
    pending = [i for i in ids if i not in done and i in preds]
    print(f"[hidden_{split}] total={len(ids)} done={len(done)} pending={len(pending)} "
          f"img_token_id={img_token_id}", flush=True)
    if not pending:
        return 0.0, 0.0

    n_none = 0
    t0 = time.perf_counter()
    with open(jfile, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, desc=f"hidden_{split}", unit="case", dynamic_ncols=True)
        for sid in pbar:
            ts = time.perf_counter()
            rec = data_recs[sid]
            pred = preds[sid]
            completion = S2V.extract_json_text(pred.get("prediction_raw") or "")
            probs, probs_norm, argmax_match, flags, hidden = None, None, None, None, None
            if completion:
                try:
                    probs, probs_norm, argmax_match, flags, hidden = forward_one(
                        bundle, instruction, rec, completion, True, img_token_id)
                except Exception as e:
                    print(f"\n[warn] {sid}: {e!r}", flush=True)
            if hidden is None or hidden.get("h_last") is None or hidden.get("img_mean") is None:
                n_none += 1
            else:
                np.savez(hdir / f"{sid}.npz", **{k: hidden[k] for k in
                        ("h_last", "h_m8", "h_mid", "img_mean")})
            row = {"id": sid, "ground_truth": pred.get("ground_truth"),
                   "greedy_prediction": pred.get("prediction_json"),
                   "argmax_match": argmax_match, "probs": probs, "probs_norm": probs_norm,
                   "hidden_ok": hidden is not None and hidden.get("h_last") is not None
                                and hidden.get("img_mean") is not None,
                   "seconds": round(time.perf_counter() - ts, 2)}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.set_postfix(avg=f"{(time.perf_counter()-t0)/max(pbar.n,1):.2f}s", none=n_none)
            # 预算检查: 剩余样本预估时间 + 10 min 安全余量
            left_min = budget_min - (time.perf_counter() - t_start) / 60
            est_min = (time.perf_counter() - t0) / max(pbar.n, 1) * (len(pending) - pbar.n) / 60
            if left_min < est_min + 10:
                print(f"\n[budget] 剩余预算 {left_min:.0f}min < 预估 {est_min:.0f}min+10min, "
                      f"提前停止 (断点续跑)", flush=True)
                break
    el = time.perf_counter() - t0
    print(f"[hidden_{split}] new={len(pending)} none={n_none} elapsed={el/60:.1f} min "
          f"({el/len(pending):.2f}s/case)", flush=True)
    return el / max(len(pending), 1), el


def run_stage_prefix(bundle, instruction, data_recs, order, preds, mode, out_root,
                     limit, t_start, budget_min):
    """prefix_gt / prefix_flip: 只打概率 (S2V.score_one 原路径), jsonl 断点续跑."""
    from tqdm.auto import tqdm
    jfile = out_root / f"probs_{mode}find_test.jsonl"
    done = set()
    if jfile.exists():
        with open(jfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    ids = order[:limit] if limit else order
    pending = [i for i in ids if i not in done and i in preds]
    print(f"[prefix_{mode}] total={len(ids)} done={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        return 0.0, 0.0

    n_none = 0
    t0 = time.perf_counter()
    with open(jfile, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, desc=f"prefix_{mode}", unit="case", dynamic_ncols=True)
        for sid in pbar:
            ts = time.perf_counter()
            rec = data_recs[sid]
            pred = preds[sid]
            completion = S2V.extract_json_text(pred.get("prediction_raw") or "")
            probs, probs_norm, argmax_match = None, None, None
            if completion:
                try:
                    comp2 = build_prefix(completion, mode, pred["ground_truth"])
                    probs, probs_norm, argmax_match, _flags = S2V.score_one(
                        *bundle, instruction, rec, comp2)
                except Exception as e:
                    print(f"\n[warn] {sid}: {e!r}", flush=True)
            if probs is None:
                n_none += 1
            row = {"id": sid, "ground_truth": pred.get("ground_truth"),
                   "argmax_match": argmax_match, "probs": probs, "probs_norm": probs_norm,
                   "seconds": round(time.perf_counter() - ts, 2)}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.set_postfix(avg=f"{(time.perf_counter()-t0)/max(pbar.n,1):.2f}s", none=n_none)
            left_min = budget_min - (time.perf_counter() - t_start) / 60
            est_min = (time.perf_counter() - t0) / max(pbar.n, 1) * (len(pending) - pbar.n) / 60
            if left_min < est_min + 10:
                print(f"\n[budget] 剩余预算 {left_min:.0f}min < 预估 {est_min:.0f}min+10min, "
                      f"提前停止 (断点续跑)", flush=True)
                break
    el = time.perf_counter() - t0
    print(f"[prefix_{mode}] new={len(pending)} none={n_none} elapsed={el/60:.1f} min "
          f"({el/len(pending):.2f}s/case)", flush=True)
    return el / max(len(pending), 1), el


def gate_check(out_root, ref_scored, limit_ids):
    """对拍: 重打概率 vs os_sub_test_with_probs.jsonl (smoke 差异需 < GATE_TOL)."""
    my = {}
    with open(out_root / "probs_greedy_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            my[r["id"]] = r["probs"]
    ref = {}
    with open(ref_scored, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref[r["id"]] = r["probs"]
    diffs, n_val = [], 0
    for sid in limit_ids:
        if sid not in my or sid not in ref or not my[sid] or not ref[sid]:
            continue
        for field in my[sid]:
            if my[sid][field] and ref[sid].get(field):
                for c in my[sid][field]:
                    if c in ref[sid][field]:
                        diffs.append(abs(my[sid][field][c] - ref[sid][field][c]))
                        n_val += 1
    mx = max(diffs) if diffs else None
    print(f"[gate] 对拍 {len(limit_ids)} id x {n_val} 概率值: max_abs_diff={mx} "
          f"(tol={GATE_TOL}) pass={mx is not None and mx < GATE_TOL}", flush=True)
    return mx, n_val


# ---------------- 指标 ----------------

def auc_score(y, s):
    """rank-based binary AUC (y: 0/1)."""
    import numpy as np
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    # 平局取平均秩
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def birads_metrics(y, p):
    """y: int 1-5 数组; p: [n,5] 概率 (已归一). 返回 dict."""
    import numpy as np
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    out = {}
    classes = [c for c in range(5) if (y == c + 1).sum() >= 10]
    aucs = [auc_score((y == c + 1).astype(int), p[:, c]) for c in classes]
    aucs = [a for a in aucs if a is not None]
    out["macro_ovr"] = float(np.mean(aucs)) if aucs else None
    s_ge3 = p[:, 2] + p[:, 3] + p[:, 4]
    out["ge3"] = auc_score((y >= 3).astype(int), s_ge3)
    s_ge4 = p[:, 3] + p[:, 4]
    out["ge4"] = auc_score((y >= 4).astype(int), s_ge4)
    m34 = np.isin(y, [3, 4])
    if m34.sum() >= 10:
        s34 = (p[m34, 3] + p[m34, 4]) / np.maximum(
            p[m34, 2] + p[m34, 3] + p[m34, 4], 1e-12)
        out["v3v4"] = auc_score((y[m34] == 4).astype(int), s34)
    else:
        out["v3v4"] = None
    pred = p.argmax(axis=1) + 1
    f1s = []
    for c in range(1, 6):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if (y == c).sum() >= 10:
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else None
    return out


def extract_p5(probs_norm_row):
    """从 jsonl 行的 probs_norm 取 5 类 BI-RADS 概率向量."""
    b = (probs_norm_row or {}).get("breast_birads")
    if not b:
        return None
    return [float(b.get(str(c), 0.0)) for c in range(1, 6)]


def study_bootstrap(y, p_by_prefix, ids, n_boot=BOOT_N, seed=SEED):
    """study 级重采样 (同一批重采样应用于所有前缀, 配对).
    返回 (ci, diff_ci): ci[k][m]=(lo,hi) 指标本身; diff_ci[k][m] = k−greedy 的 (lo,hi)."""
    import numpy as np
    studies = {}
    for i, sid in enumerate(ids):
        studies.setdefault(sid.rsplit("_", 1)[0], []).append(i)
    study_list = sorted(studies.keys())
    idx_of = [np.array(studies[s]) for s in study_list]
    rng = np.random.RandomState(seed)
    METS = ["macro_ovr", "ge3", "ge4", "v3v4", "macro_f1"]
    res = {k: {m: [] for m in METS} for k in p_by_prefix}
    y = np.asarray(y)
    P_arr = {k: np.asarray(v) for k, v in p_by_prefix.items()}
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_of), size=len(idx_of))
        idx = np.concatenate([idx_of[j] for j in pick])
        for k, P in P_arr.items():
            m = birads_metrics(y[idx], P[idx])
            for mm in METS:
                if m[mm] is not None:
                    res[k][mm].append(m[mm])
    n_valid = {k: {mm: len(res[k][mm]) for mm in METS} for k in res}

    def pctl(v):
        v = sorted(v)
        if not v:
            return None
        return (v[int(0.025 * len(v))], v[min(int(0.975 * len(v)), len(v) - 1)])

    ci = {k: {mm: (pctl(res[k][mm]) if n_valid[k][mm] >= 0.95 * n_boot else None)
              for mm in METS} for k in res}
    diff_ci = {}
    if "greedy" in res:
        for k in res:
            if k == "greedy":
                continue
            diff_ci[k] = {}
            for mm in METS:
                g, t = res["greedy"][mm], res[k][mm]
                n = min(len(g), len(t))
                if n >= 0.95 * n_boot and n > 0:
                    d = sorted(t[:n][i] - g[:n][i] for i in range(n))
                    diff_ci[k][mm] = (d[int(0.025 * n)], d[min(int(0.975 * n), n - 1)])
                else:
                    diff_ci[k][mm] = None
    return ci, diff_ci


# ---------------- 探针 ----------------

def run_probe(out_root, test_ids):
    """val 隐状态 -> LR (CV log-loss 选 C) -> test 指标; 返回结果 dict."""
    import numpy as np
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import log_loss
    except ImportError:
        print("[probe] sklearn 不可用, 跳过探针", flush=True)
        return {"error": "sklearn unavailable"}

    def feats(split, key):
        X, ids_, ys = [], [], []
        hdir = out_root / "hidden" / split
        with open(out_root / f"probs_greedy_{split}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("hidden_ok"):
                    continue
                fpth = hdir / f"{r['id']}.npz"
                if not fpth.exists():
                    continue
                z = np.load(fpth)
                if z[key].ndim != 1:
                    continue
                X.append(z[key]); ids_.append(r["id"])
                ys.append(int(r["ground_truth"]["breast_birads"]))
        return np.stack(X), ids_, np.asarray(ys)

    def cv_best_C(X, y, is_binary):
        """5 折 CV, 以 log-loss 选 C (越小越好)."""
        Cs = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        counts = [c for c in np.bincount(y) if c > 0]
        n_split = 5 if min(counts) >= 5 else 3
        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=SEED)
        best_c, best_ll = None, 1e18
        for C in Cs:
            lls = []
            for tr, te in skf.split(X, y):
                if len(np.unique(y[tr])) < 2:
                    continue
                clf = Pipeline([("sc", StandardScaler()),
                                ("lr", LogisticRegression(C=C, max_iter=3000,
                                                          random_state=SEED))])
                clf.fit(X[tr], y[tr])
                P = clf.predict_proba(X[te])
                lls.append(log_loss(y[te], P, labels=sorted(np.unique(y))))
            if not lls:
                continue
            m = float(np.mean(lls))
            if m < best_ll:
                best_ll, best_c = m, C
        return best_c, best_ll, n_split

    def boot_auc_ci(y, s, ids, n_boot=BOOT_N, seed=SEED):
        """study 级 bootstrap 的 AUC 95% CI."""
        y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
        studies = {}
        for i, sid in enumerate(ids):
            studies.setdefault(sid.rsplit("_", 1)[0], []).append(i)
        idx_of = [np.array(studies[k]) for k in sorted(studies)]
        rng = np.random.RandomState(seed)
        vals = []
        for _ in range(n_boot):
            pick = rng.randint(0, len(idx_of), size=len(idx_of))
            idx = np.concatenate([idx_of[j] for j in pick])
            a = auc_score(y[idx], s[idx])
            if a is not None:
                vals.append(a)
        if len(vals) < 0.95 * n_boot:
            return None, len(vals)
        vals.sort()
        return (vals[int(0.025 * len(vals))],
                vals[min(int(0.975 * len(vals)), len(vals) - 1)]), len(vals)

    results = {}
    for key, name in [("h_last", "最后一层"), ("h_m8", "倒数第8层"),
                      ("h_mid", "中间层"), ("img_mean", "图像token均值")]:
        Xv, idv, yv = feats("val", key)
        Xt, idt, yt = feats("test", key)
        if len(Xv) < 100 or len(Xt) < 100:
            results[key] = {"name": name, "error": f"样本不足 (val={len(Xv)}, test={len(Xt)})"}
            continue
        # ---- 5 类 LR (CV log-loss 选 C) ----
        best_c, best_ll, n_split = cv_best_C(Xv, yv, False)
        clf = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(C=best_c, max_iter=3000,
                                                  random_state=SEED))])
        clf.fit(Xv, yv)
        P = clf.predict_proba(Xt)
        P5 = np.zeros((len(Xt), 5))
        for j, c in enumerate(clf.classes_):
            P5[:, int(c) - 1] = P[:, j]
        met = birads_metrics(yt, P5)

        # ---- 对照 1: 同特征二分类 BI>=3 ----
        yv_bin = (yv >= 3).astype(int)
        yt_bin = (yt >= 3).astype(int)
        best_c_g, _, _ = cv_best_C(Xv, yv_bin, True)
        clf_g = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=best_c_g, max_iter=3000,
                                                    random_state=SEED))])
        clf_g.fit(Xv, yv_bin)
        s_ge3 = clf_g.predict_proba(Xt)[:, list(clf_g.classes_).index(1)]
        ge3_auc = auc_score(yt_bin, s_ge3)

        # ---- 对照 2: 仅真值 3/4 的 val 样本, PCA64 + LR 二分类, test 3v4 AUC ----
        m34v = np.isin(yv, [3, 4])
        m34t = np.isin(yt, [3, 4])
        v34 = {"n_val34": int(m34v.sum()), "n_test34": int(m34t.sum())}
        if m34v.sum() >= 20 and m34t.sum() >= 10 and len(np.unique(yv[m34v] == 4)) == 2:
            Xv34, yv34 = Xv[m34v], (yv[m34v] == 4).astype(int)
            n_comp = min(64, m34v.sum(), Xv.shape[1])
            pipe34 = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=n_comp)),
                               ("lr", LogisticRegression(max_iter=3000, random_state=SEED))])
            # PCA 后不做 C 搜索 (样本少), 固定 C=1; 3 折 CV log-loss 报告稳定性
            try:
                skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                lls = []
                for tr, te in skf3.split(Xv34, yv34):
                    pipe34.fit(Xv34[tr], yv34[tr])
                    lls.append(log_loss(yv34[te], pipe34.predict_proba(Xv34[te]),
                                        labels=[0, 1]))
                v34["cv_logloss"] = float(np.mean(lls))
            except Exception:
                pass
            pipe34.fit(Xv34, yv34)
            s34 = pipe34.predict_proba(Xt[m34t])[:, 1]
            v34["auc"] = auc_score((yt[m34t] == 4).astype(int), s34)
            ci34, n_ok = boot_auc_ci((yt[m34t] == 4).astype(int), s34,
                                     [i for i, m in zip(idt, m34t) if m])
            v34["auc_ci"] = ci34
            v34["boot_ok"] = n_ok
        else:
            v34["auc"] = None

        results[key] = {"name": name, "best_C": best_c, "cv_logloss": best_ll,
                        "n_cv_folds": n_split, "n_val": len(Xv), "n_test": len(Xt),
                        "metrics": met, "ge3_auc": ge3_auc, "ge3_best_C": best_c_g,
                        "v34_binary": v34,
                        "test_ids": idt, "test_y": yt.tolist(), "test_p": P5.tolist()}
        print(f"[probe] {name}: C={best_c} cv_ll={best_ll:.4f} "
              f"macro_ovr={met['macro_ovr']:.4f} ge3={ge3_auc:.4f} "
              f"v3v4(5类)={met['v3v4']} v3v4(二类)={v34.get('auc')}", flush=True)
    return results


# ---------------- 报告 ----------------

def build_report(out_root, smoke=False):
    import numpy as np
    rep = {"adapter": DEFAULT_ADAPTER, "generated_at": time.strftime("%F %T"),
           "smoke": smoke}

    # 模型自身 (贪心前缀, 官方打分文件)
    ref_ids, ref_y, ref_p = [], [], []
    with open(REF_SCORED, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p5 = extract_p5(r.get("probs_norm"))
            if p5 is None:
                continue
            ref_ids.append(r["id"]); ref_y.append(int(r["ground_truth"]["breast_birads"]))
            ref_p.append(p5)

    id2ref = {sid: j for j, sid in enumerate(ref_ids)}
    y_full = np.asarray(ref_y)
    P_full = np.asarray(ref_p)
    met_greedy = birads_metrics(y_full, P_full)
    ci_greedy, _ = study_bootstrap(y_full.tolist(), {"greedy": P_full.tolist()}, ref_ids)
    ci_greedy = ci_greedy.get("greedy", {})

    # gt / flip 各自与 greedy 取交集分别比较, 各自报 n
    cond = {}
    for tag in ("gtfind", "flipfind"):
        tag_ids, tag_y, tag_p = [], [], []
        with open(out_root / f"probs_{tag}_test.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p5 = extract_p5(r.get("probs_norm"))
                if p5 is None:
                    continue
                tag_ids.append(r["id"])
                tag_y.append(int(r["ground_truth"]["breast_birads"]))
                tag_p.append(p5)
        id2t = {sid: j for j, sid in enumerate(tag_ids)}
        order = [i for i in ref_ids if i in id2t]  # greedy ∩ tag, 按 ref 顺序
        entry = {"n": len(order)}
        if len(order) >= 100:
            y_pair = np.asarray([ref_y[id2ref[i]] for i in order])
            P_g = np.asarray([ref_p[id2ref[i]] for i in order])
            P_t = np.asarray([tag_p[id2t[i]] for i in order])
            entry["metrics"] = birads_metrics(y_pair, P_t)
            entry["greedy_metrics_pair"] = birads_metrics(y_pair, P_g)
            ci_pair, diff_pair = study_bootstrap(
                y_pair.tolist(), {"greedy": P_g.tolist(), tag: P_t.tolist()}, order)
            entry["ci"] = ci_pair.get(tag, {})
            entry["diff_ci"] = diff_pair.get(tag, {})
        else:
            entry["error"] = f"交集样本不足 (n={len(order)})"
        cond[tag] = entry

    rep["prefix_check"] = {
        "n_greedy": len(ref_ids),
        "greedy_metrics": met_greedy,
        "greedy_ci": ci_greedy,
        "per_condition": cond,
        "note": "greedy=模型自身贪心前缀(官方打分); gtfind/flipfind=findings 替换, "
                "density 与 birads 文本保持贪心; BI-RADS 概率为前缀固定下的条件概率; "
                "各条件与 greedy 取各自 id 交集分别比较 (n 各自报告); "
                "Δ=条件−greedy(同交集), CI 为同一批 study 级重采样下的配对差值 95% CI",
    }

    # 探针 (含模型自身对照: 用探针自己的测试集 id, 即 hidden_ok 的 test id)
    if not smoke:
        probe = run_probe(out_root, None)
        rep["probe"] = probe
        hid_ids, hid_y, hid_p = [], [], []
        with open(out_root / "probs_greedy_test.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("hidden_ok"):
                    continue
                sid = r["id"]
                if sid not in id2ref:
                    continue
                hid_ids.append(sid)
                hid_y.append(ref_y[id2ref[sid]])
                hid_p.append(ref_p[id2ref[sid]])
        rep["probe_model_self"] = {
            "n": len(hid_ids),
            "metrics": birads_metrics(np.asarray(hid_y), np.asarray(hid_p))
                       if hid_ids else None,
            "note": "模型自身输出 (官方打分贪心前缀 probs_norm), 限定在探针自己的 "
                    "test id (hidden_ok) 上",
        }

    # gate
    gate = None
    gfile = out_root / "gate.json"
    if gfile.exists():
        gate = json.loads(gfile.read_text(encoding="utf-8"))
    rep["gate"] = gate

    (out_root / "full_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # ---- markdown ----
    L = []
    L.append("# 前缀敏感性 + 隐状态探针 (os_sub adapter, test 2000 乳房)")
    L.append("")
    L.append("说明: BI-RADS 概率均为 teacher-forcing 条件概率 —— findings 前缀固定"
             "(greedy / 真值 / 全取反), density 与 birads 文本保持贪心, "
             "在 BI-RADS 数字位置取 5 类候选 softmax (归一化)。")
    L.append(f"study 级 bootstrap: {BOOT_N} 次, seed={SEED}; 指标只含 test 中 support>=10 的类。")
    L.append("")
    L.append("## 前缀检查 (BI-RADS 指标, Δ 为相对 greedy 前缀的配对差值)")
    L.append("")
    n_gt = rep["prefix_check"]["per_condition"]["gtfind"]["n"]
    n_flip = rep["prefix_check"]["per_condition"]["flipfind"]["n"]
    L.append(f"| 指标 | greedy 前缀 (n={rep['prefix_check']['n_greedy']}) | "
             f"gt findings 前缀 (n={n_gt}) | flip findings 前缀 (n={n_flip}) |")
    L.append("|---|---|---|---|")
    mnames = [("macro_ovr", "BI macro-OvR AUC"), ("ge3", "BI-RADS >=3 AUC"),
              ("ge4", "BI-RADS >=4 AUC"), ("v3v4", "BI-RADS 3v4 AUC"),
              ("macro_f1", "argmax BI Macro-F1")]

    def fmt_greedy(mk):
        v = met_greedy[mk]
        if v is None:
            return "n/a"
        c = ci_greedy.get(mk)
        return f"{v:.4f}" + (f" [{c[0]:.4f},{c[1]:.4f}]" if c else "")

    def fmt_cond(tag, mk):
        e = cond[tag]
        if "metrics" not in e:
            return f"n/a ({e.get('error', '')})"
        v = e["metrics"][mk]
        if v is None:
            return "n/a"
        s = f"{v:.4f}"
        c = e.get("ci", {}).get(mk)
        if c:
            s += f" [{c[0]:.4f},{c[1]:.4f}]"
        vg = e["greedy_metrics_pair"][mk]
        d = e.get("diff_ci", {}).get(mk)
        if v is not None and vg is not None and d is not None:
            s += f" (Δ {v - vg:+.4f} [{d[0]:+.4f},{d[1]:+.4f}])"
        return s

    for mk, mn in mnames:
        L.append(f"| {mn} | {fmt_greedy(mk)} | {fmt_cond('gtfind', mk)} | "
                 f"{fmt_cond('flipfind', mk)} |")
    L.append("")

    if not smoke and "error" not in rep.get("probe", {}):
        L.append("## 探针 (val 隐状态训练 LR, test 报告)")
        L.append("")
        L.append("| 特征 | best C | macro-OvR AUC | BI>=3 AUC (二分类) | 3v4 AUC (5类) | 3v4 AUC (PCA64 二分类, [CI]) |")
        L.append("|---|---|---|---|---|---|")
        for key in ("h_last", "h_m8", "h_mid", "img_mean"):
            r = rep["probe"].get(key, {})
            if "metrics" not in r:
                L.append(f"| {r.get('name', key)} | - | 样本不足 | 样本不足 | 样本不足 | 样本不足 |")
                continue
            m = r["metrics"]
            v34 = r.get("v34_binary", {})
            a34 = v34.get("auc")
            c34 = v34.get("auc_ci")
            s34 = (f"{a34:.4f} [{c34[0]:.4f},{c34[1]:.4f}]" if a34 is not None and c34
                   else ("样本不足" if a34 is None else f"{a34:.4f}"))
            L.append(f"| {r['name']} | {r['best_C']} | {m['macro_ovr']:.4f} | "
                     f"{r['ge3_auc']:.4f} | "
                     f"{m['v3v4'] if m['v3v4'] is not None else float('nan'):.4f} | {s34} |")
        ms = rep["probe_model_self"].get("metrics")
        if ms:
            L.append(f"| 模型自身输出 (对照, n={rep['probe_model_self']['n']}) | - | "
                     f"{ms['macro_ovr']:.4f} | {ms['ge3']:.4f} | "
                     f"{ms['v3v4'] if ms['v3v4'] is not None else float('nan'):.4f} | - |")
        L.append("")

    if gate:
        L.append("## 对拍 (重打概率 vs os_sub_test_with_probs.jsonl)")
        L.append("")
        L.append(f"- max_abs_diff = {gate['max_abs_diff']} (tol {GATE_TOL}), "
                 f"pass = {gate['pass']}, 比较概率值 {gate['n_vals']} 个")
    L.append("")
    (out_root / "summary_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] -> {out_root/'summary_table.md'}, {out_root/'full_report.json'}", flush=True)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=False)
    ap.add_argument("--budget-min", type=float, default=125.0,
                    help="本次分配的可用分钟数 (full 模式预算控制)")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="不加载模型, 只读 outputs/prefix_probe 已有文件跑探针和报告")
    args = ap.parse_args()
    if not args.mode and not args.report_only:
        ap.error("--mode 或 --report-only 必须指定一个")

    if args.report_only:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        build_report(OUT_ROOT, smoke=False)
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    smoke = args.mode == "smoke"
    limit = SMOKE_N if smoke else 0
    budget = 1e9 if smoke else args.budget_min

    instruction = S2V.load_instruction()
    test_recs, test_order = load_json_recs(TEST_DATA)
    val_recs, val_order = load_json_recs(VAL_DATA)
    test_preds = load_jsonl_preds(TEST_PREDS)
    val_preds = load_jsonl_preds(VAL_PREDS)

    # 复用 ablation/run_ablation.py 的模型加载 (与 infer 完全一致)
    sys.path.insert(0, str(SCRIPTS_DIR / "ablation"))
    from run_ablation import load_model
    bundle = load_model(DEFAULT_ADAPTER)

    timing = {}
    # ---- 阶段 1: hidden_test (+对拍) ----
    spc, el = run_stage_hidden(bundle, instruction, test_recs, test_order, test_preds,
                               "test", OUT_ROOT, limit, t_start, budget)
    timing["hidden_test"] = {"sec_per_case": spc, "elapsed_min": el / 60}
    if smoke:
        mx, nv = gate_check(OUT_ROOT, REF_SCORED, test_order[:SMOKE_N])
        json.dump({"max_abs_diff": mx, "n_vals": nv, "pass": mx is not None and mx < GATE_TOL},
                  open(OUT_ROOT / "gate.json", "w"), indent=1)

    # ---- 阶段 2: hidden_val ----
    spc, el = run_stage_hidden(bundle, instruction, val_recs, val_order, val_preds,
                               "val", OUT_ROOT, limit, t_start, budget)
    timing["hidden_val"] = {"sec_per_case": spc, "elapsed_min": el / 60}

    # ---- 阶段 3/4: prefix_gt / prefix_flip (仅 test) ----
    for mode in ("gt", "flip"):
        spc, el = run_stage_prefix(bundle, instruction, test_recs, test_order, test_preds,
                                   mode, OUT_ROOT, limit, t_start, budget)
        timing[f"prefix_{mode}"] = {"sec_per_case": spc, "elapsed_min": el / 60}

    total_gpu = (time.perf_counter() - t_start) / 60
    if smoke:
        n_test, n_val = len(test_order), len(val_order)
        est = (timing["hidden_test"]["sec_per_case"] * (n_test + n_val)
               + timing["prefix_gt"]["sec_per_case"] * 2 * n_test) / 60
        print(f"\n[SMOKE] sec/case: hidden_test={timing['hidden_test']['sec_per_case']:.2f} "
              f"hidden_val={timing['hidden_val']['sec_per_case']:.2f} "
              f"prefix={timing['prefix_gt']['sec_per_case']:.2f}", flush=True)
        print(f"[SMOKE] 全量预估: hidden {n_test+n_val} 条 + prefix {2*n_test} 条 "
              f"≈ {est:.0f} min (不含加载/探针/报告)", flush=True)
        smoke_rep = {"timing": timing, "est_full_min": est, "gate": json.load(
            open(OUT_ROOT / "gate.json")) if (OUT_ROOT / "gate.json").exists() else None}
        json.dump(smoke_rep, open(OUT_ROOT / "smoke_report.json", "w"),
                  ensure_ascii=False, indent=1)
    else:
        # gate (全量对拍)
        with open(OUT_ROOT / "probs_greedy_test.jsonl", "r", encoding="utf-8") as f:
            done_ids = [json.loads(l)["id"] for l in f]
        mx, nv = gate_check(OUT_ROOT, REF_SCORED, done_ids)
        json.dump({"max_abs_diff": mx, "n_vals": nv, "pass": mx is not None and mx < GATE_TOL},
                  open(OUT_ROOT / "gate.json", "w"), indent=1)

    if not args.skip_report:
        build_report(OUT_ROOT, smoke=smoke)
    print(f"[total] {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
