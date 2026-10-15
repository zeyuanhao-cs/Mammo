# relation_loss 核心: 值 token 定位 + 去 findings 第二批次 + R1/R2 关系损失
# 不修改 llamafactory 源码; 继承其 SFT trainer 与 data collator。

import json
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from llamafactory.data import SFTDataCollatorWith4DAttentionMask
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
DENSITIES = ["A", "B", "C", "D"]
BIRADS = ["1", "2", "3", "4", "5"]
DENSITY_WEIGHT = {"A": 1.0, "B": 1.0, "C": 2.0, "D": 3.0}  # R2 权重
EPS = 1e-8


# ---------------------------------------------------------------------------
# 响应解析与值 token 定位
# ---------------------------------------------------------------------------

def _parse_response(feature: dict, tokenizer) -> dict:
    """从 tokenized feature 中解析出响应 JSON 文本、真实标签与 token 布局。"""
    input_ids = list(feature["input_ids"])
    labels = list(feature["labels"])
    resp_idx = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
    assert resp_idx, "labels 中没有非 IGNORE 位置, 样本无监督信号"
    resp_start, resp_end = resp_idx[0], resp_idx[-1]
    resp_ids = input_ids[resp_start : resp_end + 1]

    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=False)
    rb = resp_text.rfind("}")
    assert rb != -1, f"响应中找不到 '}}': {resp_text[:80]!r}"
    json_text = resp_text[: rb + 1]
    obj = json.loads(json_text)  # 解析失败直接报错
    assert set(obj.keys()) == {"breast_density", "findings", "breast_birads"}, f"字段异常: {obj.keys()}"
    # 规范化 roundtrip: 确认构建端用的就是 json.dumps 默认格式
    assert json.dumps(obj) == json_text, "json_text 与 json.dumps 规范格式不一致"

    enc_json = tokenizer.encode(json_text, add_special_tokens=False)
    assert enc_json == resp_ids[: len(enc_json)], "encode(json_text) 与实际响应 token 前缀不一致"
    eos_ids = resp_ids[len(enc_json) :]
    return {"resp_start": resp_start, "json_text": json_text, "eos_ids": eos_ids,
            "enc_json": enc_json, "obj": obj}


def _locate_candidates(json_text: str, enc_json: list, tokenizer, pat: re.Pattern, candidates: list):
    """对每个候选值替换后重新分词, 与原分词对比定位值 token。

    用公共前后缀对齐, 允许替换处 token 长度变化; 候选与真值相同时文本不变,
    直接取已定位位置的 token。返回 (pos_in_json, {cand_str: token_id})。"""
    pos, cand_tokens, same_cand = None, {}, None
    for cand in candidates:
        variant = _replace_one(json_text, pat, cand)
        if variant == json_text:
            same_cand = cand  # 与真值相同, 延后处理
            continue
        enc_v = tokenizer.encode(variant, add_special_tokens=False)
        # 公共前缀
        p = 0
        while p < len(enc_json) and p < len(enc_v) and enc_json[p] == enc_v[p]:
            p += 1
        # 公共后缀
        s = 0
        while s < len(enc_json) - p and s < len(enc_v) - p and enc_json[-1 - s] == enc_v[-1 - s]:
            s += 1
        assert len(enc_json) - s - p == 1, (
            f"候选 {cand} 替换后原序列变化区域不是单个 token (prefix={p} suffix={s})")
        assert len(enc_v) - s - p == 1, f"候选 {cand} 替换后新序列变化区域不是单个 token"
        j = p
        if pos is None:
            pos = j
        else:
            assert j == pos, f"候选 {cand} 定位到的位置 {j} 与之前 {pos} 不一致"
        cand_tokens[cand] = enc_v[j]
    if same_cand is not None:
        assert pos is not None, "所有候选均与真值相同, 无法定位"
        cand_tokens[same_cand] = enc_json[pos]
    assert len(set(cand_tokens.values())) == len(cand_tokens), "候选值 token 出现重复"
    return pos, cand_tokens


_PAT_DENSITY = re.compile(r'("breast_density": ")([ABCD])(")')
_PAT_BIRADS = re.compile(r'("breast_birads": )([1-5])')
_PAT_FINDING = {f: re.compile(rf'("{f}": )(0|1)') for f in FINDINGS}


def _replace_one(text: str, pat: re.Pattern, new_val: str) -> str:
    out, n = pat.subn(lambda m: m.group(1) + new_val + (m.group(3) if m.lastindex == 3 else ""), text)
    assert n == 1, f"替换模式匹配次数 {n} != 1"
    return out


def build_relation_meta(feature: dict, tokenizer) -> dict:
    """构建单样本的关系损失元数据, 并完成值 token 正确性 assert。"""
    info = _parse_response(feature, tokenizer)
    obj, json_text, enc_json = info["obj"], info["json_text"], info["enc_json"]
    resp_start = info["resp_start"]
    input_ids, labels = list(feature["input_ids"]), list(feature["labels"])

    # --- 主序列: density / birads / 4 个 findings 的值 token 定位 + assert ---
    pos_den, cand_den = _locate_candidates(json_text, enc_json, tokenizer, _PAT_DENSITY, DENSITIES)
    pos_bi, cand_bi = _locate_candidates(json_text, enc_json, tokenizer, _PAT_BIRADS, BIRADS)
    pos_f, cand_f = {}, {}
    for f in FINDINGS:
        p, c = _locate_candidates(json_text, enc_json, tokenizer, _PAT_FINDING[f], ["0", "1"])
        pos_f[f], cand_f[f] = p, c

    true_den = obj["breast_density"]
    true_bi = str(obj["breast_birads"])
    true_f = {f: str(int(obj["findings"][f])) for f in FINDINGS}

    # assert: 定位到的位置确实是标签里的那个 token (input_ids 与 labels 均验证)
    def _chk(pos_local, tok, name):
        g = resp_start + pos_local
        assert input_ids[g] == tok, f"{name}: input_ids[{g}]={input_ids[g]} != 候选token {tok}"
        assert labels[g] == tok, f"{name}: labels[{g}]={labels[g]} != 候选token {tok} (非监督位置)"
        return g

    g_den = _chk(pos_den, cand_den[true_den], "density")
    g_bi = _chk(pos_bi, cand_bi[true_bi], "birads")
    g_f = {f: _chk(pos_f[f], cand_f[f][true_f[f]], f) for f in FINDINGS}

    # --- 第二批次 (去 findings): density 直接接 birads ---
    nf_obj = {"breast_density": true_den, "breast_birads": obj["breast_birads"]}
    nf_text = json.dumps(nf_obj)
    enc_nf = tokenizer.encode(nf_text, add_special_tokens=False)
    # density 值在两版本中位于相同偏移 (公共前缀)
    pos_nf_den, _ = _locate_candidates(nf_text, enc_nf, tokenizer, _PAT_DENSITY, DENSITIES)
    assert pos_nf_den == pos_den, "去 findings 后 density 值位置发生变化"
    pos_nf_bi, cand_nf_bi = _locate_candidates(nf_text, enc_nf, tokenizer, _PAT_BIRADS, BIRADS)
    assert cand_nf_bi == cand_bi, "去 findings 后 birads 候选 token 改变"

    prompt_ids = input_ids[:resp_start]
    nf_input_ids = prompt_ids + enc_nf + info["eos_ids"]
    nf_labels = [IGNORE_INDEX] * len(nf_input_ids)
    nf_attn = [1] * len(nf_input_ids)
    g_nf_bi = resp_start + pos_nf_bi  # nf 序列中 birads 值 token 的全局位置
    assert nf_input_ids[g_nf_bi] == cand_nf_bi[true_bi], "nf 序列 birads 值 token 校验失败"

    # --- R2: 真值为 1 的 findings 值 token (CE 用 labels 里的目标 token) ---
    r2_pos = [g_f[f] if true_f[f] == "1" else -1 for f in FINDINGS]
    r2_w = [DENSITY_WEIGHT[true_den] if true_f[f] == "1" else 0.0 for f in FINDINGS]

    return {
        "nf_input_ids": nf_input_ids,
        "nf_labels": nf_labels,
        "nf_attn": nf_attn,
        # 主序列
        "f_pos": [g_f[f] for f in FINDINGS],
        "f_cand0": [cand_f[f]["0"] for f in FINDINGS],
        "f_cand1": [cand_f[f]["1"] for f in FINDINGS],
        "g_bi": g_bi,
        # nf 序列
        "nf_pos_bi": g_nf_bi,
        "nf_cands": [cand_bi[b] for b in BIRADS],  # 按类别 "1".."5" 顺序
        "nf_target": cand_bi[true_bi],
        # R2
        "r2_pos": r2_pos,
        "r2_w": r2_w,
        "density_w": DENSITY_WEIGHT[true_den],
        # 冗余校验
        "true_den": true_den, "true_bi": true_bi, "true_f": true_f,
        "g_den": g_den,
    }


# ---------------------------------------------------------------------------
# Collator 包装: 主批次 + (随机 50% 样本的) nf 批次 + _rel 元数据
# ---------------------------------------------------------------------------

@dataclass
class RelationCollator(SFTDataCollatorWith4DAttentionMask):
    r"""在 llamafactory SFT collator 之上输出:
      主批次 (原样) + nf_* 前缀的去 findings 批次 (仅随机 50% 样本) + _rel 元数据张量。
      R1 的 50% 抽样在 collator 内用独立 RNG 决定, (a)/(b) 同 seed 完全一致。"""

    rel_seed: int = 42
    r1_frac: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert self.tokenizer.padding_side == "right", (
            f"padding_side 必须为 right (当前 {self.tokenizer.padding_side})")
        self._rel_rng = torch.Generator().manual_seed(self.rel_seed)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        metas = [build_relation_meta(f, self.tokenizer) for f in features]

        # R1 抽样: 随机 50% 样本生成 nf 批次 (独立 RNG, 不影响 DataLoader 洗牌)
        bsz = len(features)
        sel_mask = torch.rand(bsz, generator=self._rel_rng) < self.r1_frac
        sel_idx = torch.nonzero(sel_mask, as_tuple=False).flatten().tolist()

        # 构造 nf features (仅选中样本; attention_mask 重设为全 1, 长度随 nf 序列)
        nf_features = []
        for i in sel_idx:
            m, f = metas[i], features[i]
            nf = {k: v for k, v in f.items()
                  if k not in ("input_ids", "labels", "attention_mask")}
            nf["input_ids"] = m["nf_input_ids"]
            nf["labels"] = m["nf_labels"]
            nf["attention_mask"] = m["nf_attn"]
            nf_features.append(nf)

        batch = super().__call__(features)          # 主批次 (会 pop images 等)
        out = dict(batch)
        if nf_features:
            nf_batch = super().__call__(nf_features)  # 第二批次 (仅选中样本)
            nf_batch.pop("labels", None)              # nf 前向不计算内部 loss
            out.update({f"nf_{k}": v for k, v in nf_batch.items()})

        rel = {
            "f_pos": torch.tensor([m["f_pos"] for m in metas], dtype=torch.long),        # [B,4]
            "f_cand0": torch.tensor([m["f_cand0"] for m in metas], dtype=torch.long),    # [B,4]
            "f_cand1": torch.tensor([m["f_cand1"] for m in metas], dtype=torch.long),    # [B,4]
            "nf_idx": torch.tensor(sel_idx, dtype=torch.long),                           # [S]
            "nf_pos_bi": torch.tensor([metas[i]["nf_pos_bi"] for i in sel_idx], dtype=torch.long),   # [S]
            "nf_cands": torch.tensor([metas[i]["nf_cands"] for i in sel_idx], dtype=torch.long),     # [S,5]
            "nf_target": torch.tensor([metas[i]["nf_target"] for i in sel_idx], dtype=torch.long),   # [S]
            "r2_pos": torch.tensor([m["r2_pos"] for m in metas], dtype=torch.long),      # [B,4]
            "r2_w": torch.tensor([m["r2_w"] for m in metas], dtype=torch.float),         # [B,4]
        }
        out["_rel"] = rel
        return out


# ---------------------------------------------------------------------------
# Trainer: compute_loss 重写 (LM loss + lambda * (R1 + R2))
# ---------------------------------------------------------------------------

class RelationTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, lambda_rel: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_rel = float(lambda_rel)
        self._rel_acc: dict | None = None  # logging 区间内的小批次累计器

    def _ensure_acc(self) -> dict:
        if self._rel_acc is None:
            self._rel_acc = {"n": 0, "n_r1": 0, "n_r2": 0, "r_lm": 0.0,
                             "r1": 0.0, "r1_ce": 0.0, "r1_kl": 0.0, "r2": 0.0}
        return self._rel_acc

    def flush_rel_stats(self) -> dict:
        r"""返回当前 logging 区间内的小批次平均统计并清零累计器。

        r_lm 对全部小批次平均; r1/r1_ce/r1_kl 只对 r_sel>0 的小批次平均;
        r2 只对有阳性 findings 的小批次平均; r_n/r_n1/r_n2 为总小批次数
        及这两类小批次的个数。"""
        acc, self._rel_acc = self._rel_acc, None
        if not acc or acc["n"] == 0:
            return {}
        out = {"r_lm": acc["r_lm"] / acc["n"], "r_n": acc["n"],
               "r_n1": acc["n_r1"], "r_n2": acc["n_r2"]}
        if acc["n_r1"] > 0:
            out.update({"r1": acc["r1"] / acc["n_r1"],
                        "r1_ce": acc["r1_ce"] / acc["n_r1"],
                        "r1_kl": acc["r1_kl"] / acc["n_r1"]})
        if acc["n_r2"] > 0:
            out["r2"] = acc["r2"] / acc["n_r2"]
        return out

    def _clean_rel_inputs(self, inputs: dict) -> tuple[dict, dict, dict]:
        rel = inputs.pop("_rel", None)
        nf = {k[3:]: inputs.pop(k) for k in list(inputs.keys()) if k.startswith("nf_")}
        return inputs, rel, nf

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        inputs, rel, nf = self._clean_rel_inputs(inputs)
        if rel is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # ---- 主前向: 走标准 compute_loss (labels/num_items_in_batch 语义与 os_sub 一致) ----
        lm_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        if not model.training:
            # eval: 早停只用 LM loss, eval_loss 由 Trainer 自行聚合
            return (lm_loss, outputs) if return_outputs else lm_loss

        # ---- 区间累计: 每个 logging 区间内对小批次累计, on_log 时 flush 出平均 ----
        acc = self._ensure_acc()
        acc["n"] += 1
        acc["r_lm"] += float(lm_loss.detach())

        if self.lambda_rel == 0.0:
            # 模型(a): 只用 LM loss
            return (lm_loss, outputs) if return_outputs else lm_loss

        logits = outputs.logits  # [B, T, V]
        dev = logits.device
        rel = {k: v.to(dev) for k, v in rel.items()}
        nf = {k: v.to(dev) for k, v in nf.items()}
        labels = inputs["labels"]
        sel = rel["nf_idx"]  # 已在 logits 所在设备

        # ---- R1: 选中样本 (collator 内随机 50%), 去 findings 前向 ----
        r1_val, r1_ce_val, r1_kl_val = 0.0, 0.0, 0.0
        if sel.numel() > 0 and nf:
            nf_out = model(**nf)  # 带梯度的第二次前向 (batch 维 = S)
            nf_logits = nf_out.logits
            # P(BI-RADS>=3 | 无 findings): 5 类候选 softmax
            ar = torch.arange(sel.numel(), device=dev)
            pre_nf = (rel["nf_pos_bi"] - 1).clamp(min=0)
            cand_logits = nf_logits[ar, pre_nf, :].gather(1, rel["nf_cands"]).float()  # [S,5]
            cand_p = torch.softmax(cand_logits, dim=-1)
            p_ge3 = cand_p[:, 2:].sum(dim=-1)                    # 类别 3/4/5
            # CE: nf 前向对 BI-RADS 值 token (全词表)
            ce = F.cross_entropy(
                nf_logits[ar, pre_nf, :].float(), rel["nf_target"], reduction="none")  # [S]
            # 第一遍的 P(任一 finding=1): 选中样本的 4 个 finding 位置
            f_pos_sel = rel["f_pos"][sel]                        # [S,4]
            pre_f = (f_pos_sel - 1).clamp(min=0)
            l0 = logits[sel.unsqueeze(1), pre_f, rel["f_cand0"][sel]].float()
            l1 = logits[sel.unsqueeze(1), pre_f, rel["f_cand1"][sel]].float()
            p_f1 = torch.sigmoid(l1 - l0)                        # 2 类候选 softmax
            p_any = 1.0 - torch.prod(1.0 - p_f1, dim=-1)         # [S]
            # 对称二元 KL
            p, q = p_any.clamp(EPS, 1 - EPS), p_ge3.clamp(EPS, 1 - EPS)
            kl = 0.5 * (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
                        + q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p)))
            r1 = (ce + kl).mean()
            r1_val, r1_ce_val, r1_kl_val = float(r1.detach()), float(ce.mean().detach()), float(kl.mean().detach())
            acc["n_r1"] += 1
            acc["r1"] += r1_val
            acc["r1_ce"] += r1_ce_val
            acc["r1_kl"] += r1_kl_val
        else:
            r1 = logits.new_zeros(())

        # ---- R2: 真值为 1 的 findings 值 token CE, 按密度加权 ----
        r2_pos = rel["r2_pos"]                                   # [B,4], -1 为无
        r2_w = rel["r2_w"]                                       # [B,4]
        valid = r2_pos >= 0
        if valid.any():
            bi, bj = torch.nonzero(valid, as_tuple=False).unbind(1)
            pos = r2_pos[bi, bj]
            tgt = labels[bi, pos]                                # 监督目标 token
            pre = (pos - 1).clamp(min=0)
            ce2 = F.cross_entropy(logits[bi, pre, :].float(), tgt, reduction="none")
            r2 = (r2_w[bi, bj] * ce2).mean()                     # 加权后取平均
            r2_val = float(r2.detach())
            acc["n_r2"] += 1
            acc["r2"] += r2_val
        else:
            r2 = logits.new_zeros(())
            r2_val = 0.0

        rel_loss = r1 + r2
        assert torch.isfinite(rel_loss), f"关系损失非有限值: R1={r1_val} R2={r2_val}"

        # ---- 缩放: 若 LM loss 已按 num_items_in_batch 归一 (不再被除以累积步数),
        #      关系损失需补除以梯度累积步数, 保持与 LM loss 同等缩放 ----
        nib = kwargs.get("num_items_in_batch")
        if nib is not None and (self.model_accepts_loss_kwargs or self.compute_loss_func is not None):
            rel_scaled = rel_loss / self.args.gradient_accumulation_steps
        else:
            rel_scaled = rel_loss  # 总 loss 会被 training_step 统一 /累积步数

        loss = lm_loss + self.lambda_rel * rel_scaled
        return (loss, outputs) if return_outputs else loss
