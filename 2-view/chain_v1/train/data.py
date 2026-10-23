#!/usr/bin/env python3
"""chain_v1 训练数据：prompt + GT JSON，返回 input_ids/labels/位置/图像张量（batch=1）。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

PROJ = Path(__file__).resolve().parent.parent.parent      # chain_v1/train -> 2-view
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
from score_2view import FIELD_SPECS, find_token_index, load_instruction  # noqa: E402

FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
FIND_FIELDS = [f"findings.{k}" for k in FIND]


def gt_text(gt: Dict) -> str:
    return json.dumps({
        "breast_density": gt["breast_density"],
        "findings": {k: int(gt["findings"][k]) for k in FIND},
        "breast_birads": int(gt["breast_birads"]),
    }, ensure_ascii=False)


def positions_in_text(text: str, tokenizer) -> Tuple[List[int], Dict[str, int]]:
    """completion 的 token id 及 6 个字段取值 token 在 completion 内的下标。"""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    pos = {}
    for field, (pat, _c) in FIELD_SPECS.items():
        m = re.search(pat, text)
        pos[field] = find_token_index(offs, m.start(1)) if m else -1
    return ids, pos


class SoftKDDataset(Dataset):
    """读取 llamafactory json（instruction/output string/images）或 splits json（output dict）。"""

    def __init__(self, data_file: str, tokenizer, template, limit: int = 0):
        raw = json.load(open(data_file, encoding="utf-8"))
        if limit:
            raw = raw[:limit]
        self.items = []
        for r in raw:
            out = r["output"]
            gt = json.loads(out) if isinstance(out, str) else out
            instr = r.get("instruction")
            self.items.append({"images": list(r["images"]), "gt": gt, "instruction": instr})
        self.tokenizer = tokenizer
        self.template = template
        self.default_instruction = load_instruction(PROJ / "dataset_build" / "prompt.txt")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        instr = it["instruction"] or self.default_instruction
        text = gt_text(it["gt"])
        comp_ids, pos = positions_in_text(text, self.tokenizer)
        return {
            "instruction": instr,
            "images": it["images"],
            "completion_text": text,
            "comp_ids": comp_ids,
            "finding_pos": [pos[f] for f in FIND_FIELDS],
            "birads_pos": pos["breast_birads"],
            "gt_findings": {k: int(it["gt"]["findings"][k]) for k in FIND},
            "gt_birads": int(it["gt"]["breast_birads"]),
        }


def build_inputs(sample: Dict, tokenizer, template, processor) -> Dict:
    """拼 prompt + completion，返回 input_ids/labels/位置（completion 内下标）。"""
    images = sample["images"]
    messages = [{"role": "user", "content": sample["instruction"]}]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired = messages + [{"role": "assistant", "content": sample["completion_text"]}]
    prompt_ids, resp_ids = template.encode_oneturn(tokenizer, paired, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, [], [], tokenizer, processor)
    assert resp_ids, "response 编码为空"

    input_ids = list(prompt_ids) + list(resp_ids)
    labels = [-100] * len(prompt_ids) + list(resp_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "images": images,
        "finding_pos": sample["finding_pos"],
        "birads_pos": sample["birads_pos"],
        "gt_findings": sample["gt_findings"],
        "gt_birads": sample["gt_birads"],
    }


def to_tensors(built: Dict, processor, template, model=None, dtype=None) -> Dict:
    """转 tensor + 视觉输入（batch=1）。

    model=None 时全部留在 CPU（交给 Trainer 搬设备），浮点按 dtype 转；
    传 model 时直接在 model.device 上并按 model.dtype 转（推理/调试用）。
    """
    ids = torch.tensor([built["input_ids"]], dtype=torch.long)
    labels = torch.tensor([built["labels"]], dtype=torch.long)
    attn = torch.ones_like(ids)
    mm = template.mm_plugin.get_mm_inputs(
        images=built["images"], videos=[], audios=[], imglens=[len(built["images"])],
        vidlens=[0], audlens=[0], batch_ids=[built["input_ids"]], processor=processor)
    kw = {}
    for k, v in mm.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
        elif isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], torch.Tensor):
            v = torch.stack([torch.stack(x) for x in v])
        elif not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        if torch.is_floating_point(v):
            if model is not None:
                v = v.to(model.dtype)
            elif dtype is not None:
                v = v.to(dtype)
        if k == "second_per_grid_ts":
            kw[k] = v.tolist()
        elif model is not None:
            kw[k] = v.to(model.device)
        else:
            kw[k] = v
    if hasattr(processor, "create_mm_token_type_ids"):
        mtt = processor.create_mm_token_type_ids([built["input_ids"]])
        if not isinstance(mtt, torch.Tensor):
            mtt = torch.tensor(mtt)
        kw["mm_token_type_ids"] = mtt.to(model.device) if model is not None else mtt

    return {
        "input_ids": ids.to(model.device) if model is not None else ids,
        "attention_mask": attn.to(model.device) if model is not None else attn,
        "labels": labels.to(model.device) if model is not None else labels,
        "finding_pos": torch.tensor([built["finding_pos"]], dtype=torch.long,
                                    device=model.device if model is not None else None),
        "birads_pos": torch.tensor([built["birads_pos"]], dtype=torch.long,
                                   device=model.device if model is not None else None),
        "mm_kwargs": kw,
        "gt_findings": built["gt_findings"],
        "gt_birads": built["gt_birads"],
    }
