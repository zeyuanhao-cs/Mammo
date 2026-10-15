# CPU 小测试: 确认 rel_core 可导入、值 token 定位在覆盖各真值组合下正常。
# 用法: 在计算节点上 python test_locate.py (无需 GPU)

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rel_core  # noqa: F401  导入即检查模块级代码
from rel_core import build_relation_meta

from transformers import AutoTokenizer

MODEL = "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print("padding_side:", tok.padding_side)

    def make_feature(den, mass, sc, asym, ad, bi):
        obj = {"breast_density": den, "findings": {"mass": mass, "suspicious_calcification": sc,
                "asymmetry": asym, "architectural_distortion": ad}, "breast_birads": bi}
        jt = json.dumps(obj)
        enc = tok.encode(jt, add_special_tokens=False)
        prompt = tok.encode("describe.", add_special_tokens=False)
        ids = prompt + enc + [tok.eos_token_id]
        labs = [-100] * len(prompt) + enc + [tok.eos_token_id]
        return {"input_ids": ids, "labels": labs}

    # 覆盖: 真值 C/A/D/B、findings 全 0 / 全 1 / 混合、birads 1-5
    cases = [("C", 0, 0, 0, 0, 2), ("A", 0, 0, 0, 0, 1), ("D", 1, 1, 1, 1, 4),
             ("B", 1, 0, 1, 0, 5), ("C", 0, 1, 0, 1, 3)]
    for c in cases:
        feat = make_feature(*c)
        meta = build_relation_meta(feat, tok)
        print(f"den={c[0]} bi={c[5]} findings={c[1:5]} -> OK "
              f"nf_target={meta['nf_target']} r2_pos={meta['r2_pos']} r2_w={meta['r2_w']}")
    print("ALL_LOCATE_OK")


if __name__ == "__main__":
    main()
