# 生成 val_llamafactory.json (1600): 与训练文件同格式 (alpaca, 同一 instruction),
# output 用 json.dumps 规范格式 (density, findings, birads 顺序与训练一致)。
# 只新增文件, 不修改任何已有文件。

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# instruction 取训练文件第一条 (所有样本共用同一 instruction)
with open(DATA / "train_llamafactory_os_sub.json", "r", encoding="utf-8") as f:
    train = json.load(f)
instruction = train[0]["instruction"]
assert all(t["instruction"] == instruction for t in train), "训练集 instruction 不一致"

with open(DATA / "val_raw.json", "r", encoding="utf-8") as f:
    val = json.load(f)

out = []
for v in val:
    obj = {
        "breast_density": v["output"]["breast_density"],
        "findings": {k: int(v["output"]["findings"][k])
                     for k in ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]},
        "breast_birads": int(v["output"]["breast_birads"]),
    }
    out.append({"instruction": instruction, "input": "", "output": json.dumps(obj),
                "images": v["images"]})

with open(HERE / "data" / "val_llamafactory.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"val_llamafactory.json: {len(out)} 条, 字段顺序 density/findings/birads")
print("示例 output:", out[0]["output"])
