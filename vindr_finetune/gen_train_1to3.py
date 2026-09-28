#!/usr/bin/env python3
# 生成 1:3 (有bbox : 无bbox) 训练集 → data/direct_train_1to3.json
# 阳性: 全部 1411 张有bbox图
# 阴性: 抽 4233 张 (3倍), 从 birads∈{1,2} 的阴性池中按
#       (birads × density × laterality × view) 32层均匀配额抽取(稀缺层封顶), 层内随机
# finding 类别映射与 direct_train.json 一致(细类别→粗5类)
import json, random
from collections import Counter, defaultdict

random.seed(42)

SRC = '/Users/haozeyuan/Desktop/phd/vlm/9_9/vindr-mammo/vlm_dataset/train.json'
REF = '/Users/haozeyuan/Desktop/phd/vlm/9_9/Mammo/vindr_finetune/data/direct_train.json'
OUT = '/Users/haozeyuan/Desktop/phd/vlm/9_9/Mammo/vindr_finetune/data/direct_train_1to3.json'

CAT_MAP = {
    'Mass': 'Mass',
    'Suspicious Calcification': 'Calcification',
    'Asymmetry': 'Asymmetry',
    'Focal Asymmetry': 'Asymmetry',
    'Global Asymmetry': 'Asymmetry',
    'Architectural Distortion': 'Architectural Distortion',
    'Nipple Retraction': 'Associated Feature',
    'Skin Retraction': 'Associated Feature',
    'Skin Thickening': 'Associated Feature',
    'Suspicious Lymph Node': 'Associated Feature',
}

src = json.load(open(SRC))
instruction = json.load(open(REF))[0]['instruction']

def to_entry(r):
    findings = [{
        'finding_category': CAT_MAP[f['finding_category']],
        'finding_birads': f['finding_birads'],
        'bbox': f['bbox'],
    } for f in r['findings'] if f['bbox']]
    output = json.dumps({
        'breast_birads': r['breast_birads'],
        'breast_density': r['breast_density'],
        'findings': findings,
    })
    return {'instruction': instruction, 'output': output, 'images': [r['image_path']]}

pos = [r for r in src if any(f['bbox'] for f in r['findings'])]
# 阴性池: 全部findings无bbox, 且birads∈{1,2}(与instruction规则一致: findings空→birads 1或2)
neg = [r for r in src if not any(f['bbox'] for f in r['findings'])
       and r['breast_birads'] in (1, 2)]
NEED = 3 * len(pos)
print(f"阳性: {len(pos)} | 阴性池: {len(neg)} | 需抽阴性: {NEED}")

# 32层: (birads, density, laterality, view), 层内随机
strata = defaultdict(list)
for r in neg:
    strata[(r['breast_birads'], r['breast_density'], r['laterality'], r['view'])].append(r)
for k in strata:
    random.shuffle(strata[k])

# 轮询均匀配额: 每轮从所有还有余量的层各取1个, 稀缺层自动封顶
pools = {k: v[:] for k, v in strata.items()}
picked = []
remaining = NEED
while remaining > 0:
    progressed = False
    for k in list(pools.keys()):
        if not pools[k]:
            del pools[k]
            continue
        picked.append(pools[k].pop())
        remaining -= 1
        progressed = True
        if remaining == 0:
            break
    if not progressed:
        break
print(f"实际抽到阴性: {len(picked)} (剩余需求: {remaining})")

data = [to_entry(r) for r in pos + picked]
random.shuffle(data)
json.dump(data, open(OUT, 'w'), indent=2, ensure_ascii=False)
print(f"已写入: {OUT} 共 {len(data)} 条 (阳性 {len(pos)} : 阴性 {len(picked)} = 1:{len(picked)/len(pos):.1f})")

# 分布统计
def dist(rows, field):
    return dict(sorted(Counter(r[field] for r in rows).items()))

print("\n-- 阴性样本分布 --")
print("birads:", dist(picked, 'breast_birads'))
print("density:", dist(picked, 'breast_density'))
print("laterality:", dist(picked, 'laterality'))
print("view:", dist(picked, 'view'))
print("\n-- 阳性样本分布 --")
print("birads:", dist(pos, 'breast_birads'))
print("density:", dist(pos, 'breast_density'))
print("laterality:", dist(pos, 'laterality'))
print("view:", dist(pos, 'view'))
fc = Counter(CAT_MAP[f['finding_category']] for r in pos for f in r['findings'] if f['bbox'])
print("finding类别:", dict(sorted(fc.items())))
