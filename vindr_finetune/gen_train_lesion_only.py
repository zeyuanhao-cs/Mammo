#!/usr/bin/env python3
# 生成 lesion-only 1:3 训练集 → data/train_lesion_only_1to3.json
# 1. bbox finding 只保留 Mass/Calcification/Asymmetry/Architectural Distortion 四类 lesion
# 2. 删除 Associated Feature; 同图内相同 bbox 的重复 finding 只保留一条
# 3. 删完后没有 lesion 的图直接丢弃(不转阴性)
# 4. 阳性全保留 + 从 No Finding 池随机抽 3 倍阴性(池限定 birads∈{1,2}, 与 instruction 规则一致)
import json, random
from collections import Counter

random.seed(42)

SRC = '/Users/haozeyuan/Desktop/phd/vlm/9_9/vindr-mammo/vlm_dataset/train.json'
REF = '/Users/haozeyuan/Desktop/phd/vlm/9_9/Mammo/vindr_finetune/data/direct_train.json'
OUT = '/Users/haozeyuan/Desktop/phd/vlm/9_9/Mammo/vindr_finetune/data/train_lesion_only_1to3.json'

LESION_MAP = {
    'Mass': 'Mass',
    'Suspicious Calcification': 'Calcification',
    'Asymmetry': 'Asymmetry',
    'Focal Asymmetry': 'Asymmetry',
    'Global Asymmetry': 'Asymmetry',
    'Architectural Distortion': 'Architectural Distortion',
}

src = json.load(open(SRC))
instruction = json.load(open(REF))[0]['instruction']
# lesion-only prompt: 删除 Associated Feature 相关内容, finding_birads 只允许 [3, 4, 5]
instruction = (instruction
    .replace('- finding_category: one of ["Mass", "Calcification", "Asymmetry", "Architectural Distortion", "Associated Feature"]',
             '- finding_category: one of ["Mass", "Calcification", "Asymmetry", "Architectural Distortion"]')
    .replace('- finding_birads: one of [3, 4, 5, null]',
             '- finding_birads: one of [3, 4, 5]')
    .replace(', and Associated Feature for secondary signs such as skin/nipple retraction, skin thickening, or suspicious lymph node.',
             '.')
    .replace('- Use finding_birads=null only for Associated Feature; otherwise use one of [3, 4, 5].\n', ''))

pos, stats = [], Counter()
for r in src:
    lesions, seen_bbox = [], set()
    for f in r['findings']:
        if not f['bbox']:
            continue
        if f['finding_category'] not in LESION_MAP:
            stats['removed_associated_feature'] += 1
            continue
        key = tuple(f['bbox'])
        if key in seen_bbox:
            stats['removed_duplicate'] += 1
            continue
        seen_bbox.add(key)
        lesions.append({
            'finding_category': LESION_MAP[f['finding_category']],
            'finding_birads': f['finding_birads'],
            'bbox': f['bbox'],
        })
    if lesions:
        pos.append((r, lesions))
    elif any(f['bbox'] for f in r['findings']):
        stats['dropped_images'] += 1

neg_pool = [r for r in src if not any(f['bbox'] for f in r['findings'])
            and r['breast_birads'] in (1, 2)]
neg = random.sample(neg_pool, 3 * len(pos))

def entry(r, lesions):
    output = json.dumps({
        'breast_birads': r['breast_birads'],
        'breast_density': r['breast_density'],
        'findings': lesions,
    })
    return {'instruction': instruction, 'output': output, 'images': [r['image_path']]}

data = [entry(r, ls) for r, ls in pos] + [entry(r, []) for r in neg]
random.shuffle(data)
json.dump(data, open(OUT, 'w'), indent=2, ensure_ascii=False)

def dist(rows, field):
    return dict(sorted(Counter(r[field] for r in rows).items()))

print(f"阳性: {len(pos)} | 丢弃仅含非lesion的图: {stats['dropped_images']}")
print(f"删除 Associated Feature: {stats['removed_associated_feature']} 条 | 相同bbox去重: {stats['removed_duplicate']} 条")
print(f"阴性: {len(neg)} (池 {len(neg_pool)} 随机抽) | 总计: {len(data)} → {OUT}")
print("\n-- 阳性 lesion 类别 --")
print(dict(sorted(Counter(l['finding_category'] for _, ls in pos for l in ls).items())))
print("\n-- 阳性分布 --")
print("birads:", dist([r for r, _ in pos], 'breast_birads'))
print("density:", dist([r for r, _ in pos], 'breast_density'))
print("\n-- 阴性分布(随机采样) --")
print("birads:", dist(neg, 'breast_birads'))
print("density:", dist(neg, 'breast_density'))
