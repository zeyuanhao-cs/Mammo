"""合成数据离线验证 run_probe.py 的探针/报告新代码路径 (不占 GPU, 跑完可删)."""
import sys, json, time, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/zhao/Mammography/Mammo/2-view/scripts/prefix_probe')
import run_probe as RP

root = Path('/home/zhao/Mammography/Mammo/2-view/outputs/prefix_probe/_synthetic')
if root.exists():
    shutil.rmtree(root)
rng = np.random.RandomState(0)

def make_split(split, n):
    (root / 'hidden' / split).mkdir(parents=True)
    rows = []
    for i in range(n):
        sid = f"study{i:05d}_L"
        y = int(rng.choice([1, 2, 3, 4, 5], p=[.45, .45, .05, .04, .01]))
        X = rng.randn(4, 2560).astype(np.float16)
        np.savez(root / 'hidden' / split / f'{sid}.npz',
                 h_last=X[0], h_m8=X[1], h_mid=X[2], img_mean=X[3])
        pn = {'breast_birads': {str(c): float(v) for c, v in
                                zip(range(1, 6), np.exp(rng.randn(5)))}}
        rows.append({'id': sid, 'ground_truth': {'breast_birads': y},
                     'hidden_ok': True, 'greedy_prediction': {}, 'argmax_match': {},
                     'probs': None, 'probs_norm': pn})
    with open(root / f'probs_greedy_{split}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return rows

val = make_split('val', 1200)
test = make_split('test', 1500)
for tag in ('gtfind', 'flipfind'):
    with open(root / f'probs_{tag}_test.jsonl', 'w') as f:
        for r in test:
            r2 = dict(r)
            r2['probs_norm'] = {'breast_birads': {
                str(c): float(v) for c, v in zip(range(1, 6), np.exp(rng.randn(5)))}}
            f.write(json.dumps(r2) + '\n')
RP.REF_SCORED = str(root / 'ref.jsonl')
with open(root / 'ref.jsonl', 'w') as f:
    for r in test:
        f.write(json.dumps({'id': r['id'], 'ground_truth': r['ground_truth'],
                            'probs_norm': r['probs_norm']}) + '\n')

t0 = time.time()
probe = RP.run_probe(root, [])
RP.build_report(root, smoke=False)
print('=== summary_table.md ===')
print((root / 'summary_table.md').read_text())
print('elapsed', round(time.time() - t0, 1), 's')
