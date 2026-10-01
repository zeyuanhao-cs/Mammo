#!/usr/bin/env python3
"""Produce aggregate 4B/9B/thinking comparisons without copying prediction text."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from evaluate_mammo import evaluate, ground_truth, load_jsonl, TABLE_4B_2TO1

METRIC_NAMES = {
    'json_valid_rate': 'JSON 有效率',
    'breast_birads_accuracy': '乳房 BI-RADS Accuracy',
    'breast_birads_macro_f1': '乳房 BI-RADS Macro-F1',
    'breast_density_accuracy': '密度 Accuracy',
    'breast_density_macro_f1': '密度 Macro-F1',
    'category_macro_f1': '病灶类别 Macro-F1',
    'localized_f1_iou_0_3': '定位 F1 @ IoU 0.3',
    'finding_birads_accuracy': '病灶 BI-RADS Accuracy',
    'finding_birads_macro_f1': '病灶 BI-RADS Macro-F1',
    'joint_finding_f1_iou_0_3': '联合病灶 F1 @ IoU 0.3',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--predictions-9b', required=True, type=Path)
    parser.add_argument('--baseline-git-repo', required=True)
    parser.add_argument('--baseline-ref', default='20e980b591a666d95aa098bdb4709c51a4c3864a')
    parser.add_argument('--test-data', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    baseline_path = 'vindr_finetune/outputs/direct_full/predictions.jsonl'
    baseline_blob = subprocess.check_output(['git', '-c', 'safe.directory=' + args.baseline_git_repo,
                                            '-C', args.baseline_git_repo, 'show',
                                            args.baseline_ref + ':' + baseline_path])
    sha1 = hashlib.sha1(b'blob ' + str(len(baseline_blob)).encode() + b'\0' + baseline_blob).hexdigest()
    assert sha1 == 'c0ebb1017f883159866b0ee222242036958439e5', 'baseline version mismatch'
    old = load_jsonl(baseline_blob)
    nine_blob = args.predictions_9b.read_bytes()
    assert hashlib.sha256(nine_blob).hexdigest() == '3f327ddab60e8e0427c393edec29e9c20ddd8fc1183f9300bda2243a84dcc417'
    nine = load_jsonl(nine_blob)
    current_blob = args.predictions.read_bytes()
    current = load_jsonl(current_blob)
    test = json.loads(args.test_data.read_text())
    for rows, count in [(old, 500), (nine, 4000), (current, len(current))]:
        assert count in (500, 4000) and len(rows) == count
        assert {r['index'] for r in rows} == set(range(count)), 'missing or duplicate indices'
        assert all(ground_truth(r) == json.loads(test[r['index']]['output']) for r in rows)
        if rows is not old:
            assert all(r['image_id'] == Path(test[r['index']]['images'][0]).stem for r in rows)
            assert not any(r.get('error') for r in rows), 'runtime inference errors'
    nine_map = {r['index']: r for r in nine}
    current_map = {r['index']: r for r in current}
    paired = {
        '4b_original_500': evaluate(old),
        '9b_500': evaluate([nine_map[i] for i in range(500)]),
        '4b_thinking_500': evaluate([current_map[i] for i in range(500)]),
    }
    assert all(round(paired['4b_original_500']['metrics'][k]*100, 1) == v
               for k, v in TABLE_4B_2TO1.items()), 'collaborator table cannot be reproduced'
    result = {'job_id': args.job_id, 'state': 'complete_paired500',
              'validation': {'paired_count': 500, 'ground_truth_mismatches': 0,
                             'image_mapping_mismatches': 0, 'baseline_metrics_reproduced': 10},
              'source': {'baseline_commit': args.baseline_ref, 'baseline_blob_sha1': sha1,
                         'thinking_predictions_sha256': hashlib.sha256(current_blob).hexdigest(),
                         'nine_predictions_sha256': hashlib.sha256(nine_blob).hexdigest()},
              'paired500': paired, '9b_full4000': evaluate(nine),
              '4b_thinking_full4000': evaluate(current) if len(current) == 4000 else None,
              'comparison_note': 'Matched test cases and evaluator; thinking has a larger generation budget. '
                                 'Original 4B has only 500 predictions, no full-4000 comparison.',
              'thinking_output': {'rows': len(current),
                                  'closed_thinking': sum(r.get('thinking_closed', False) for r in current),
                                  'token_limit_reached': sum(r.get('finish_reason') == 'length' for r in current)},
              'evaluator_sha256': hashlib.sha256(Path(__file__).with_name('evaluate_mammo.py').read_bytes()).hexdigest()}
    if len(current) == 4000:
        result['state'] = 'complete_full4000'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    lines = ['# 三方案对比（相同 500 张测试图）', '',
             '| 指标 | 原始 4B 2:1 | 9B 2:1 | 蒸馏思考 4B | 对原始 4B Δ(pp) | 对 9B Δ(pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for key, label in METRIC_NAMES.items():
        a, b, c = [paired[name]['metrics'][key]*100 for name in paired]
        lines.append(f'| {label} | {a:.2f}% | {b:.2f}% | {c:.2f}% | {c-a:+.2f} | {c-b:+.2f} |')
    lines += ['', '原始 4B 仅有 500 条预测。主表使用完全相同的测试样本与评测实现。',
              '思考模型使用更大的生成 token 上限；结果同时包含训练方案与推理预算的差异。', '']
    if len(current) == 4000:
        lines += ['## 全量 4000 张（9B 与蒸馏思考 4B）', '', '| 指标 | 9B | 蒸馏思考 4B |', '|---|---:|---:|']
        for key, label in METRIC_NAMES.items():
            lines.append(f"| {label} | {result['9b_full4000']['metrics'][key]*100:.2f}% | "
                         f"{result['4b_thinking_full4000']['metrics'][key]*100:.2f}% |")
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'phase': 'comparison_complete', 'state': result['state'], 'paired_rows': 500,
                      'thinking_rows': len(current)}), flush=True)


if __name__ == '__main__':
    main()
