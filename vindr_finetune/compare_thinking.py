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


def baseline_subset(rows, baseline, test, require_image_ids=True):
    """Score exactly the baseline indices, preserving any extra output separately."""
    baseline_ids = [r['index'] for r in baseline]
    assert len(baseline_ids) == len(set(baseline_ids)) == 500
    assert set(baseline_ids) == set(range(500)), 'baseline test subset changed'
    by_index = {r['index']: r for r in rows}
    assert len(by_index) == len(rows), 'duplicate prediction indices'
    assert set(baseline_ids) <= set(by_index), 'missing baseline predictions'
    selected = [by_index[i] for i in baseline_ids]
    assert all(ground_truth(r) == json.loads(test[r['index']]['output']) for r in selected), 'ground truth mismatch'
    if require_image_ids:
        assert all(r['image_id'] == Path(test[r['index']]['images'][0]).stem for r in selected), 'image mapping mismatch'
        assert not any(r.get('error') for r in selected), 'runtime inference errors'
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--predictions-9b', required=True, type=Path)
    baseline_input = parser.add_mutually_exclusive_group(required=True)
    baseline_input.add_argument('--baseline-git-repo')
    baseline_input.add_argument('--baseline-predictions', type=Path)
    parser.add_argument('--baseline-ref', default='20e980b591a666d95aa098bdb4709c51a4c3864a')
    parser.add_argument('--test-data', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    baseline_path = 'vindr_finetune/outputs/direct_full/predictions.jsonl'
    if args.baseline_predictions:
        baseline_blob = args.baseline_predictions.read_bytes()
    else:
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
    assert len(test) == 4000 and len(old) == 500
    old_selected = baseline_subset(old, old, test, require_image_ids=False)
    nine_selected = baseline_subset(nine, old, test)
    current_selected = baseline_subset(current, old, test)
    paired = {
        '4b_original_500': evaluate(old_selected),
        '9b_500': evaluate(nine_selected),
        '4b_thinking_500': evaluate(current_selected),
    }
    assert all(round(paired['4b_original_500']['metrics'][k]*100, 1) == v
               for k, v in TABLE_4B_2TO1.items()), 'collaborator table cannot be reproduced'
    result = {'job_id': args.job_id, 'state': 'complete_paired500',
              'validation': {'paired_count': 500, 'ground_truth_mismatches': 0,
                             'image_mapping_mismatches': 0, 'baseline_metrics_reproduced': 10,
                             'baseline_image_mapping': 'frozen dataset indices; baseline has no image-id field',
                             'baseline_indices_sha256': hashlib.sha256(json.dumps(sorted(r['index'] for r in old)).encode()).hexdigest()},
              'source': {'baseline_commit': args.baseline_ref, 'baseline_blob_sha1': sha1,
                         'thinking_predictions_sha256': hashlib.sha256(current_blob).hexdigest(),
                         'nine_predictions_sha256': hashlib.sha256(nine_blob).hexdigest()},
              'paired500': paired,
              'scope': {'test_source_rows': 4000, 'evaluated_rows_each': 500,
                        'nine_output_rows': len(nine), 'thinking_output_rows': len(current),
                        'nine_extra_rows_excluded': len(nine)-500, 'thinking_extra_rows_excluded': len(current)-500},
              'comparison_note': 'Matched test cases and evaluator; thinking has a larger generation budget. '
                                 'All three methods score only the original baseline 500 cases.',
              'thinking_output': {'rows': 500,
                                  'closed_thinking': sum(r.get('thinking_closed', False) for r in current_selected),
                                  'token_limit_reached': sum(r.get('finish_reason') == 'length' for r in current_selected)},
              'evaluator_sha256': hashlib.sha256(Path(__file__).with_name('evaluate_mammo.py').read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    lines = ['# 三方案对比（相同 500 张测试图）', '',
             '| 指标 | 原始 4B 2:1 | 9B 2:1 | 蒸馏思考 4B | 对原始 4B Δ(pp) | 对 9B Δ(pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for key, label in METRIC_NAMES.items():
        a, b, c = [paired[name]['metrics'][key]*100 for name in paired]
        lines.append(f'| {label} | {a:.2f}% | {b:.2f}% | {c:.2f}% | {c-a:+.2f} | {c-b:+.2f} |')
    lines += ['', '三方案均只评测 baseline 的索引 0–499，共 500 张（357 张有病灶、143 张正常）。',
              '训练样本为 4657 条、2948 张不同图片；蒸馏版保留图像、顺序、重复权重和最终标签，增加思考指令与监督。',
              f'已排除额外输出：9B {len(nine)-500} 条，蒸馏4B {len(current)-500} 条。',
              '思考模型使用更大的生成 token 上限；结果同时包含训练方案与推理预算的差异。', '']
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'phase': 'comparison_complete', 'state': result['state'], 'paired_rows': 500,
                      'thinking_rows': 500, 'extra_rows_excluded': len(current)-500}), flush=True)


if __name__ == '__main__':
    main()
