#!/usr/bin/env python3
"""Evaluate saved predictions on the server; emit aggregate JSON only.

Uses fixed BI-RADS labels 1..5, density labels A..D, four lesion categories,
and greedy descending-IoU category-aware matching at IoU >= 0.3.
Finding BI-RADS and joint F1 reuse the same localization matches.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

CATEGORIES = ['Mass', 'Calcification', 'Asymmetry', 'Architectural Distortion']
TABLE_4B_2TO1 = {
    'json_valid_rate': 99.4,
    'breast_birads_accuracy': 36.2,
    'breast_birads_macro_f1': 24.5,
    'breast_density_accuracy': 78.8,
    'breast_density_macro_f1': 49.4,
    'category_macro_f1': 19.0,
    'localized_f1_iou_0_3': 25.3,
    'finding_birads_accuracy': 42.5,
    'finding_birads_macro_f1': 18.4,
    'joint_finding_f1_iou_0_3': 10.8,
}


def integer_label(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (ValueError, TypeError):
        return None


def ground_truth(row):
    value = row['ground_truth']
    return json.loads(value) if isinstance(value, str) else value


def prediction(row):
    value = row.get('eval_json')
    if value is None:
        value = row.get('prediction_json')
    if isinstance(value, dict) and isinstance(value.get('final_json'), dict):
        value = value['final_json']
    return value if isinstance(value, dict) else None


def macro_f1(pairs, labels):
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in pairs)
        fp = sum(t != label and p == label for t, p in pairs)
        fn = sum(t == label and p != label for t, p in pairs)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0)
    return sum(scores) / len(scores)


def bbox_iou(a, b):
    try:
        a, b = [float(x) for x in a], [float(x) for x in b]
        if len(a) != 4 or len(b) != 4 or not all(math.isfinite(x) for x in a + b):
            return 0
        aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        union = aa + ab - intersection
        return intersection / union if union > 0 else 0
    except (TypeError, ValueError):
        return 0


def match_findings(gt, pred):
    candidates = []
    for g, truth in enumerate(gt):
        for p, guess in enumerate(pred):
            if truth.get('finding_category') != guess.get('finding_category'):
                continue
            score = bbox_iou(truth.get('bbox'), guess.get('bbox'))
            if score >= 0.3:
                candidates.append((score, g, p))
    used_gt, used_pred, pairs = set(), set(), []
    # Negative indices give stable ascending-index tie breaking.
    for _, g, p in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
        if g not in used_gt and p not in used_pred:
            used_gt.add(g)
            used_pred.add(p)
            pairs.append((g, p))
    return pairs


def evaluate(rows):
    assert rows, 'empty evaluation set'
    breast, density, finding_labels = [], [], []
    category_gt, category_pred, category_tp = Counter(), Counter(), Counter()
    valid = errors = gt_total = pred_total = localized_tp = joint_tp = 0
    positive_images = predicted_positive_images = false_positive_images = 0
    for row in rows:
        gt, pred = ground_truth(row), prediction(row)
        valid += pred is not None
        errors += bool(row.get('error'))
        pred = pred or {}
        breast.append((integer_label(gt.get('breast_birads')), integer_label(pred.get('breast_birads'))))
        density.append((gt.get('breast_density'), pred.get('breast_density')))
        gf, pf = gt.get('findings', []), pred.get('findings', [])
        pf = [f for f in pf if isinstance(f, dict)] if isinstance(pf, list) else []
        positive_images += bool(gf)
        predicted_positive_images += bool(pf)
        false_positive_images += bool(pf) and not bool(gf)
        gt_total += len(gf)
        pred_total += len(pf)
        category_gt.update(f.get('finding_category') for f in gf)
        category_pred.update(f.get('finding_category') for f in pf)
        pairs = match_findings(gf, pf)
        localized_tp += len(pairs)
        for g, p in pairs:
            category_tp[gf[g].get('finding_category')] += 1
            truth = integer_label(gf[g].get('finding_birads'))
            guess = integer_label(pf[p].get('finding_birads'))
            joint_tp += truth == guess
            if truth is not None:
                finding_labels.append((truth, guess))
    per_category = {}
    for category in CATEGORIES:
        denominator = category_gt[category] + category_pred[category]
        per_category[category] = {
            'gt': category_gt[category], 'predicted': category_pred[category],
            'tp': category_tp[category],
            'f1': 2 * category_tp[category] / denominator if denominator else 0,
        }
    detection_denominator = gt_total + pred_total
    metrics = {
        'json_valid_rate': valid / len(rows),
        'breast_birads_accuracy': sum(g == p for g, p in breast) / len(rows),
        'breast_birads_macro_f1': macro_f1(breast, range(1, 6)),
        'breast_density_accuracy': sum(g == p for g, p in density) / len(rows),
        'breast_density_macro_f1': macro_f1(density, list('ABCD')),
        'category_macro_f1': sum(x['f1'] for x in per_category.values()) / 4,
        'localized_f1_iou_0_3': 2 * localized_tp / detection_denominator if detection_denominator else 0,
        'finding_birads_accuracy': sum(g == p for g, p in finding_labels) / len(finding_labels) if finding_labels else 0,
        'finding_birads_macro_f1': macro_f1(finding_labels, range(1, 6)),
        'joint_finding_f1_iou_0_3': 2 * joint_tp / detection_denominator if detection_denominator else 0,
    }
    return {'metrics': metrics, 'counts': {
        'rows': len(rows), 'valid_json': valid, 'runtime_errors': errors,
        'gt_findings': gt_total, 'predicted_findings': pred_total,
        'localized_tp': localized_tp, 'joint_tp': joint_tp,
        'finding_birads_matched_support': len(finding_labels),
        'positive_images': positive_images, 'predicted_positive_images': predicted_positive_images,
        'false_positive_images': false_positive_images,
    }, 'per_category': per_category}


def load_jsonl(blob):
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--baseline-git-repo', required=True)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--baseline-path', default='vindr_finetune/outputs/direct_full/predictions.jsonl')
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--verify-table', action='store_true')
    args = parser.parse_args()
    blob = Path(args.predictions).read_bytes()
    rows = load_jsonl(blob)
    assert len(rows) in (500, 4000) and {r['index'] for r in rows} == set(range(len(rows))), 'incomplete or duplicate inference'
    old_blob = subprocess.check_output(['git', '-C', args.baseline_git_repo, 'show', args.baseline_ref + ':' + args.baseline_path])
    old = load_jsonl(old_blob)
    old_ids = {row['index'] for row in old}
    assert len(old_ids) == len(old) == 500 and old_ids == set(range(500)), 'baseline is not the expected 500 cases'
    new_map = {row['index']: row for row in rows}
    data = json.loads(Path(args.test_data).read_text())
    mismatches = sum(ground_truth(row) != json.loads(data[row['index']]['output']) for row in old)
    mismatches += sum(ground_truth(row) != json.loads(data[row['index']]['output']) for row in rows)
    image_mismatches = sum(row.get('image_id') != Path(data[row['index']]['images'][0]).stem for row in rows)
    assert mismatches == image_mismatches == 0, 'ground truth or test image mapping mismatch'
    baseline = evaluate(old)
    new_paired = evaluate([new_map[row['index']] for row in old])
    new_full = evaluate(rows) if len(rows) == 4000 else None
    table_checks = {key: round(baseline['metrics'][key] * 100, 1) == value for key, value in TABLE_4B_2TO1.items()}
    if args.verify_table:
        assert all(table_checks.values()), 'cannot reproduce collaborator table: ' + str(table_checks)
    result = {
        'job_id': 12080,
        'source': {'predictions_sha256': hashlib.sha256(blob).hexdigest(),
                   'baseline_commit': args.baseline_ref,
                   'baseline_blob_sha1': hashlib.sha1(b'blob ' + str(len(old_blob)).encode() + b'\0' + old_blob).hexdigest(),
                   'baseline_path': args.baseline_path},
        'validation': {'ground_truth_mismatches': mismatches, 'image_mapping_mismatches': image_mismatches,
                       'table_metrics_reproduced': sum(table_checks.values()), 'table_metrics_total': len(table_checks)},
        'methods': {'iou_threshold': 0.3, 'matching': 'same category; descending-IoU greedy; ascending-index ties',
                    'joint': 'same localization matches with equal finding BI-RADS',
                    'category_macro_labels': CATEGORIES, 'breast_and_finding_birads_macro_labels': [1, 2, 3, 4, 5],
                    'density_macro_labels': list('ABCD'), 'invalid_predictions': 'counted incorrect; no findings'},
        '4b_2to1_paired500': baseline, '9b_paired500': new_paired, '9b_full4000': new_full,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
