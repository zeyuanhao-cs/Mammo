#!/usr/bin/env python3
"""Server-side aggregate evaluation of paired 4B training/inference CoT arms.

Prediction paths: RUN/seed42/{A,B,C,D}/predictions.jsonl. Raw rows never
appear in output. Optional groups JSON must contain unit (patient or study),
source, and index_to_group mapping every test index 0..499 to its group ID.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

from evaluate_mammo import evaluate, ground_truth, integer_label, prediction
from compare_thinking import METRIC_NAMES

TEST_SHA256 = '6eddbabedff8e37df2f5629585b4947fea3cbea6df9f61d9a1fed13548b8249e'
ARMS = {'A': '仅 JSON 训练 / 关闭思考', 'B': '仅 JSON 训练 / 开启思考',
        'C': 'CoT 训练 / 关闭思考', 'D': 'CoT 训练 / 开启思考'}
EFFECTS = {'D-A': ('D', 'A'), 'B-A': ('B', 'A'),
           'D-C': ('D', 'C'), 'C-A': ('C', 'A')}


class ValidationError(ValueError):
    """Closed codes only; never include row text in exceptions."""


def load_test(path):
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != TEST_SHA256:
        raise ValidationError('TEST_HASH_MISMATCH')
    data = json.loads(blob)
    if not isinstance(data, list) or len(data) != 4000:
        raise ValidationError('TEST_SHAPE_MISMATCH')
    return data


def validate_rows(rows, test, expected=500):
    if len(rows) != expected:
        raise ValidationError('ROW_COUNT_MISMATCH')
    if any(not isinstance(r, dict) or type(r.get('index')) is not int for r in rows):
        raise ValidationError('INVALID_INDEX')
    indices = [r['index'] for r in rows]
    if len(set(indices)) != len(indices):
        raise ValidationError('DUPLICATE_INDEX')
    if set(indices) != set(range(expected)):
        raise ValidationError('INDEX_SET_MISMATCH')
    rows = sorted(rows, key=lambda row: row['index'])
    for row in rows:
        target = test[row['index']]
        try:
            truth = target['output']
            truth = json.loads(truth) if isinstance(truth, str) else truth
            if ground_truth(row) != truth:
                raise ValidationError('GROUND_TRUTH_MISMATCH')
            if row.get('image_id') != Path(target['images'][0]).stem:
                raise ValidationError('IMAGE_MAPPING_MISMATCH')
        except (KeyError, TypeError, json.JSONDecodeError):
            raise ValidationError('ROW_SCHEMA_INVALID') from None
    return rows


def scoring_rows(rows):
    # Runtime failures stay in the denominator even if a stale parsed object
    # happens to be present on their records.
    return [dict(r, eval_json=None, prediction_json=None) if r.get('error') else r for r in rows]


def numeric_summary(values):
    values = [v for v in values if isinstance(v, (int, float))
              and not isinstance(v, bool) and math.isfinite(v) and v >= 0]
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'sum': sum(values) if values else None}


def class_counts(rows, key, labels):
    truth, predicted, correct = Counter(), Counter(), Counter()
    for row in rows:
        gt, pred = ground_truth(row), prediction(row) or {}
        t, p = gt.get(key), pred.get(key)
        if key == 'breast_birads':
            t, p = integer_label(t), integer_label(p)
        truth[t] += 1
        predicted[p] += 1
        if t == p:
            correct[t] += 1
    return {str(label): {'support': truth[label], 'predicted': predicted[label],
                         'correct': correct[label], 'missed': truth[label] - correct[label],
                         'recall': correct[label] / truth[label] if truth[label] else None}
            for label in labels}


def summarize_cell(rows):
    scored = scoring_rows(rows)
    result = evaluate(scored)
    total = len(rows)
    normal = total - result['counts']['positive_images']
    detected = sum(bool(ground_truth(r).get('findings')) and
                   bool((prediction(r) or {}).get('findings')) for r in scored)
    positive = result['counts']['positive_images']
    result['quality'] = {
        'invalid_json_rate': 1 - result['metrics']['json_valid_rate'],
        'runtime_error_rate': result['counts']['runtime_errors'] / total,
        'truncated': sum(r.get('finish_reason') == 'length' for r in rows),
        'finish_reason_known': sum(r.get('finish_reason') is not None for r in rows),
        'normal_images': normal,
        'normal_false_positive_rate': result['counts']['false_positive_images'] / normal if normal else None,
        'positive_image_detection_rate': detected / positive if positive else None,
        'missed_positive_images': positive - detected,
    }
    # Missing generation metadata is unknown, never silently a zero rate.
    result['quality']['truncation_rate'] = (result['quality']['truncated'] / total
        if result['quality']['finish_reason_known'] == total else None)
    result['cost'] = {'output_tokens': numeric_summary([
        r.get('generated_tokens') if r.get('generated_tokens') is not None else r.get('output_tokens')
        for r in rows]),
                      'seconds': numeric_summary([r.get('seconds') for r in rows])}
    result['per_breast_birads'] = class_counts(scored, 'breast_birads', range(1, 6))
    result['per_density'] = class_counts(scored, 'breast_density', 'ABCD')
    for category in result['per_category'].values():
        category['false_negative'] = category['gt'] - category['tp']
        category['false_positive'] = category['predicted'] - category['tp']
        category['recall'] = category['tp'] / category['gt'] if category['gt'] else None
    return result


def mean_std(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'std': statistics.stdev(values) if len(values) > 1 else None}


def load_groups(path):
    obj = json.loads(path.read_text())
    if obj.get('unit') not in ('patient', 'study') or not obj.get('source'):
        raise ValidationError('GROUP_PROVENANCE_REQUIRED')
    mapping = obj.get('index_to_group', {})
    if set(mapping) != {str(i) for i in range(500)}:
        raise ValidationError('GROUP_INDEX_SET_MISMATCH')
    groups = defaultdict(list)
    for i in range(500):
        value = mapping[str(i)]
        if not isinstance(value, (str, int)) or isinstance(value, bool) or value == '':
            raise ValidationError('INVALID_GROUP_ID')
        groups[str(value)].append(i)
    if len(groups) < 2:
        raise ValidationError('INSUFFICIENT_GROUPS')
    return list(groups.values()), {'unit': obj['unit'], 'groups': len(groups),
                                   'mapping_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def bootstrap(rows, seeds, groups, replicates, random_seed=271828):
    """Crossed paired resampling: same seeds and groups for all four arms.

    Seeds and test clusters are independent crossed sampling dimensions, not
    nested image observations. A patient/study cluster always stays intact.
    """
    rng = random.Random(random_seed)
    distribution = {e: {m: [] for m in METRIC_NAMES} for e in EFFECTS}
    clean = {s: {a: scoring_rows(rows[s][a]) for a in ARMS} for s in seeds}
    for _ in range(replicates):
        seed_sample = rng.choices(seeds, k=len(seeds))
        indices = [i for group in rng.choices(groups, k=len(groups)) for i in group]
        cache = {s: {a: evaluate([clean[s][a][i] for i in indices])['metrics']
                     for a in ARMS} for s in set(seed_sample)}
        for effect, (high, low) in EFFECTS.items():
            for metric in METRIC_NAMES:
                distribution[effect][metric].append(statistics.mean(
                    cache[s][high][metric] - cache[s][low][metric] for s in seed_sample))
    return {e: {m: [percentile(v, .025), percentile(v, .975)] for m, v in d.items()}
            for e, d in distribution.items()}


def collect(run_dir, test, seeds):
    cells, raw = {}, {}
    for seed in seeds:
        cells[str(seed)], raw[seed] = {}, {}
        for arm in ARMS:
            path = run_dir / f'seed{seed}' / arm / 'predictions.jsonl'
            cell = {'state': 'pending'}
            if path.exists():
                blob = path.read_bytes()
                cell['sha256'] = hashlib.sha256(blob).hexdigest()
                try:
                    rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
                    cell['observed_rows'] = len(rows)
                    rows = validate_rows(rows, test)
                    cell.update(state='complete', **summarize_cell(rows))
                    raw[seed][arm] = rows
                except ValidationError as exc:
                    cell.update(state='pending' if str(exc) == 'ROW_COUNT_MISMATCH'
                                and cell.get('observed_rows', 501) < 500 else 'invalid',
                                error_code=str(exc))
                except (ValueError, TypeError, KeyError, AttributeError):
                    cell.update(state='invalid', error_code='PREDICTION_SCHEMA_INVALID')
            cells[str(seed)][arm] = cell
    return cells, raw


def assemble(run_dir, test, seeds, groups_path=None, replicates=500):
    cells, rows = collect(run_dir, test, seeds)
    count = sum(c['state'] == 'complete' for arms in cells.values() for c in arms.values())
    complete = len(seeds) == 3 and count == 12
    result = {'state': 'complete' if complete else 'partial', 'complete_cells': count,
              'expected_cells': 12, 'seeds': seeds, 'arms': ARMS,
              'test': {'source_sha256': TEST_SHA256, 'evaluated_indices': [0, 499], 'rows': 500},
              'cells': cells, 'arm_summary': {}, 'paired_effects': {},
              'uncertainty': {'method': 'paired seed mean and sample standard deviation',
                  'ci': None, 'reason': 'No verified patient/study grouping; no image-independent CI.'},
              'limitations': ['B and C use a generation mode absent from their training targets.',
                  'D-A changes training supervision and inference together.',
                  '500 cases have been inspected previously; not a pristine confirmatory holdout.',
                  'Finding BI-RADS metrics are conditional on successful localization.',
                  'Runtime errors and invalid predictions remain in all-image denominators.']}
    for arm in ARMS:
        available = [cells[str(s)][arm] for s in seeds if cells[str(s)][arm]['state'] == 'complete']
        result['arm_summary'][arm] = {'state': 'complete' if len(available) == 3 else 'pending',
            'metrics': {m: mean_std([c['metrics'][m] for c in available]) for m in METRIC_NAMES}}
    for effect, (high, low) in EFFECTS.items():
        paired = [s for s in seeds if high in rows[s] and low in rows[s]]
        result['paired_effects'][effect] = {'state': 'complete' if len(paired) == 3 else 'pending',
            'metrics': {m: mean_std([cells[str(s)][high]['metrics'][m] -
                                     cells[str(s)][low]['metrics'][m] for s in paired]) for m in METRIC_NAMES}}
    if groups_path:
        groups, metadata = load_groups(groups_path)
        result['uncertainty'] = dict(metadata, method='paired crossed bootstrap of seeds and test clusters',
                                     replicates=replicates, ci=None,
                                     limitation='Only three training seeds; bootstrap intervals are exploratory.')
        if complete:
            result['uncertainty']['ci'] = bootstrap(rows, seeds, groups, replicates)
    return result


def markdown(result):
    lines = ['# 4B CoT 配对实验', '', f"状态：{result['state']}；完成 {result['complete_cells']}/12 个实验单元。",
             '', '所有单元固定相同 500 张（索引 0–499）；数值为三个训练种子的均值 ± 样本标准差。',
             '未完成单元标记 pending；部分种子的数值不作为最终对比。', '',
             '| 指标 | A 直接训练/关闭思考 | B 直接训练/开启思考 | C CoT训练/关闭思考 | D CoT训练/开启思考 |',
             '|---|---:|---:|---:|---:|']
    for metric, label in METRIC_NAMES.items():
        values = []
        for arm in ARMS:
            summary = result['arm_summary'][arm]
            v = summary['metrics'][metric]
            values.append(f"{100*v['mean']:.2f} ± {100*v['std']:.2f}%" if summary['state'] == 'complete'
                          else f"pending ({v['n']}/3)")
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '| 配对效应（百分点） | 定位 F1 | 联合病灶 F1 | 乳房 BI-RADS Accuracy |', '|---|---:|---:|---:|']
    for effect, summary in result['paired_effects'].items():
        values = []
        for metric in ('localized_f1_iou_0_3', 'joint_finding_f1_iou_0_3', 'breast_birads_accuracy'):
            v = summary['metrics'][metric]
            value = f"{v['mean']*100:+.2f} ± {v['std']*100:.2f}" if summary['state'] == 'complete' else 'pending'
            ci = result['uncertainty'].get('ci')
            if ci:
                low, high = ci[effect][metric]
                value += f' [95% CI {100*low:+.2f}, {100*high:+.2f}]'
            values.append(value)
        lines.append('| ' + effect + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'D−A 为完整方案效应；B−A 和 D−C 为各自训练条件下切换思考模式的效应；C−A 为关闭思考时训练目标的效应。',
              'B、C 存在训练/推理模式不匹配；D−A 不能单独归因为推理时思考。',
              '500 张已经用于错误分析，结果不属于未接触测试集上的确认性结论。', '',
              '| seed / arm | 状态 | 无效JSON | 运行错误 | 截断 | 正常图误报 | 平均生成token | 平均秒/图 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for seed, arms in result['cells'].items():
        for arm, cell in arms.items():
            if cell['state'] != 'complete':
                lines.append(f"| {seed}/{arm} | {cell['state']} | — | — | — | — | — | — |")
                continue
            q, cost = cell['quality'], cell['cost']
            rate = lambda v: f'{100*v:.2f}%' if v is not None else 'unknown'
            num = lambda v: f'{v:.2f}' if v is not None else 'unknown'
            lines.append(f"| {seed}/{arm} | complete | {rate(q['invalid_json_rate'])} | {rate(q['runtime_error_rate'])} | "
                         f"{rate(q['truncation_rate'])} | {rate(q['normal_false_positive_rate'])} | "
                         f"{num(cost['output_tokens']['mean'])} | {num(cost['seconds']['mean'])} |")
    lines += ['', '逐类别漏检/误报、密度及 BI-RADS 类别召回见 comparison.json。',
              '若没有经核实的患者/检查分组，仅报告种子均值和标准差，不假设每张图片独立。', '']
    return '\n'.join(lines)


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--test-data', type=Path, required=True)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--groups', type=Path)
    parser.add_argument('--bootstrap-replicates', type=int, default=500)
    args = parser.parse_args()
    try:
        seeds = [int(x) for x in args.seeds.split(',')]
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValidationError('EXACTLY_THREE_UNIQUE_SEEDS_REQUIRED')
        if args.bootstrap_replicates < 100:
            raise ValidationError('BOOTSTRAP_REPLICATES_TOO_SMALL')
        result = assemble(args.run_dir, load_test(args.test_data), seeds, args.groups, args.bootstrap_replicates)
        result['evaluator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['core_evaluator_sha256'] = hashlib.sha256(Path(__file__).with_name('evaluate_mammo.py').read_bytes()).hexdigest()
        atomic_write(args.run_dir / 'comparison.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        atomic_write(args.run_dir / 'comparison.md', markdown(result))
        print(json.dumps({'state': result['state'], 'complete_cells': result['complete_cells'], 'expected_cells': 12}))
    except (ValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        code = str(exc) if isinstance(exc, ValidationError) else 'EVALUATION_INPUT_ERROR'
        print(json.dumps({'state': 'failed', 'error_code': code}))
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
