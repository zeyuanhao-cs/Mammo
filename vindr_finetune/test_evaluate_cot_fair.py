import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_cot_fair import (ValidationError, assemble, bootstrap, load_groups,
                               markdown, summarize_cell, validate_rows)


def fixture(count=500):
    truth = {'breast_birads': 2, 'breast_density': 'B', 'findings': [
        {'finding_category': 'Mass', 'finding_birads': 2, 'bbox': [0, 0, 10, 10]}]}
    test = [{'output': json.dumps(truth), 'images': [f'/public/{i}.png']} for i in range(count)]
    rows = [{'index': i, 'image_id': str(i), 'ground_truth': copy.deepcopy(truth),
             'eval_json': copy.deepcopy(truth), 'seconds': 2, 'output_tokens': 10,
             'finish_reason': 'stop'} for i in range(count)]
    return test, rows


class FairEvaluationTests(unittest.TestCase):
    def test_strict_case_alignment(self):
        test, rows = fixture()
        self.assertEqual(validate_rows(list(reversed(rows)), test)[0]['index'], 0)
        with self.assertRaisesRegex(ValidationError, 'ROW_COUNT_MISMATCH'):
            validate_rows(rows[:-1], test)
        duplicate = copy.deepcopy(rows)
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValidationError, 'DUPLICATE_INDEX'):
            validate_rows(duplicate, test)
        wrong_gt = copy.deepcopy(rows)
        wrong_gt[0]['ground_truth']['breast_density'] = 'D'
        with self.assertRaisesRegex(ValidationError, 'GROUND_TRUTH_MISMATCH'):
            validate_rows(wrong_gt, test)
        wrong_image = copy.deepcopy(rows)
        wrong_image[0]['image_id'] = 'another'
        with self.assertRaisesRegex(ValidationError, 'IMAGE_MAPPING_MISMATCH'):
            validate_rows(wrong_image, test)
        with self.assertRaisesRegex(ValidationError, 'ROW_COUNT_MISMATCH'):
            validate_rows(rows + [rows[0]], test)

    def test_errors_and_invalid_remain_in_denominator(self):
        _, rows = fixture(4)
        rows[0]['error'] = 'RUNTIME_ERROR'
        rows[1]['eval_json'] = None
        rows[1]['finish_reason'] = 'length'
        result = summarize_cell(rows)
        self.assertEqual(result['metrics']['breast_birads_accuracy'], .5)
        self.assertEqual(result['metrics']['json_valid_rate'], .5)
        self.assertAlmostEqual(result['metrics']['localized_f1_iou_0_3'], 2 / 3)
        self.assertEqual(result['counts']['runtime_errors'], 1)
        self.assertEqual(result['quality']['truncation_rate'], .25)
        self.assertEqual(result['per_density']['B']['missed'], 2)
        self.assertEqual(result['per_category']['Mass']['false_negative'], 2)
        self.assertEqual(result['cost']['output_tokens']['mean'], 10)
        self.assertTrue(rows[0]['eval_json'])  # Input is preserved.

    def test_normal_false_positive_and_unknown_cost(self):
        _, rows = fixture(2)
        for row in rows:
            row['ground_truth']['findings'] = []
            row.pop('output_tokens')
        rows[1]['eval_json']['findings'] = []
        rows[1].pop('finish_reason')
        result = summarize_cell(rows)
        self.assertEqual(result['quality']['normal_false_positive_rate'], .5)
        self.assertIsNone(result['quality']['truncation_rate'])
        self.assertIsNone(result['cost']['output_tokens']['mean'])

    def test_generated_tokens_preferred_with_legacy_fallback(self):
        _, rows = fixture(3)
        rows[0]['generated_tokens'] = 20
        rows[1]['generated_tokens'] = None
        rows[2]['generated_tokens'] = 0
        cost = summarize_cell(rows)['cost']['output_tokens']
        self.assertEqual(cost['count'], 3)
        self.assertEqual(cost['sum'], 30)
        self.assertEqual(cost['mean'], 10)

    def test_never_complete_until_all_twelve_cells(self):
        test, rows = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                for arm in 'ABCD':
                    path = root / f'seed{seed}' / arm / 'predictions.jsonl'
                    path.parent.mkdir(parents=True)
                    selected = rows[:-1] if (seed, arm) == (44, 'D') else rows
                    path.write_text('\n'.join(json.dumps(r) for r in selected))
            partial = assemble(root, test, [42, 43, 44])
            self.assertEqual(partial['state'], 'partial')
            self.assertEqual(partial['complete_cells'], 11)
            self.assertIn('pending (2/3)', markdown(partial))
            self.assertIsNone(partial['uncertainty']['ci'])
            (root / 'seed44/D/predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            complete = assemble(root, test, [42, 43, 44])
            self.assertEqual(complete['state'], 'complete')
            self.assertEqual(complete['paired_effects']['D-A']['metrics']['localized_f1_iou_0_3']['mean'], 0)
            self.assertEqual(complete['arm_summary']['A']['metrics']['json_valid_rate']['std'], 0)

    def test_groups_require_explicit_provenance_and_complete_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'groups.json'
            mapping = {'unit': 'study', 'source': 'public dataset mapping',
                       'index_to_group': {str(i): str(i // 4) for i in range(500)}}
            path.write_text(json.dumps(mapping))
            groups, metadata = load_groups(path)
            self.assertEqual(len(groups), 125)
            self.assertEqual(metadata['unit'], 'study')
            del mapping['index_to_group']['0']
            path.write_text(json.dumps(mapping))
            with self.assertRaisesRegex(ValidationError, 'GROUP_INDEX_SET_MISMATCH'):
                load_groups(path)

    def test_bootstrap_paired_identical_arms_have_zero_difference(self):
        _, rows = fixture(4)
        all_rows = {s: {a: rows for a in 'ABCD'} for s in (42, 43, 44)}
        ci = bootstrap(all_rows, [42, 43, 44], [[0, 1], [2, 3]], 10)
        for effect in ci.values():
            for interval in effect.values():
                self.assertEqual(interval, [0, 0])


if __name__ == '__main__':
    unittest.main()
