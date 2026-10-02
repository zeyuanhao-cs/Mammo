"""Baseline scope must survive extra predictions and reject missing/duplicate cases."""
import unittest

from compare_thinking import baseline_subset


class BaselineAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.test = [{'images': [f'images_png/{i}.png'], 'output': '{}'} for i in range(501)]
        self.baseline = [{'index': i, 'ground_truth': '{}'} for i in range(500)]
        self.rows = [dict(r, image_id=str(r['index'])) for r in self.baseline]

    def test_extra_cases_are_excluded_and_baseline_order_is_used(self):
        extra = {'index': 500, 'ground_truth': '{}', 'image_id': '500'}
        selected = baseline_subset([extra] + list(reversed(self.rows)), self.baseline, self.test)
        self.assertEqual([r['index'] for r in selected], list(range(500)))

    def test_missing_and_duplicate_baseline_cases_are_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaises(AssertionError):
                baseline_subset(rows, self.baseline, self.test)

    def test_wrong_image_and_labels_are_rejected(self):
        for field, value in [('image_id', 'wrong'), ('ground_truth', '{"wrong":true}')]:
            rows = [dict(r) for r in self.rows]
            rows[0][field] = value
            with self.assertRaises(AssertionError):
                baseline_subset(rows, self.baseline, self.test)


if __name__ == '__main__':
    unittest.main()
