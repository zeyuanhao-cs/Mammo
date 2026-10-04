import json
import tempfile
import unittest
from pathlib import Path
from cot_protocol import completed_indices, neutral_instruction


class CotProtocolTest(unittest.TestCase):
    def test_neutral_policy_preserves_schema(self):
        text = ('<image> Analyze the mammography image and return ONLY valid JSON.\n'
                'Return ONLY a JSON object with exactly these keys: breast_birads, findings.\n'
                'Do not output explanations, markdown, code fences, or extra text.')
        actual = neutral_instruction(text)
        self.assertIn('breast_birads, findings', actual)
        self.assertIn('<image>', actual)
        self.assertNotIn('ONLY', actual)
        self.assertNotIn('Do not output explanations', actual)

    def test_recovery_retains_complete_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'predictions.jsonl'
            path.write_bytes(b'{"index":0}\n{"index":')
            self.assertEqual(completed_indices(path), {0})
            self.assertEqual(path.read_bytes(), b'{"index":0}\n')

    def test_duplicate_and_middle_corruption_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'predictions.jsonl'
            for blob in [b'{"index":0}\n{"index":0}\n', b'bad\n{"index":1}\n']:
                path.write_bytes(blob)
                with self.assertRaises(ValueError):
                    completed_indices(path)
                self.assertEqual(path.read_bytes(), blob)


if __name__ == '__main__':
    unittest.main()
