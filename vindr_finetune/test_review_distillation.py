import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_distillation as review


class ReviewGateTests(unittest.TestCase):
    def test_changed_label_and_truncation_are_rejected(self):
        label = '{"breast_birads":1,"breast_density":"B","findings":[]}'
        value = {"verdict": "revised", "issue_codes": [], "rationale": "Grounded rationale. " * 10,
                 "final_answer": json.loads(label)}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
        self.assertEqual(review.validate(response, label), value)
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(review.Rejected, "REVIEW_TRUNCATED"):
            review.validate(response, label)
        response["choices"][0]["finish_reason"] = "stop"
        value["final_answer"]["breast_birads"] = 5
        response["choices"][0]["message"]["content"] = json.dumps(value)
        with self.assertRaisesRegex(review.Rejected, "REVIEW_LABEL_CONFLICT"):
            review.validate(response, label)

    def test_reviewed_dataset_preserves_sampling_and_excludes_rejected_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            out = root / "reviewed"
            source.mkdir()
            first = {"images": ["images_png/a.png"], "instruction": "<image>",
                     "output": '{"breast_birads":1,"breast_density":"B","findings":[]}'}
            second = {**first, "images": ["images_png/b.png"]}
            rows = [first, first, second]
            train = root / "train.json"
            train.write_text(json.dumps(rows))
            source_data = [{**row, "output": "<think>" + "Original rationale. " * 10 +
                            "</think>\n" + row["output"]} for row in rows]
            data = source / "train_balanced_2to1_thinking.json"
            data.write_text(json.dumps(source_data))
            original_blob = data.read_bytes()
            (source / "summary.json").write_text(json.dumps({"state": "complete", "output_rows": 3,
                                                          "output_sha256": review.digest(original_blob)}))
            cached = [{"key": review.key_for(row), "reasoning": "Original rationale. " * 10}
                      for row in [first, second]]
            (source / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cached))
            audit = root / "audit.json"
            audit.write_text(json.dumps({"samples": [
                {"cache_position": 1, "model_audit": {"recommended_review": True},
                 "review_followup": {"recommended_review": False}},
                {"cache_position": 2, "model_audit": {"recommended_review": True}}]}))

            def fake_review(args, row, cached):
                value = {"verdict": "revised" if row == first else "reject", "issue_codes": [],
                         "rationale": "Repaired grounded rationale. " * 10,
                         "final_answer": json.loads(row["output"])}
                response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
                return {"review": value, "response": response,
                        "original_rationale_sha256": review.digest(cached["reasoning"].encode())}

            argv = ["review", "--source-run", str(source), "--audit", str(audit), "--train", str(train),
                    "--output-dir", str(out), "--git-commit", "fixture", "--await-final-dataset"]
            with patch("sys.argv", argv), patch.object(review, "SOURCE_SHA", review.digest(train.read_bytes())), \
                    patch.object(review, "review", side_effect=fake_review) as calls:
                review.main()
                self.assertEqual(calls.call_count, 2)
                review.main()
                self.assertEqual(calls.call_count, 2, "resume must not call the teacher again")
            result = json.loads((out / "train_balanced_2to1_thinking.json").read_text())
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], result[1], "oversampling must be preserved")
            self.assertTrue(result[0]["output"].endswith("</think>\n" + first["output"]))
            self.assertIn("Repaired grounded rationale", result[0]["output"])
            self.assertEqual(len(json.loads((out / "excluded.json").read_text())), 1)
            self.assertEqual(data.read_bytes(), original_blob, "original outputs must remain untouched")


if __name__ == "__main__":
    unittest.main()
