"""Offline gates for fair data preparation; no images or teacher network needed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_cot_fair as fair

LABEL = '{"breast_birads":1,"breast_density":"B","findings":[]}'
RATIONALE = ("The breast shows scattered fibroglandular tissue. No discrete suspicious mass "
             "or focal distortion is apparent on this image; subtle changes remain uncertain.")


def row(image="a"):
    return {"instruction": "<image>\nAnalyze the mammography image and return ONLY valid JSON.\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            "breast_birads, breast_density, findings.\n"
            "Do not output explanations, markdown, code fences, or extra text.",
            "images": ["images_png/" + image + ".png"], "output": LABEL}


def decision(verdict="accept", rationale=RATIONALE):
    return {"verdict": verdict, "rationale": rationale if verdict != "reject" else "",
            "final_answer": json.loads(LABEL)}


def response(value, reasoning="Do not use this internal reasoning as a training target."):
    return {"choices": [{"finish_reason": "stop", "message": {
        "reasoning_content": reasoning, "content": json.dumps(value)}}]}


class PreparationTests(unittest.TestCase):
    def test_only_structured_final_rationale_is_used(self):
        value = decision()
        self.assertEqual(fair.unpack(response(value), LABEL, "generate"), value)
        reply = response(value)
        reply["choices"][0]["message"]["content"] = "<think>untrusted planning</think>" + json.dumps(value)
        self.assertEqual(fair.unpack(reply, LABEL, "review"), value)
        reply["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(fair.Rejected, "INCOMPLETE_GENERATION"):
            fair.unpack(reply, LABEL, "review")

    def test_label_meta_length_and_invalid_semantic_reject_fail(self):
        for value, code in (
            ({**decision(), "final_answer": {}}, "FINAL_LABEL_MISMATCH"),
            (decision(rationale="The reference annotation indicates this density. " * 4), "RATIONALE_META_TEXT"),
            (decision(rationale="x" * 1001), "RATIONALE_LENGTH"),
            ({**decision("reject"), "rationale": RATIONALE}, "REJECT_MUST_HAVE_EMPTY_RATIONALE"),
            ({**decision(), "extra": "not allowed"}, "INVALID_RESPONSE_SCHEMA"),
        ):
            with self.subTest(code=code), self.assertRaisesRegex(fair.Rejected, code):
                fair.validate_value(value, LABEL, "review")
        self.assertEqual(fair.validate_value(decision("reject"), LABEL, "review")["verdict"], "reject")

    def test_paired_arms_keep_order_multiplicity_and_exact_label(self):
        rows = [row("a"), row("b"), row("a"), row("c")]
        decisions = {fair.key_for(r): decision("reject" if r["images"] == ["images_png/b.png"] else "revised")
                     for r in rows}
        direct, cot, excluded = fair.build_arms(rows, decisions)
        self.assertEqual([r["images"] for r in direct], [["images_png/a.png"], ["images_png/a.png"], ["images_png/c.png"]])
        self.assertEqual(direct[0], direct[1])
        self.assertEqual(excluded[0]["source_index"], 1)
        for a, b in zip(direct, cot):
            self.assertEqual(a["instruction"], b["instruction"])
            self.assertEqual(a["images"], b["images"])
            self.assertEqual(a["output"], LABEL)
            self.assertEqual(b["output"], "<think>" + RATIONALE + "</think>\n" + LABEL)
        with self.assertRaisesRegex(fair.Rejected, "INCOMPLETE_DECISIONS"):
            fair.build_arms(rows, {})

    def test_transient_failure_stops_without_publication_and_resume_reuses_drafts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images_png").mkdir()
            for name in ("a", "b"):
                (root / "images_png" / (name + ".png")).write_bytes(b"fixture")
            source, test, out = root / "train.json", root / "test.json", root / "out"
            out.mkdir()
            source.write_text(json.dumps([row("a"), row("b"), row("a")]))
            test.write_text(json.dumps([row("test")]))
            args = SimpleNamespace(source=source, test=test, output_dir=out, image_root=root,
                                   model="fixture", base_url="http://unused", git_commit="fixture")
            calls = []

            def request(args, r, stage, candidate=None):
                calls.append((r["images"][0], stage))
                if len(calls) == 2:
                    raise fair.Rejected("NETWORK_ERROR")
                return decision("revised" if stage == "review" else "accept")

            with patch.object(fair, "SOURCE_SHA", fair.digest(source.read_bytes())), \
                    patch.object(fair, "TEST_SHA", fair.digest(test.read_bytes())), \
                    patch.object(fair, "EXPECTED_ROWS", 3), patch.object(fair, "EXPECTED_UNIQUE", 2), \
                    patch.object(fair, "request_stage", side_effect=request), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(fair.Rejected, "NETWORK_ERROR"):
                    fair.execute(args)
                self.assertFalse((out / "summary.json").exists())
                self.assertFalse((out / "direct_train.json").exists())
                fair.execute(args)
                self.assertEqual(calls.count(("images_png/a.png", "generate")), 1)
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["state"], "complete")
                self.assertEqual(summary["output_rows"], 3)
                self.assertEqual(summary["excluded_rows"], 0)
                before = len(calls)
                fair.execute(args)
                self.assertEqual(before, len(calls), "completed rerun should make no teacher calls")
                for output in summary["outputs"].values():
                    self.assertEqual(output["sha256"], fair.digest(Path(output["path"]).read_bytes()))

    def test_network_retries_bounded_and_never_become_semantic_reject(self):
        args = SimpleNamespace(image_root=Path("/unused"), model="fixture", base_url="http://unused", timeout=1)
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            image.write_bytes(b"fixture")
            with patch.object(fair, "image_path", return_value=image), \
                    patch.object(fair.urllib.request, "urlopen", side_effect=TimeoutError) as calls, \
                    patch.object(fair.time, "sleep"):
                with self.assertRaisesRegex(fair.Rejected, "NETWORK_ERROR"):
                    fair.request_stage(args, row(), "generate")
                self.assertEqual(calls.call_count, 3)


if __name__ == "__main__":
    unittest.main()
