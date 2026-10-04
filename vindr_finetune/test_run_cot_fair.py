import argparse
import fcntl
import json
from pathlib import Path
import tempfile
import unittest

from run_cot_fair import (ClosedError, Runner, adapter_complete,
                          latest_checkpoint, prediction_count, verify_evaluation)


class RunnerTests(unittest.TestCase):
    def test_incomplete_new_checkpoint_does_not_displace_resumable_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (100, 200):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
                for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                             "scheduler.pt", "rng_state.pth"):
                    (checkpoint / name).write_bytes(b"x")
            (root / "checkpoint-200/optimizer.pt").unlink()
            self.assertEqual(latest_checkpoint(root).name, "checkpoint-100")

    def test_final_adapter_requires_finished_epoch_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (root / name).write_bytes(b"x")
            state = root / "trainer_state.json"
            state.write_text(json.dumps({"global_step": 584, "epoch": 1.99}))
            self.assertFalse(adapter_complete(root, 584))
            state.write_text(json.dumps({"global_step": 584, "epoch": 2.0}))
            self.assertTrue(adapter_complete(root, 584))

    def test_prediction_resume_repairs_only_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            row = {"index": 0, "image_id": "synthetic", "ground_truth": {"a": 1}}
            test = [{"images": ["images_png/synthetic.png"], "output": '{"a":1}'}]
            good = json.dumps(row) + "\n"
            path.write_text(good + '{"index":')
            self.assertEqual(prediction_count(path, test), 1)
            self.assertEqual(path.read_text(), good)
            row["ground_truth"] = {"a": 2}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ClosedError, "prediction_truth_mismatch"):
                prediction_count(path, test)

    def test_cannot_reuse_stage_with_changed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(run_dir=root)
            first = Runner(args, {"seed": 42})
            first.bind(root / "stage", {"adapter": "a"})
            first.bind(root / "stage", {"adapter": "a"})
            with self.assertRaisesRegex(ClosedError, "stage_config_changed"):
                Runner(args, {"seed": 43}).bind(root / "stage", {"adapter": "a"})

    def test_runtime_error_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            row = {"index": 0, "image_id": "synthetic", "ground_truth": {"a": 1},
                   "error": "RuntimeError", "eval_json": None}
            test = [{"images": ["images_png/synthetic.png"], "output": '{"a":1}'}]
            path.write_text(json.dumps(row) + "\n")
            config = {"adapter_sha256": "expected", "instruction_mode": "neutral"}
            (path.parent / "run_config.json").write_text(json.dumps(config))
            self.assertEqual(prediction_count(path, test, config), 1)
            with self.assertRaisesRegex(ClosedError, "prediction_run_config_mismatch"):
                prediction_count(path, test, dict(config, adapter_sha256="changed"))

    def test_tail_repair_refuses_active_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text('{"index":')
            with (path.parent / ".inference.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ClosedError, "inference_writer_active"):
                    prediction_count(path, [])
                self.assertEqual(path.read_text(), '{"index":')

    def test_partial_evaluation_cannot_mark_experiment_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = {"state": "partial", "complete_cells": 4,
                       "cells": {"42": {arm: {"state": "complete"} for arm in "ABCD"}}}
            (root / "comparison.json").write_text(json.dumps(receipt))
            verify_evaluation(root, 42)
            with self.assertRaisesRegex(ClosedError, "final_evaluation_incomplete"):
                verify_evaluation(root)
            receipt.update(state="complete", complete_cells=12)
            for seed in (43, 44):
                receipt["cells"][str(seed)] = {arm: {"state": "complete"} for arm in "ABCD"}
            (root / "comparison.json").write_text(json.dumps(receipt))
            verify_evaluation(root)


if __name__ == "__main__":
    unittest.main()
