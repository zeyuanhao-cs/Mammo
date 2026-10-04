#!/usr/bin/env python3
"""Resumable paired-seed CoT experiment. Raw records and logs stay server-side."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TRAIN_SHA = "93d53adefac511f8d355e5dc72ead78526a54e1c2e1d55a6aba0890bc1584ce9"
TEST_SHA = "6eddbabedff8e37df2f5629585b4947fea3cbea6df9f61d9a1fed13548b8249e"
ARMS = {"A": ("direct", False), "B": ("direct", True),
        "C": ("cot", False), "D": ("cot", True)}


class ClosedError(Exception):
    """Only a fixed diagnostic code, never dataset text."""


def require(condition, code):
    if not condition:
        raise ClosedError(code)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def final_label(output):
    return json.loads(output.rsplit("</think>", 1)[-1].strip())


def sample_key(row):
    return (tuple(row["images"]), json.dumps(final_label(row["output"]), sort_keys=True))


def latest_checkpoint(adapter):
    candidates = []
    for path in Path(adapter).glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
            state = read_json(path / "trainer_state.json")
            valid = (state.get("global_step") == step and
                     all((path / f).is_file() and (path / f).stat().st_size
                         for f in ("adapter_config.json", "adapter_model.safetensors",
                                   "optimizer.pt", "scheduler.pt")) and
                     bool(list(path.glob("rng_state*.pth"))))
            if valid:
                candidates.append((step, path))
        except (ValueError, OSError):
            continue
    return max(candidates, default=(0, None))[1]


def adapter_complete(adapter, expected_steps):
    adapter = Path(adapter)
    if not all((adapter / f).is_file() and (adapter / f).stat().st_size
               for f in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json")):
        return False
    state = read_json(adapter / "trainer_state.json")
    return (state.get("epoch", 0) >= 2 - 1e-6 and
            state.get("global_step", 0) >= expected_steps)


def prediction_count(path, test, expected_config=None):
    if not Path(path).exists():
        return 0
    path = Path(path)
    with (path.parent / ".inference.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ClosedError("inference_writer_active") from None
        if expected_config is not None:
            require((path.parent / "run_config.json").is_file(), "predictions_without_run_config")
            actual = read_json(path.parent / "run_config.json")
            require(all(actual.get(key) == value for key, value in expected_config.items()),
                    "prediction_run_config_mismatch")
        return _prediction_count_locked(path, test)


def _prediction_count_locked(path, test):
    from cot_protocol import completed_indices
    # The inference writer flushes each row. A preemption can leave an unfinished
    # final row; use the same narrow recovery rule as inference before validation.
    completed_indices(Path(path))
    seen = set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ClosedError("prediction_jsonl_incomplete") from None
            index = row.get("index")
            require(type(index) is int and 0 <= index < 500 and index not in seen,
                    "prediction_index_mismatch")
            truth = row.get("ground_truth")
            if isinstance(truth, str):
                truth = json.loads(truth)
            require(truth == json.loads(test[index]["output"]), "prediction_truth_mismatch")
            require(row.get("image_id") == Path(test[index]["images"][0]).stem,
                    "prediction_image_mismatch")
            seen.add(index)
    return len(seen)


def verify_evaluation(root, seed=None):
    comparison = read_json(Path(root) / "comparison.json")
    if seed is not None:
        cells = comparison.get("cells", {}).get(str(seed), {})
        require(set(cells) == set(ARMS) and all(cell.get("state") == "complete" for cell in cells.values()),
                "seed_evaluation_incomplete")
    else:
        require(comparison.get("state") == "complete" and comparison.get("complete_cells") == 12,
                "final_evaluation_incomplete")
        for expected_seed in (42, 43, 44):
            verify_evaluation(root, expected_seed)


def preflight(args):
    import yaml
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer

    dataset = args.dataset_dir.resolve()
    require((dataset / "summary.json").is_file(), "dataset_not_ready")
    summary = read_json(dataset / "summary.json")
    require(summary.get("state") == "complete", "dataset_not_ready")
    info = read_json(dataset / "dataset_info.json")
    original_path = HERE / "data/train_balanced_2to1.json"
    test_path = HERE / "data/direct_test.json"
    require(digest(original_path) == TRAIN_SHA and digest(test_path) == TEST_SHA,
            "baseline_source_hash_mismatch")
    original, test = read_json(original_path), read_json(test_path)
    require(len(original) == 4657 and len(test) == 4000, "baseline_count_mismatch")
    files = {kind: (dataset / info["fair_" + kind]["file_name"]).resolve()
             for kind in ("direct", "cot")}
    require(all(p.is_relative_to(dataset) for p in files.values()), "dataset_path_escape")
    hashes = {kind: digest(path) for kind, path in files.items()}
    receipt_outputs = summary["outputs"]
    require(hashes["direct"] == receipt_outputs["direct_train"]["sha256"] and
            hashes["cot"] == receipt_outputs["cot_train"]["sha256"] and
            digest(dataset / "dataset_info.json") == receipt_outputs["dataset_info"]["sha256"],
            "dataset_receipt_hash_mismatch")
    rows = {kind: read_json(path) for kind, path in files.items()}
    count = len(rows["direct"])
    require(count == len(rows["cot"]) == summary["output_rows"] and count > 0,
            "paired_train_count_mismatch")
    require(summary["source_rows"] == len(original) and summary["excluded_rows"] == len(original) - count,
            "excluded_row_count_mismatch")
    original_counts = Counter(sample_key(row) for row in original)
    kept_counts = Counter(sample_key(row) for row in rows["direct"])
    require(all(original_counts[key] == n for key, n in kept_counts.items()),
            "original_multiplicity_changed")
    require(summary["accepted_unique"] == len(kept_counts) and
            summary["excluded_unique"] == len(original_counts) - len(kept_counts),
            "excluded_unique_count_mismatch")
    selected_original = [row for row in original if sample_key(row) in kept_counts]
    train_images = set()
    from cot_protocol import neutral_instruction
    for old, direct, cot in zip(selected_original, rows["direct"], rows["cot"]):
        require(old["images"] == direct["images"] == cot["images"], "paired_train_image_mismatch")
        require(direct["instruction"] == cot["instruction"] == neutral_instruction(old["instruction"]),
                "paired_train_prompt_mismatch")
        require(direct["instruction"].count("<image>") == len(direct["images"]) == 1,
                "train_image_placeholder_mismatch")
        require(final_label(direct["output"]) == final_label(cot["output"]) == json.loads(old["output"]),
                "paired_train_label_mismatch")
        require("<think>" not in direct["output"] and cot["output"].startswith("<think>") and
                cot["output"].count("</think>") == 1, "train_thinking_format_mismatch")
        train_images.update(direct["images"])
    test_images = {image for row in test for image in row["images"]}
    require(len({image for row in original for image in row["images"]}) == 2948
            and not train_images & test_images, "split_overlap_or_count")
    for name in train_images | test_images:
        image = (Path("/mammo") / name).resolve()
        require(image.is_relative_to(Path("/mammo/images_png")) and image.is_file()
                and os.access(image, os.R_OK), "image_missing_or_unreadable")
    model = args.model.resolve()
    index = read_json(model / "model.safetensors.index.json")
    require(all((model / shard).is_file() and (model / shard).stat().st_size
                for shard in set(index["weight_map"].values())), "model_shard_missing")
    config = yaml.safe_load((HERE / "trial_4b_thinking.yaml").read_text())
    config.update(dataset_dir=str(dataset), model_name_or_path=str(model), media_dir="/mammo")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    token_max = {}
    for kind in ("direct", "cot"):
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(
            template="qwen3_5", enable_thinking=(kind == "cot")))
        maximum = 0
        seen = set()
        for row in rows[kind]:
            key = (row["instruction"], row["output"])
            if key in seen:
                continue
            seen.add(key)
            prompt, target = template.encode_oneturn(tokenizer, [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]}])
            if kind == "cot":
                decoded = tokenizer.decode(target)
                require("</think>" in decoded and final_label(decoded.split("<|im_end|>")[0]) == final_label(row["output"]),
                        "template_lost_thinking_or_answer")
            maximum = max(maximum, len(prompt) + len(target))
        require(maximum + 2048 <= config["cutoff_len"], "sequence_may_truncate")
        token_max[kind] = maximum
    versions = {}
    for package in ("torch", "transformers", "peft", "llamafactory", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    protocol = {"version": 1, "git_commit": os.environ["MAMMO_COMMIT"],
                "seeds": args.seeds, "dataset_sha256": hashes,
                "original_train_sha256": TRAIN_SHA, "test_sha256": TEST_SHA,
                "train_rows": count, "train_unique_images": len(train_images),
                "source_train_rows": len(original), "excluded_rows": summary["excluded_rows"],
                "excluded_unique": summary["excluded_unique"],
                "neutral_prompt_sha256": fingerprint([row["instruction"] for row in rows["direct"]]),
                "test_source_rows": len(test), "evaluation_rows": 500,
                "evaluation_indices": [0, 499], "train_test_overlap": 0,
                "max_text_tokens": token_max, "vision_token_allowance": 2048,
                "train_config": config, "versions": versions,
                "model_config_sha256": digest(model / "config.json"),
                "model_index_sha256": digest(model / "model.safetensors.index.json"),
                "inference": {"instruction_mode": "neutral", "max_new_tokens": 4096,
                              "do_sample": False, "image_max_pixels": 786432,
                              "image_min_pixels": 262144, "flash_attn": "sdpa"},
                "arms": ARMS}
    # Round-trip tuples to JSON lists before equality checks on a resumed protocol.
    return json.loads(json.dumps(protocol)), config, test


class Runner:
    def __init__(self, args, protocol):
        self.args = args
        self.root = args.run_dir
        self.fingerprint = fingerprint(protocol)
        self.current = {}

    def report(self, stage, state, **fields):
        event = {"stage": stage, "state": state, "timestamp": time.time(),
                 "job_id": os.environ["JOB_ID"], "protocol_sha256": self.fingerprint, **fields}
        self.current = event
        atomic_json(self.root / "progress.json", event)
        with (self.root / "journal.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        print(json.dumps(event, sort_keys=True), flush=True)

    def budget(self, stage):
        if self.args.stop_before_unix and time.time() + 300 >= self.args.stop_before_unix:
            self.report(stage, "incomplete", code="reservation_deadline")
            raise SystemExit(75)

    def command(self, stage, command, log):
        self.budget(stage)
        self.report(stage, "running")
        with Path(log).open("ab") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            last_report = time.monotonic()
            while process.poll() is None:
                if self.args.stop_before_unix and time.time() + 300 >= self.args.stop_before_unix:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    self.report(stage, "incomplete", code="reservation_deadline")
                    raise SystemExit(75)
                if time.monotonic() - last_report >= 60:
                    self.report(stage, "running")
                    last_report = time.monotonic()
                time.sleep(5)
        require(process.returncode == 0, "stage_process_failed")

    def bind(self, directory, config):
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "stage_config.json"
        record = {"protocol_sha256": self.fingerprint, "config": config}
        if marker.exists():
            require(read_json(marker) == record, "stage_config_changed")
        else:
            require(not any(directory.iterdir()), "unowned_output_directory")
            atomic_json(marker, record)


def execute(args):
    import yaml
    require(os.environ.get("JOB_ID") and os.environ.get("MAMMO_COMMIT"), "missing_job_identity")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ClosedError("another_runner_active") from None
        protocol, base_config, test = preflight(args)
        manifest = args.run_dir / "protocol.json"
        if manifest.exists():
            require(read_json(manifest) == protocol, "protocol_changed")
        else:
            require(not list(args.run_dir.glob("seed*")), "unowned_seed_outputs")
            atomic_json(manifest, protocol)
        runner = Runner(args, protocol)
        runner.report("preflight", "complete")
        expected_steps = math.ceil(protocol["train_rows"] / 16) * 2
        for seed in args.seeds:
            seed_dir = args.run_dir / f"seed{seed}"
            seed_dir.mkdir(exist_ok=True)
            for kind in ("direct", "cot"):
                adapter = seed_dir / ("adapter_" + kind)
                config = dict(base_config, dataset="fair_" + kind, enable_thinking=kind == "cot",
                              output_dir=str(adapter), seed=seed, data_seed=seed)
                # Keep the ownership receipt outside Trainer's output directory:
                # LLaMA-Factory refuses nonempty output directories without a checkpoint.
                runner.bind(seed_dir / ("train_" + kind + "_metadata"), config)
                stage = f"seed{seed}.train.{kind}"
                if not adapter_complete(adapter, expected_steps):
                    checkpoint = latest_checkpoint(adapter)
                    if checkpoint:
                        config["resume_from_checkpoint"] = str(checkpoint)
                    elif adapter.exists() and any(adapter.iterdir()):
                        # A failure before the first checkpoint has no optimizer state
                        # to resume. Preserve all artifacts and restart the same seed.
                        preserved = adapter.with_name(adapter.name + f".interrupted-{time.time_ns()}")
                        adapter.rename(preserved)
                        runner.report(stage, "restarting", code="no_complete_checkpoint")
                    config_file = seed_dir / f"train_{kind}.yaml"
                    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
                    runner.command(stage, ["llamafactory-cli", "train", str(config_file)],
                                   seed_dir / f"train_{kind}.log")
                    require(adapter_complete(adapter, expected_steps), "final_adapter_incomplete")
                atomic_json(seed_dir / f"train_{kind}.complete.json",
                            {"protocol_sha256": runner.fingerprint, "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
                             "trainer_state": {k: read_json(adapter / "trainer_state.json").get(k)
                                               for k in ("epoch", "global_step", "max_steps")}})
                runner.report(stage, "complete", steps=expected_steps)
                for arm, (arm_kind, thinking) in ARMS.items():
                    if arm_kind != kind:
                        continue
                    output = seed_dir / arm
                    runner.bind(output, {"arm": arm, "adapter": str(adapter), "thinking": thinking})
                    stage = f"seed{seed}.infer.{arm}"
                    from cot_protocol import neutral_instruction
                    messages = [[{"role": "user", "content": neutral_instruction(row["instruction"])}]
                                for row in test[:500]]
                    infer_config = {
                        "base_model": str(args.model), "adapter": str(adapter),
                        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
                        "template": "qwen3_5", "prompt": "direct", "split": "test",
                        "data_sha256": TEST_SHA, "image_root": "/mammo",
                        "selected_indices": list(range(500)), "n_records_this_run": 500,
                        "selected_messages_sha256": fingerprint(messages),
                        "instruction_mode": "neutral", "enable_thinking": thinking,
                        "max_new_tokens": 4096, "image_max_pixels": 786432,
                        "image_min_pixels": 262144, "do_sample": False,
                        "flash_attn": "sdpa", "infer_dtype": "bfloat16",
                    }
                    count = prediction_count(output / "predictions.jsonl", test, infer_config)
                    if count < 500:
                        command = [sys.executable, str(HERE / "infer_lora.py"), "--prompt", "direct", "--split", "test",
                                   "--model", str(args.model), "--adapter", str(adapter), "--image-root", "/mammo",
                                   "--image-max-pixels", "786432", "--image-min-pixels", "262144", "--flash-attn", "sdpa",
                                   "--instruction-mode", "neutral", "--output-dir", str(output),
                                   "--limit", "500", "--max-new-tokens", "4096"]
                        if thinking:
                            command.append("--enable-thinking")
                        runner.command(stage, command, seed_dir / f"infer_{arm}.log")
                    require(prediction_count(output / "predictions.jsonl", test, infer_config) == 500,
                            "inference_incomplete")
                    runner.report(stage, "complete", rows=500)
            runner.command(f"seed{seed}.evaluate", [sys.executable, str(HERE / "evaluate_cot_fair.py"),
                           "--run-dir", str(args.run_dir), "--test-data", str(HERE / "data/direct_test.json")],
                           seed_dir / "evaluate.log")
            verify_evaluation(args.run_dir, seed)
            runner.report(f"seed{seed}.evaluate", "complete")
        verify_evaluation(args.run_dir)
        runner.report("experiment", "complete", seeds=len(args.seeds), arms_per_seed=4, rows_per_arm=500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--model", type=Path, default=Path("/data/models/Qwen3.5-4B"))
    parser.add_argument("--stop-before-unix", type=float)
    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(",")]
    require(args.seeds == [42, 43, 44], "paired_seeds_must_be_42_43_44")
    args.run_dir = args.run_dir.resolve()
    args.model = args.model.resolve()
    try:
        execute(args)
    except Exception as exc:
        code = str(exc) if isinstance(exc, ClosedError) else type(exc).__name__
        event = {"state": "incomplete" if code == "dataset_not_ready" else "failed",
                 "code": code, "timestamp": time.time(), "job_id": os.environ.get("JOB_ID")}
        if args.run_dir.exists() and code != "another_runner_active":
            atomic_json(args.run_dir / "progress.json", event)
        print(json.dumps(event), flush=True)
        return 75 if code == "dataset_not_ready" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
