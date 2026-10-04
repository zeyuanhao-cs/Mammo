#!/usr/bin/env python3
"""Prepare matched answer-only/short-rationale training sets on the data server.

The teacher's internal reasoning is never a training target. Only its final,
structured rationale, independently reviewed in a second call, is retained.
A transient or malformed response stops this resumable run instead of excluding
an example. summary.json is published last and is the only completion marker.
"""
import argparse
import base64
import collections
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import inspect
import json
import os
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.request

from cot_protocol import neutral_instruction
from distill_thinking import Rejected, SOURCE_SHA, atomic_json, digest, key_for

TEST_SHA = "6eddbabedff8e37df2f5629585b4947fea3cbea6df9f61d9a1fed13548b8249e"
EXPECTED_ROWS, EXPECTED_UNIQUE = 4657, 2948
MIN_CHARS, MAX_CHARS = 80, 1000
# These rules remove answer-conditioned/meta narration, not medically meaningful
# language about uncertainty. Model review remains essential for visual support.
META = re.compile(
    r"\b(reference|ground[ -]?truth|annotation|provided label|given label|supplied label|"
    r"target label|training label|supervision|prompt|json|schema|task instructions?|"
    r"need to (?:output|return)|must (?:output|return))\b", re.I)
SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["accept", "revised", "reject"]},
    "rationale": {"type": "string"}, "final_answer": {"type": "object"}},
    "required": ["verdict", "rationale", "final_answer"], "additionalProperties": False}
RULES = (
    "Inspect this single public mammogram. The fixed final_answer below is a training "
    "label, not evidence that an invisible feature is present. Return only structured JSON "
    "with verdict, rationale, final_answer. Keep final_answer exactly unchanged. "
    "Your final rationale must be 80 to 1000 characters in English, preferably 2 to 4 short "
    "sentences: describe only visible breast density, visible findings and their location, "
    "and a cautious imaging interpretation. Do not invent history, prior examinations, "
    "modality findings, lesion details, or reasons unsupported by the image. Express "
    "uncertainty when warranted. Do not mention reference labels, annotations, supplied "
    "answers, prompts, JSON, supervision, or plans for completing the task in the rationale. "
    "Do not merely paraphrase the final labels. Your private reasoning may be longer; "
    "only the short final rationale will be used for training. If a grounded explanation "
    "consistent with the fixed labels is impossible, use verdict reject and rationale ''. "
)
GENERATE = RULES + "Use verdict accept or reject."
REVIEW = RULES + (
    "Independently inspect the image before judging the candidate below. Verify every "
    "visual claim, remove unsupported detail, and fix overconfidence. Use accept only if "
    "the candidate is already supported, revised with a replacement rationale if repairable, "
    "or reject if it cannot be made grounded. A rejected first draft can be replaced. "
    "Do not approve based on agreement with the fixed labels alone."
)


def validate_value(value, reference, stage):
    if not isinstance(value, dict) or set(value) != set(SCHEMA["required"]):
        raise Rejected("INVALID_RESPONSE_SCHEMA")
    allowed = ("accept", "reject") if stage == "generate" else ("accept", "revised", "reject")
    if value["verdict"] not in allowed:
        raise Rejected("INVALID_VERDICT")
    if value["final_answer"] != json.loads(reference):
        raise Rejected("FINAL_LABEL_MISMATCH")
    rationale = value["rationale"]
    if not isinstance(rationale, str):
        raise Rejected("INVALID_RATIONALE")
    rationale = rationale.strip()
    if value["verdict"] != "reject":
        if not MIN_CHARS <= len(rationale) <= MAX_CHARS:
            raise Rejected("RATIONALE_LENGTH")
        if "<think>" in rationale or "</think>" in rationale or META.search(rationale):
            raise Rejected("RATIONALE_META_TEXT")
    elif rationale:
        raise Rejected("REJECT_MUST_HAVE_EMPTY_RATIONALE")
    return {**value, "rationale": rationale}


def unpack(reply, reference, stage):
    try:
        choice = reply["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise Rejected("INCOMPLETE_GENERATION")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise Rejected("INVALID_RESPONSE_SCHEMA")
        # Some OpenAI-compatible deployments expose reasoning in-band. Never use
        # it as rationale; accept only the complete JSON after its closing tag.
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        value = json.loads(content.strip())
        return validate_value(value, reference, stage)
    except (KeyError, IndexError, TypeError, ValueError):
        raise Rejected("INVALID_RESPONSE_SCHEMA") from None


def image_path(args, row):
    path = (args.image_root / row["images"][0]).resolve()
    if not path.is_relative_to((args.image_root / "images_png").resolve()):
        raise Rejected("IMAGE_PATH_OUTSIDE_DATASET")
    if not path.is_file() or not os.access(path, os.R_OK):
        raise Rejected("IMAGE_NOT_READABLE")
    return path


def request_stage(args, row, stage, candidate=None):
    prompt = (GENERATE if stage == "generate" else REVIEW)
    prompt += "\nFIXED FINAL ANSWER:\n" + row["output"]
    if candidate is not None:
        prompt += "\nCANDIDATE:\n" + json.dumps(candidate, ensure_ascii=False)
    body = {
        "model": args.model, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                base64.b64encode(image_path(args, row).read_bytes()).decode()}},
            {"type": "text", "text": prompt}]}],
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": 4096, "temperature": 0.2, "top_p": 0.95, "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "short_grounded_rationale", "strict": True, "schema": SCHEMA}},
    }
    # Two retries, three calls maximum per stage. Do not convert any transport,
    # format, truncation or label mismatch error into a semantic exclusion.
    for attempt in range(3):
        request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                reply = json.load(response)
            return unpack(reply, row["output"], stage)
        except urllib.error.HTTPError as error:
            code = "HTTP_" + str(error.code)
            retryable = error.code == 429 or 500 <= error.code < 600
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            code, retryable = "NETWORK_ERROR", True
        except Rejected as error:
            code, retryable = str(error), True
        except (ValueError, TypeError, KeyError):
            code, retryable = "INVALID_RESPONSE_SCHEMA", True
        if not retryable or attempt == 2:
            raise Rejected(code) from None
        if code == "INCOMPLETE_GENERATION" and getattr(args, "retry_truncation", False):
            body["max_tokens"] = min(body["max_tokens"] * 2, 16384)
            # A second truncation can be a runaway internal reasoning sequence.
            # Request the same short structured rationale without internal thinking
            # on the final bounded retry; target validation and review are unchanged.
            if attempt == 1:
                body["chat_template_kwargs"]["enable_thinking"] = False
                body["max_tokens"] = 4096
            receipt_dir = args.output_dir / "retry_metadata"
            receipt_dir.mkdir(exist_ok=True)
            atomic_json(receipt_dir / (key_for(row) + "." + stage + ".json"), {
                "code": code, "next_attempt": attempt + 2,
                "max_tokens": body["max_tokens"],
                "teacher_enable_thinking": body["chat_template_kwargs"]["enable_thinking"],
                "implementation_sha256": digest(Path(__file__).read_bytes()),
                "timestamp": time.time()})
        time.sleep(5)


def build_arms(rows, decisions):
    expected = {key_for(row) for row in rows}
    if set(decisions) != expected:
        raise Rejected("INCOMPLETE_DECISIONS")
    direct, cot, excluded = [], [], []
    for index, row in enumerate(rows):
        key = key_for(row)
        decision = validate_value(decisions[key], row["output"], "review")
        if decision["verdict"] == "reject":
            excluded.append({"source_index": index, "pair_sha256": key})
            continue
        baseline = {**row, "instruction": neutral_instruction(row["instruction"])}
        direct.append(baseline)
        cot.append({**baseline, "output": "<think>" + decision["rationale"] + "</think>\n" + row["output"]})
    if not direct:
        raise Rejected("NO_RETAINED_SAMPLES")
    if len(direct) != len(cot) or any(
        a["images"] != b["images"] or a["instruction"] != b["instruction"] or
        b["output"].split("</think>\n", 1)[1] != a["output"] for a, b in zip(direct, cot)):
        raise Rejected("ARM_ALIGNMENT_FAILED")
    return direct, cot, excluded


def cached_stage(args, key, row, stage, candidate=None):
    path = args.output_dir / "cache" / (key + "." + stage + ".json")
    candidate_sha = digest(json.dumps(candidate, sort_keys=True).encode()) if candidate is not None else None
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("pair_sha256") != key or cached.get("candidate_sha256") != candidate_sha:
            raise Rejected("CACHE_PROVENANCE_MISMATCH")
        return validate_value(cached["value"], row["output"], stage)
    value = request_stage(args, row, stage, candidate)
    atomic_json(path, {"pair_sha256": key, "candidate_sha256": candidate_sha, "value": value})
    return value


def prepared_pairs(args, unique):
    """Yield completed pairs with at most workers futures outstanding.

    Only a worker's own per-pair cache is mutated concurrently. The caller owns
    all aggregate counters and publication. On failure, pending work is cancelled
    and already running calls can finish writing their resumable caches.
    """
    workers = args.workers
    if type(workers) is not int or not 1 <= workers <= 8:
        raise Rejected("INVALID_WORKER_COUNT")

    def prepare(key, row):
        draft = cached_stage(args, key, row, "generate")
        reviewed = cached_stage(args, key, row, "review", draft)
        return key, draft, reviewed

    remaining = iter(unique.items())
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = set()
    try:
        for _ in range(workers):
            item = next(remaining, None)
            if item is not None:
                pending.add(executor.submit(prepare, *item))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Resolve the entire completed batch before scheduling anything new;
            # one failing future stops publication even if others succeeded.
            results = [future.result() for future in done]
            for result in results:
                yield result
            for _ in results:
                item = next(remaining, None)
                if item is not None:
                    pending.add(executor.submit(prepare, *item))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def execute(args):
    if type(args.workers) is not int or not 1 <= args.workers <= 8:
        raise Rejected("INVALID_WORKER_COUNT")
    source_blob, test_blob = args.source.read_bytes(), args.test.read_bytes()
    if digest(source_blob) != SOURCE_SHA or digest(test_blob) != TEST_SHA:
        raise Rejected("SOURCE_VERSION_MISMATCH")
    rows, test = json.loads(source_blob), json.loads(test_blob)
    unique = {}
    for row in rows:
        if len(row["images"]) != 1 or row["instruction"].count("<image>") != 1:
            raise Rejected("INVALID_INPUT_SCHEMA")
        unique.setdefault(key_for(row), row)
    if len(rows) != EXPECTED_ROWS or len(unique) != EXPECTED_UNIQUE:
        raise Rejected("UNEXPECTED_INPUT_COUNTS")
    train_images = {str(image_path(args, row)) for row in unique.values()}
    test_images = {str((args.image_root / image).resolve()) for row in test for image in row["images"]}
    if train_images & test_images:
        raise Rejected("TRAIN_TEST_OVERLAP")
    manifest = {"source_sha256": SOURCE_SHA, "test_sha256": TEST_SHA,
        "source_rows": len(rows), "unique_pairs": len(unique), "test_rows": len(test),
        "base_url": args.base_url, "model": args.model, "max_tokens": 4096,
        "temperature": 0.2, "top_p": 0.95, "teacher_enable_thinking": True,
        "rationale_limit_unit": "characters", "rationale_min": MIN_CHARS, "rationale_max": MAX_CHARS,
        "prompt_sha256": digest((GENERATE + REVIEW + json.dumps(SCHEMA, sort_keys=True)).encode()),
        "neutral_instruction_sha256": digest(inspect.getsource(neutral_instruction).encode()),
        "git_commit": args.git_commit, "protocol_version": 1}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise Rejected("RESUME_CONFIG_MISMATCH")
    atomic_json(manifest_path, manifest)
    (args.output_dir / "cache").mkdir(exist_ok=True)
    decisions, generated, started = {}, collections.Counter(), time.monotonic()
    print(json.dumps({"phase": "start", "source_rows": len(rows), "unique_total": len(unique),
                      "workers": args.workers}), flush=True)
    for key, draft, reviewed in prepared_pairs(args, unique):
        generated[draft["verdict"]] += 1
        decisions[key] = reviewed
        if decisions:
            progress = {"phase": "reviewing", "reviewed_unique": len(decisions),
                "unique_total": len(unique), "review_verdict_counts": dict(collections.Counter(
                    d["verdict"] for d in decisions.values())),
                "elapsed_this_run_s": round(time.monotonic() - started, 1), "workers": args.workers}
            atomic_json(args.output_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    direct, cot, excluded = build_arms(rows, decisions)
    info = {name: {"file_name": filename, "columns": {
        "prompt": "instruction", "response": "output", "images": "images"}}
        for name, filename in (("fair_direct", "direct_train.json"), ("fair_cot", "cot_train.json"))}
    outputs = {}
    for name, filename, value in (("direct_train", "direct_train.json", direct),
            ("cot_train", "cot_train.json", cot), ("dataset_info", "dataset_info.json", info),
            ("excluded", "excluded.json", excluded)):
        path = args.output_dir / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise Rejected("EXISTING_OUTPUT_MISMATCH")
        atomic_json(path, value)
        outputs[name] = {"path": str(path), "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
    lengths = [len(d["rationale"]) for d in decisions.values() if d["verdict"] != "reject"]
    summary = {**manifest, "state": "complete", "output_rows": len(direct),
        "accepted_unique": len(lengths), "excluded_unique": len(unique) - len(lengths),
        "excluded_rows": len(excluded), "generation_verdict_counts": dict(generated),
        "review_verdict_counts": dict(collections.Counter(d["verdict"] for d in decisions.values())),
        "train_test_overlap": 0, "all_retained_pairs_model_reviewed": True,
        "all_samples_model_reviewed": True, "clinically_validated": False,
        "final_labels_preserved": True, "sampling_order_and_multiplicity_preserved": True,
        "rationale_chars": {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 1)},
        "outputs": outputs, "workers_at_completion": args.workers}
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {"phase": "complete", "output_rows": len(direct),
                                                  "reviewed_unique": len(decisions)})
    print(json.dumps({"phase": "complete", "output_rows": len(direct),
                      "accepted_unique": len(lengths), "excluded_rows": len(excluded)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/mammo"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://10.222.10.107:9241/v1")
    parser.add_argument("--model", default="qwen38-flash-next")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retry-truncation", action="store_true",
                        help="Retry truncation at 8192, then direct structured response at 4096; preserve target limits")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4,
                        help="Concurrent independent pairs; operational only, may change on resume")
    parser.add_argument("--git-commit", default="unspecified")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Rejected("PREPARATION_ALREADY_ACTIVE") from None
        try:
            execute(args)
        except Rejected as error:
            atomic_json(args.output_dir / "progress.json", {"phase": "failed", "code": str(error)})
            raise


if __name__ == "__main__":
    try:
        main()
    except Rejected as error:
        print(json.dumps({"phase": "failed", "code": str(error)}), flush=True)
        raise SystemExit(1)
    except Exception as error:
        print(json.dumps({"phase": "failed", "code": "INTERNAL_" + type(error).__name__}), flush=True)
        raise SystemExit(1)
