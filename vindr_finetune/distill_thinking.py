#!/usr/bin/env python3
"""Resume label-conditioned Flash distillation; raw replies stay in the run directory."""
import argparse
import base64
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.request

SOURCE_SHA = "93d53adefac511f8d355e5dc72ead78526a54e1c2e1d55a6aba0890bc1584ce9"
PROMPT = (
    "Analyze this public VinDr-Mammo mammogram using the reference annotation below. "
    "Provide a concise, image-grounded diagnostic rationale in English: discuss breast "
    "density, visible findings and location, and BI-RADS. The reference is supervision, "
    "not proof of features invisible in the image; explicitly acknowledge uncertainty "
    "and do not invent clinical history or unobservable evidence. Then output only the "
    "reference JSON as the final answer. Reference annotation: "
)


class Rejected(Exception):
    """Closed error code, never raw response text."""


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def key_for(row):
    return digest(json.dumps([row["images"], json.loads(row["output"])],
                             sort_keys=True, separators=(",", ":")).encode())


def unpack(result, reference):
    choice = result["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise Rejected("INCOMPLETE_GENERATION")
    message = choice["message"]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    if not reasoning:
        match = re.match(r"\s*<think>(.*?)</think>\s*(.*)", content, re.S)
        if match:
            reasoning, content = match.groups()
    reasoning = reasoning.strip()
    if len(reasoning) < 80 or "<think>" in reasoning or "</think>" in reasoning:
        raise Rejected("INVALID_REASONING")
    final = content.strip()
    if final.startswith("```"):
        final = re.sub(r"^```(?:json)?\s*|\s*```$", "", final).strip()
    try:
        parsed = json.loads(final)
    except ValueError:
        # Some compatible servers put a short explanation before the final JSON.
        # Accept only one schema object at the end, never an echoed label amid text.
        candidates = []
        decoder = json.JSONDecoder()
        for pos, char in enumerate(final):
            if char != "{":
                continue
            try:
                obj, length = decoder.raw_decode(final[pos:])
            except ValueError:
                continue
            if isinstance(obj, dict) and set(obj) == {"breast_birads", "breast_density", "findings"}:
                candidates.append((obj, final[pos + length:].strip()))
        if len(candidates) != 1 or candidates[0][1] not in ("", "```"):
            raise Rejected("INVALID_FINAL_JSON") from None
        parsed = candidates[0][0]
    if parsed != json.loads(reference):
        raise Rejected("REFERENCE_MISMATCH")
    return reasoning


def request_teacher(args, row):
    image = (args.image_root / row["images"][0]).resolve()
    if not image.is_relative_to((args.image_root / "images_png").resolve()):
        raise Rejected("IMAGE_PATH_OUTSIDE_DATASET")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}},
            {"type": "text", "text": PROMPT + row["output"]}]}],
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": args.max_tokens, "temperature": 0.6, "top_p": 0.95, "stream": False,
    }
    request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    for attempt in range(2):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.load(response)
            reasoning = unpack(result, row["output"])
            return {"reasoning": reasoning, "response": result,
                    "elapsed_s": round(time.monotonic() - started, 2)}
        except urllib.error.HTTPError as error:
            code = "HTTP_" + str(error.code)
            retry = error.code == 429 or 500 <= error.code < 600
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            code, retry = "NETWORK_ERROR", True
        except (KeyError, IndexError, TypeError, ValueError):
            raise Rejected("INVALID_RESPONSE_SCHEMA") from None
        if not retry or attempt == 1:
            raise Rejected(code) from None
        time.sleep(10)


def thinking_instruction(original):
    text = original.replace("Analyze the mammography image and return ONLY valid JSON.",
                            "Analyze the mammography image. Explain your image-grounded "
                            "reasoning in <think>...</think>, then give the final JSON.")
    text = text.replace("Return ONLY a JSON object with exactly these keys:",
                        "After reasoning, return a JSON object with exactly these keys:")
    return text.replace("Do not output explanations, markdown, code fences, or extra text.",
                        "Keep reasoning inside <think>...</think>. Outside it, output only "
                        "the final JSON, without markdown or code fences.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/mammo"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://10.222.10.107:9241/v1")
    parser.add_argument("--model", default="qwen38-flash-next")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; never publish a partial dataset")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # The lock persists for the entire run, including resume, to prevent duplicate clients.
    lock = (args.run_dir / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise Rejected("RUN_ALREADY_ACTIVE") from None
    blob = args.train.read_bytes()
    if digest(blob) != SOURCE_SHA:
        raise Rejected("SOURCE_VERSION_MISMATCH")
    rows = json.loads(blob)
    unique = {}
    for row in rows:
        if len(row["images"]) != 1 or row["instruction"].count("<image>") != 1:
            raise Rejected("INVALID_INPUT_SCHEMA")
        unique.setdefault(key_for(row), row)
    test_images = {im for row in json.loads(args.test.read_text()) for im in row["images"]}
    if {im for row in rows for im in row["images"]} & test_images:
        raise Rejected("TRAIN_TEST_OVERLAP")
    if len(rows) != 4657 or len(unique) != 2948:
        raise Rejected("UNEXPECTED_INPUT_COUNTS")
    for row in unique.values():
        image = (args.image_root / row["images"][0]).resolve()
        if not image.is_relative_to((args.image_root / "images_png").resolve()):
            raise Rejected("IMAGE_PATH_OUTSIDE_DATASET")
        if not image.is_file() or not os.access(image, os.R_OK):
            raise Rejected("IMAGE_NOT_READABLE")
    manifest = {"source_sha256": SOURCE_SHA, "source_rows": len(rows), "unique_pairs": len(unique),
                "test_sha256": digest(args.test.read_bytes()), "test_overlap": 0,
                "base_url": args.base_url, "model": args.model, "enable_thinking": True,
                "max_tokens": args.max_tokens, "temperature": 0.6, "top_p": 0.95,
                "prompt_sha256": digest(PROMPT.encode()), "git_commit": args.git_commit,
                "supervision": "training images and ground-truth labels; label-conditioned rationale"}
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise Rejected("RESUME_CONFIG_MISMATCH")
    atomic_json(manifest_path, manifest)
    cache = args.run_dir / "responses.jsonl"
    successes = {}
    # Repair only an interrupted last line; complete malformed lines are an error.
    if cache.exists():
        with cache.open("rb+") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                cached = json.loads(line)
                key = cached["key"]
                if key not in unique or key in successes:
                    raise Rejected("INVALID_CACHE_KEY")
                unpack(cached["response"], unique[key]["output"])
                successes[key] = cached
    errors = collections.Counter()
    tokens = collections.Counter()
    started = time.monotonic()
    targets = list(unique.items())[:args.limit or None]
    consecutive = attempted = 0
    print(json.dumps({"phase": "start", "source_rows": len(rows), "unique_pairs": len(unique),
                      "cached": len(successes), "targets": len(targets)}), flush=True)

    def progress(phase):
        value = {"phase": phase, "successful_unique": len(successes), "unique_total": len(unique),
                 "attempted_this_run": attempted, "errors_this_run": dict(errors),
                 "elapsed_s": round(time.monotonic() - started, 1), "usage_this_run": dict(tokens)}
        atomic_json(args.run_dir / "progress.json", value)
        print(json.dumps(value), flush=True)

    with cache.open("a") as stream, (args.run_dir / "errors.jsonl").open("a") as failures:
        for key, row in targets:
            if key in successes:
                continue
            attempted += 1
            try:
                result = request_teacher(args, row)
                result["key"] = key
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                successes[key] = result
                consecutive = 0
                for name, count in result["response"].get("usage", {}).items():
                    if isinstance(count, int):
                        tokens[name] += count
            except Rejected as error:
                code = str(error)
                errors[code] += 1
                consecutive += 1
                failures.write(json.dumps({"key": key, "code": code}) + "\n")
                failures.flush()
            if attempted == 1 or attempted % 10 == 0:
                progress("running")
            if consecutive >= 10:
                progress("stopped_consecutive_errors")
                raise Rejected("TEACHER_VALIDATION_FAILED")
    if len(successes) != len(unique):
        progress("smoke_complete" if args.limit else "partial")
        if not args.limit:
            raise Rejected("INCOMPLETE_DISTILLATION")
        return
    dataset = []
    for row in rows:
        reasoning = successes[key_for(row)]["reasoning"]
        dataset.append({"instruction": thinking_instruction(row["instruction"]),
                        "images": row["images"],
                        "output": "<think>" + reasoning + "</think>\n" + row["output"]})
    output = args.run_dir / "train_balanced_2to1_thinking.json"
    # Completed outputs are immutable, including on resume.
    if output.exists():
        if json.loads(output.read_text()) != dataset:
            raise Rejected("OUTPUT_ALREADY_EXISTS")
    else:
        atomic_json(output, dataset)
    atomic_json(args.run_dir / "dataset_info.json", {"vindr_balanced_thinking_train": {
        "file_name": output.name, "columns": {
            "prompt": "instruction", "response": "output", "images": "images"}}})
    atomic_json(args.run_dir / "summary.json", {
        **manifest, "state": "complete", "output_rows": len(dataset),
        "output_sha256": digest(output.read_bytes()),
        "final_labels_preserved": True, "reasoning_is_teacher_generated_not_clinically_validated": True})
    progress("complete")


if __name__ == "__main__":
    try:
        main()
    except Rejected as error:
        print(json.dumps({"phase": "failed", "code": str(error)}), flush=True)
        raise SystemExit(1)
    except Exception as error:
        # Do not expose response bodies, data, or unrestricted traceback in Slurm output.
        print(json.dumps({"phase": "failed", "code": "INTERNAL_" + type(error).__name__}), flush=True)
        raise SystemExit(1)
