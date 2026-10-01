#!/usr/bin/env python3
"""Re-reason flagged samples on the server and publish a separate reviewed dataset."""
import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from distill_thinking import (Rejected, SOURCE_SHA, atomic_json, digest, key_for,
                             thinking_instruction)

PROMPT = (
    "Re-evaluate this public mammogram independently of the previous rationale. "
    "The annotation is fixed supervision, not proof of visual features. Inspect the "
    "original image and scrutinize the previous rationale for unsupported visual "
    "details, invented history, or contradictions. Distinguish what this single image "
    "shows, what is given only by the annotation, and what is uncertain. Remove or "
    "qualify unverified assertions. Do not infer patient facts or prior examinations. "
    "Return verdict accept if the previous rationale is sound, revised if you can "
    "repair it, or reject if you cannot produce a grounded, label-consistent rationale. "
    "Provide a concise replacement rationale in English suitable for training, rather "
    "than commentary about reviewing. Never change the reference final_answer. "
    "You must output only a JSON object with verdict, issue_codes, rationale, and "
    "final_answer, obeying the following schema: "
)
ISSUES = ["UNSUPPORTED_VISUAL_CLAIM", "FABRICATED_HISTORY", "LABEL_CONFLICT",
          "OVERCONFIDENT_DIAGNOSIS", "LABEL_REPETITION_ONLY", "OTHER"]
SCHEMA = {
    "type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revised", "reject"]},
        "issue_codes": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
        "rationale": {"type": "string"},
        "final_answer": {"type": "object"},
    }, "required": ["verdict", "issue_codes", "rationale", "final_answer"],
    "additionalProperties": False,
}


def validate(reply, reference):
    choice = reply["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise Rejected("REVIEW_TRUNCATED")
    value = json.loads(choice["message"]["content"])
    if set(value) != set(SCHEMA["required"]):
        raise Rejected("INVALID_REVIEW_SCHEMA")
    if value["verdict"] not in ("accept", "revised", "reject"):
        raise Rejected("INVALID_REVIEW_VERDICT")
    if not isinstance(value["issue_codes"], list) or any(x not in ISSUES for x in value["issue_codes"]):
        raise Rejected("INVALID_REVIEW_ISSUE_CODE")
    if value["final_answer"] != json.loads(reference):
        raise Rejected("REVIEW_LABEL_CONFLICT")
    rationale = value["rationale"]
    if not isinstance(rationale, str):
        raise Rejected("INVALID_REVIEW_RATIONALE")
    if value["verdict"] != "reject" and (
        len(rationale.strip()) < 80 or "<think>" in rationale or "</think>" in rationale
    ):
        raise Rejected("INVALID_REVIEW_RATIONALE")
    return value


def review(args, row, cached):
    image = (args.image_root / row["images"][0]).resolve()
    if not image.is_relative_to((args.image_root / "images_png").resolve()):
        raise Rejected("IMAGE_PATH_OUTSIDE_DATASET")
    prompt = (PROMPT + json.dumps(SCHEMA) + "\nREFERENCE ANNOTATION:\n" + row["output"] +
              "\nPREVIOUS RATIONALE:\n" + cached["reasoning"])
    body = {
        "model": args.model, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
             base64.b64encode(image.read_bytes()).decode()}}, {"type": "text", "text": prompt}]}],
        "chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 6144,
        "temperature": 0.2, "top_p": 0.95, "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "rationale_review", "strict": True, "schema": SCHEMA}},
    }
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                reply = json.load(response)
            value = validate(reply, row["output"])
            return {"review": value, "response": reply,
                    "original_rationale_sha256": digest(cached["reasoning"].encode())}
        except urllib.error.HTTPError as error:
            code = "HTTP_" + str(error.code)
            retry = error.code == 429 or 500 <= error.code < 600
        except (urllib.error.URLError, TimeoutError):
            code, retry = "NETWORK_ERROR", True
        except (KeyError, TypeError, ValueError):
            code, retry = "INVALID_REVIEW_RESPONSE", True
        except Rejected as error:
            code, retry = str(error), True
        if not retry or attempt == 1:
            raise Rejected(code) from None
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/mammo"))
    parser.add_argument("--model", default="qwen38-flash-next")
    parser.add_argument("--base-url", default="http://10.222.10.107:9241/v1")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--await-final-dataset", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "review.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Rejected("REVIEW_ALREADY_ACTIVE") from None
        execute(args)


def execute(args):
    blob = args.train.read_bytes()
    if digest(blob) != SOURCE_SHA:
        raise Rejected("SOURCE_VERSION_MISMATCH")
    rows = json.loads(blob)
    inputs = {key_for(row): row for row in rows}
    audit_blob = args.audit.read_bytes()
    audits = json.loads(audit_blob)["samples"]
    # A negative follow-up does not erase an earlier risk flag.
    positions = {a["cache_position"] for a in audits if any(
        a.get(name, {}).get("recommended_review", False)
        for name in ("model_audit", "review_followup"))}
    candidates = {}
    with (args.source_run / "responses.jsonl").open() as stream:
        for position, line in enumerate(stream, 1):
            if position in positions:
                cached = json.loads(line)
                if cached["key"] not in inputs:
                    raise Rejected("AUDIT_CACHE_MISMATCH")
                candidates[cached["key"]] = cached
            if positions and position >= max(positions):
                break
    if len(candidates) != len(positions):
        raise Rejected("AUDIT_CACHE_MISMATCH")
    manifest = {"source_run": str(args.source_run), "source_sha256": SOURCE_SHA,
                "audit_sha256": digest(audit_blob), "flagged_unique": len(candidates),
                "candidate_hashes": {k: digest(v["reasoning"].encode()) for k, v in candidates.items()},
                "model": args.model, "base_url": args.base_url, "enable_thinking": True,
                "prompt_sha256": digest((PROMPT + json.dumps(SCHEMA)).encode()),
                "git_commit": args.git_commit, "max_tokens": 6144,
                "scope": "flagged samples only; same model with a separate review prompt"}
    path = args.output_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise Rejected("REVIEW_RESUME_CONFIG_MISMATCH")
    atomic_json(path, manifest)
    decisions = {}
    journal = args.output_dir / "reviews.jsonl"
    if journal.exists():
        with journal.open("rb+") as stream:
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
                if key not in candidates or key in decisions:
                    raise Rejected("INVALID_REVIEW_CACHE")
                if cached.get("original_rationale_sha256") != digest(candidates[key]["reasoning"].encode()):
                    raise Rejected("REVIEW_SOURCE_HASH_MISMATCH")
                if "response" in cached:
                    if validate(cached["response"], inputs[key]["output"]) != cached["review"]:
                        raise Rejected("REVIEW_CACHE_MISMATCH")
                elif cached.get("review", {}).get("verdict") != "reject" or not cached.get("error_code"):
                    raise Rejected("INVALID_REVIEW_CACHE")
                decisions[key] = cached
    with journal.open("a") as stream:
        for key, cached in candidates.items():
            if key in decisions:
                continue
            try:
                result = review(args, inputs[key], cached)
            except Rejected as error:
                result = {"review": {"verdict": "reject", "issue_codes": [],
                          "rationale": "", "final_answer": json.loads(inputs[key]["output"])},
                          "error_code": str(error),
                          "original_rationale_sha256": digest(cached["reasoning"].encode())}
            result["key"] = key
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            decisions[key] = result
            print(json.dumps({"phase": "reviewed", "reviewed_unique": len(decisions),
                              "flagged_unique": len(candidates), "verdict": result["review"]["verdict"],
                              "issue_codes": result["review"]["issue_codes"],
                              "error_code": result.get("error_code")}), flush=True)
    counts = {v: sum(d["review"]["verdict"] == v for d in decisions.values())
              for v in ("accept", "revised", "reject")}
    summary = {**manifest, "state": "review_complete", "verdict_counts": counts}
    atomic_json(args.output_dir / "review_summary.json", summary)
    if not args.await_final_dataset:
        return
    print(json.dumps({"phase": "awaiting_source_dataset", "verdict_counts": counts}), flush=True)
    while not (args.source_run / "summary.json").exists():
        progress_path = args.source_run / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("phase") in ("partial", "failed", "stopped_consecutive_errors"):
                raise Rejected("SOURCE_DISTILLATION_INCOMPLETE")
            if time.time() - progress_path.stat().st_mtime > 900:
                raise Rejected("SOURCE_DISTILLATION_STALLED")
        time.sleep(30)
    source_summary = json.loads((args.source_run / "summary.json").read_text())
    if source_summary.get("state") != "complete" or source_summary.get("output_rows") != len(rows):
        raise Rejected("SOURCE_DISTILLATION_INCOMPLETE")
    source_file = args.source_run / "train_balanced_2to1_thinking.json"
    if digest(source_file.read_bytes()) != source_summary["output_sha256"]:
        raise Rejected("SOURCE_DATASET_HASH_MISMATCH")
    source_data = json.loads(source_file.read_text())
    if len(source_data) != len(rows):
        raise Rejected("SOURCE_DATASET_COUNT_MISMATCH")
    accepted, excluded = [], []
    for row, distilled in zip(rows, source_data):
        key = key_for(row)
        if distilled["images"] != row["images"] or not distilled["output"].endswith("</think>\n" + row["output"]):
            raise Rejected("SOURCE_DATASET_LABEL_MISMATCH")
        decision = decisions.get(key)
        if decision:
            if decision["review"]["verdict"] == "reject":
                excluded.append({"key": key, "error_code": decision.get("error_code"),
                                 "issue_codes": decision["review"]["issue_codes"]})
                continue
            distilled = {"images": row["images"], "instruction": thinking_instruction(row["instruction"]),
                         "output": "<think>" + decision["review"]["rationale"].strip() + "</think>\n" + row["output"]}
        accepted.append(distilled)
    out = args.output_dir / "train_balanced_2to1_thinking.json"
    if out.exists() and json.loads(out.read_text()) != accepted:
        raise Rejected("REVIEWED_OUTPUT_ALREADY_EXISTS")
    if not out.exists():
        atomic_json(out, accepted)
    atomic_json(args.output_dir / "excluded.json", excluded)
    atomic_json(args.output_dir / "dataset_info.json", {"vindr_balanced_thinking_train": {
        "file_name": out.name, "columns": {"prompt": "instruction", "response": "output", "images": "images"}}})
    atomic_json(args.output_dir / "summary.json", {
        **summary, "state": "complete", "output_rows": len(accepted), "excluded_rows": len(excluded),
        "output_sha256": digest(out.read_bytes()), "final_labels_preserved": True,
        "all_samples_model_reviewed": False, "clinically_validated": False})
    print(json.dumps({"phase": "complete", "output_rows": len(accepted),
                      "excluded_rows": len(excluded), "verdict_counts": counts}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Rejected as error:
        print(json.dumps({"phase": "failed", "code": str(error)}), flush=True)
        raise SystemExit(1)
    except Exception as error:
        print(json.dumps({"phase": "failed", "code": "INTERNAL_" + type(error).__name__}), flush=True)
        raise SystemExit(1)
