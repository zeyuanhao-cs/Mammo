# Fair CoT experiment (2026-10-04)

Status: protocol and executable pipeline; results require terminal receipts.

## Frozen comparison

- Base model: Qwen3.5-4B; paired seeds 42, 43, 44.
- Source training data: `train_balanced_2to1.json` (4,657 rows, 2,948 unique image/label pairs).
- Evaluation: exactly baseline indices 0–499 of `direct_test.json`. This enriched subset is exploratory and is not representative of all 4,000 test images.
- Teacher creates a short image-grounded rationale, then independently reviews every pair against the original image and label. Only the final short rationale is retained; internal teacher reasoning is discarded.
- Rationales are 80–1,000 characters. A semantic rejection excludes the same pair and all its repeated rows from both training arms. Transport/schema failures stop preparation resumably; they never silently exclude examples.
- Both arms preserve identical images, final labels, neutral instructions, order, and sampling multiplicity. Model review is not clinical validation.

| Arm | Training target | Inference thinking |
|---|---|---|
| A | Final JSON | Off |
| B | Final JSON | On |
| C | Short rationale + same final JSON | Off |
| D | Short rationale + same final JSON | On |

All arms use 2 epochs, effective batch size 16, learning rate 5e-5, LoRA rank 8 / alpha 16 / dropout 0.05, cutoff 8,192, max image pixels 786,432, bf16 and SDPA. Ordinary response-token SFT is used; no additional final-answer weighting is introduced. Inference uses greedy decoding and the same 4,096-token limit. Changing the target changes token allocation and compute; measured cost is reported, not assumed equal.

Primary comparison: D minus A. B minus A and D minus C estimate inference-mode changes conditional on the trained model; B/C are cross-mode diagnostics. C minus A compares supervision with thinking disabled. Report all arms and all three seeds, mean and sample standard deviation, without selecting the best seed. Historical 4B/9B scores are references, not matched causal controls.

Invalid, truncated and runtime-error predictions remain in the 500-case denominator. Report JSON validity, breast BI-RADS/density, category/localized/joint finding metrics, generated tokens, elapsed time and truncation. Finding BI-RADS accuracy must include its matched support. Patient/study bootstrap intervals are allowed only when a verified grouping map is available; otherwise omit them.

## Execution and completion

1. `prepare_cot_fair.py`: resumable, hash-bound CPU preparation with atomic per-pair generation/review caches and final `summary.json`.
2. `run_cot_fair.py`: paired training and A–D inference, validated checkpoint resume and reservation deadline guard.
3. `evaluate_cot_fair.py`: exact coverage/provenance checks and aggregate `comparison.json` / `comparison.md`.

Completion requires all six final adapters and all twelve 500-case evaluation cells. A running job, partial table or exit code alone is not completion. Raw images, teacher responses, training records and predictions remain on the server. Re-evaluation of the old adapter with its training-consistent prompt is saved separately as a diagnostic.
