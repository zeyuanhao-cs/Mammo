# Paired dataset acceptance

- Preparation job 12556: COMPLETED after resumable attempts 12524 and 12551.
- Reviewed all 2,948 unique pairs: 2,679 accepted, 250 revised, 19 rejected.
- Both training arms contain 4,632 rows and 2,929 unique images. The same 25 repeated/source rows were excluded from both arms.
- Server-side checks passed for every output hash, image order, sampling multiplicity, identical final labels, and zero train/test image overlap.
- Final rationale length: 155–647 characters, mean 340.2 (unique retained pairs).
- Teacher retry metadata contains 21 pair/stage records, not a count of all API calls. Twenty legacy records do not carry an explicit thinking-mode field. One record schedules the final retry with internal thinking disabled after repeated truncation. Short structured rationale and independent review requirements remain unchanged. These metadata describe retry policy, not a causal effect of teacher thinking mode.
- `summary.json` preserves the original preparation manifest identity and initial teacher defaults. Recovery implementations are `1e06406` and `b85fc4a`; the latter uses 4096 thinking → 8192 thinking → 4096 non-thinking only after repeated truncation. Do not interpret the initial manifest defaults as the settings of every retry.
- No manual clinical review was performed. Model review is not clinical validation.

Training/inference runner 12559 was submitted on H200 physical GPU1 with code `05be86ce062e59a3a289efa4fa84a6b37e055912`. Runtime preflight passed; protocol fingerprint `f4d14c2828b0f32cd655cdf84263f0578c73e9095e3fb3b31e6b3aebd2e2bf75`. First stage: seed42 direct training. Full A–D results are pending.
