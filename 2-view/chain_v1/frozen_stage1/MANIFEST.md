# 阶段一评测代码冻结 (初冻 2026-10-24 16:47:17；16:59 修正 quick_analyze 的 shape_q 变量)

- git rev: c9e83405
- 冻结目录: chain_v1/frozen_stage1/（与 live 逐字节一致）
- 17:20 追加：soft_decode 支持 force_p+force_hard 时 token 跟随强制 p（默认行为不变），eval_soft 新增 --hard-inject；5/5 起点扫描延后到阶段二末 2.6b，用本目录 eval_soft.py 从断点（198 行/18 例）续跑，规模不变 100×11

```
709ce95b872f668acd79392fad21d2ee  frozen_stage1/cert_perturb.py
91d8ed69eba679850fa6c3922d7c5ee1  frozen_stage1/eval_soft.py
b178f657cbed98ff9156b54b050076af  frozen_stage1/make_cert_ids.py
58c36ed8ffe3659729ba7efb1cbd4bc4  frozen_stage1/make_ids.py
13c803db5bfac2ba290b01b726ffd3e7  frozen_stage1/pilot_metrics.py
6f5d964d09a70d82c5fc36c1650b961d  frozen_stage1/quick_analyze.py
819673276bea8e8d672f591999515837  frozen_stage1/soft_decode.py
3271b7a1bfd68e7fe0a250feb0948021  frozen_stage1/soft_mixer.py
d2458a6118eac6c83ba5ffaa9b4b153b  frozen_stage1/run_pilot.sh
```
