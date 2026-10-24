# 六项修复核对（2026-10-24 14:20，逐项 grep/日志核实，全部在代码里）

- a. LoRA 目标与 LLaMA-Factory 一致：`run_train.py:61-66` 用 LLaMA-Factory 同款
  `patch_target_modules`，`:64 freeze_multi_modal_projector=True`（merger 不挂 LoRA）。
  逐键一致证据：`logs/pilot_init_test.log` ——
  `[init-adapter] 从 .../checkpoint-439 加载 688 个张量 (lora_missing=0, unexpected=0)`
  （688 = 344 模块 × lora_A/B，checkpoint 全部命中，无缺失无多余）。
- b. mixer small 初始化按 p=0.5 校准到偏移 ≈1%：`run_train.py:211-234`
  （`scale = 0.01 / rs0[2]`，`fc2.weight.mul_(scale)`）。实测日志：
  `[mixer-scale] 校准前 ||delta||/||E[base]|| = ['0.1845',...,'0.3638'] median=0.2066；`
  `按计划 1% 校准（×0.0484）后 = ['0.0089','0.0099','0.0100','0.0164','0.0176'] median=0.0100`
- c. 创建 mixer 前固定 seed：`run_train.py:203-206`（`torch.manual_seed(args.seed)` 在
  `SoftMixer(...)` 之前，TrainingArguments 之后）。
- d. soft_vector device 修复：`soft_mixer.py:59-62`
  `feat = torch.stack([log p, log1p(-p)]).to(device=fw.device, dtype=fw.dtype)`。
- e. soft_decode 显式传 position_ids：`soft_decode.py:50-72`（prefill 用
  `model.model.get_rope_index` 构 4 通道位置；decode 步 `position_ids = past_len + rope_delta`）。
  本次同步移植到探针 `soft_evidence_probe.py::forward_birads`。
- f. 训练结束产物落 output_dir 根：`run_train.py:399-406`
  （`trainer.save_model(output_dir)` + `save_soft_mixer(...)`（soft_mixer.pt）+
  `trainer_state.json`（global_step/max_steps））。

结论：六项齐备，无需补丁；pilot 总控可在对齐检查通过后照常启动。
（P0.5 上次 FAIL 是 GPU 显存冲突 OOM，非代码问题。）

---

[2026-10-24 13:39:21] ==== pilot 开始（PID 3040297, host hpc-ywang01）====
[2026-10-24 13:39:21] P0: PYTORCH_CUDA_ALLOC_CONF 未设置；起点 adapter=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt/checkpoint-439
[2026-10-24 13:39:21] P0.5: --init-adapter 2 步测通开始
[2026-10-24 13:39:47] P0.5: FAIL rc=1（见 logs/pilot_init_test.log），停下
[2026-10-24 14:30] 对齐检查结论: 已知 b8 批量差异，约 2% 翻转（4/200），主比较统一用 eval_soft。证据: 196 一致样本 P(>=3) 差 median=0.00002/p95=0.00017, max_sign_p median=0.00377, 5类概率差(合并) median=0.00001/p95=0.032; 4 分叉点 eval_soft margin=0.124/0.245/0.000/0.000 全 <0.3（0.754 极值来自分叉后文本错位）。详见 /tmp/align_diag.py + align_check.json。
[2026-10-24 14:21:30] ==== pilot(时间压缩版) 开始（PID 3052423）====
[2026-10-24 14:21:30] P0.5: --init-adapter 2 步测通开始
[2026-10-24 14:23:29] P0.5: [init-adapter] 从 /home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt/checkpoint-439 加载 688 个张量 (lora_missing=0, unexpected=0)
[2026-10-24 14:23:29] P0.5: [mixer-scale] 校准前 ||delta||/||E[base]|| = ['0.1845', '0.2039', '0.2066', '0.3383', '0.3638'] median=0.2066；按计划 1% 校准（×0.0484）后 = ['0.0089', '0.0099', '0.0100', '0.0164', '0.0176'] median=0.0100
[2026-10-24 14:23:29] P0.5: PASS
[2026-10-24 14:23:29] ==== 阶段一: QC1 训练开始（50 步, seed 42）====
[2026-10-24 15:01:29] [QC1] 进度 50/50  {'loss': '0.5411', 'grad_norm': '8.514', 'learning_rate': '5.583e-08', 'epoch': '0.114'}
[2026-10-24 15:01:29] 产物检查 PASS: /home/z/zhao/Mammography/model/chain_v1_QC1 (global_step=50)
[2026-10-24 15:01:29] QC1: [mixer-scale] 校准前 ||delta||/||E[base]|| = ['0.1845', '0.2039', '0.2066', '0.3383', '0.3638'] median=0.2066；按计划 1% 校准（×0.0484）后 = ['0.0089', '0.0099', '0.0100', '0.0164', '0.0176'] median=0.0100
[2026-10-24 15:01:29] 阶段一评测 1/5: 起点 hard 全验证集 1600（对齐检查已过，改用 b8 管线，~0.6s/例 vs ~2.6s/例）
[2026-10-24 15:20:16] 阶段一评测 1/5 done: 1600 行（b8 格式，make_ids/quick_analyze 已兼容）
[2026-10-24 15:20:16] 阶段一: 挑批次
[2026-10-24 15:20:16] 阶段一评测 1.5/5: 起点 hard 子集（eval_soft，与 QC1 同管线同口径，机制指标用）
[2026-10-24 15:29:01] 阶段一评测 1.5/5 done: 200 行
[2026-10-24 15:29:01] 阶段一评测 2/5: QC1 soft（≤200）
[2026-10-24 15:29:27] 阶段一评测 2/5 FAIL
[2026-10-24 15:43:13] ==== pilot(时间压缩版) 开始（PID 3071621）====
[2026-10-24 15:43:13] 产物检查 PASS: /home/z/zhao/Mammography/model/chain_v1_QC1 (global_step=50)
[2026-10-24 15:43:13] QC1 产物已齐（global_step=50），跳过训练
[2026-10-24 15:43:13] 评测 1/5 已有 1600 行，跳过
[2026-10-24 15:43:13] 阶段一: 挑批次
[2026-10-24 15:43:13] 评测 1.5/5 已有 200 行，跳过
[2026-10-24 15:43:13] 阶段一评测 2/5: QC1 soft（≤200）
[2026-10-24 15:52:02] 阶段一评测 3/5: QC1 swap_hard（软换硬；batch=1 与 soft 同口径）
[2026-10-24 15:56:52] ==== pilot(时间压缩版) 开始（PID 3075750）====
[2026-10-24 15:56:52] 产物检查 PASS: /home/z/zhao/Mammography/model/chain_v1_QC1 (global_step=50)
[2026-10-24 15:56:52] QC1 产物已齐（global_step=50），跳过训练
[2026-10-24 15:56:52] 评测 1/5 已有 1600 行，跳过
[2026-10-24 15:56:52] 阶段一: 挑批次
[2026-10-24 15:56:52] 评测 1.5/5 已有 200 行，跳过
[2026-10-24 15:56:52] 评测 2/5 已有 200 行，跳过
[2026-10-24 15:56:52] 阶段一评测 3/5: QC1 swap_hard（软换硬；batch=1 与 soft 同口径）
[2026-10-24 16:03:31] 阶段一评测 4/5: QC1 scan（50+50）
[2026-10-24 16:40:47] 阶段一评测 5/5: 起点 scan 对照（linear）
[2026-10-24 16:47:47] ==== pilot(时间压缩版) 开始（PID 3087288）====
[2026-10-24 16:47:47] 产物检查 PASS: /home/z/zhao/Mammography/model/chain_v1_QC1 (global_step=50)
[2026-10-24 16:47:47] QC1 产物已齐（global_step=50），跳过训练
[2026-10-24 16:47:47] 评测 1/5 已有 1600 行，跳过
[2026-10-24 16:47:47] 阶段一: 挑批次
[2026-10-24 16:47:47] 评测 1.5/5 已有 200 行，跳过
[2026-10-24 16:47:47] 评测 2/5 已有 200 行，跳过
[2026-10-24 16:47:47] 评测 3/5 已有 200 行，跳过
[2026-10-24 16:47:47] 评测 4/5 已有 1100 行，跳过
[2026-10-24 16:47:47] 阶段一评测 6/6: 确定样本软注入扰动（100 例，QC1 软 vs 软换硬，batch=1）
[2026-10-24 16:57:00] 阶段一: 分析 + quick_report + GATE
- P(BI-RADS=1) 中位：软 0.8187 / 软换硬 0.8642
- |ΔP(BI-RADS=1)| 平均：0.0453（中位 0.0462，最大 0.0929）
- 预测 BI-RADS 不一致：0/100
[2026-10-24 16:57:01] 阶段一: GATE=STOP（rc=1），停下等用户
[2026-10-24 17:38:51] linear 模式 2 步训练测通 PASS（44.4s/步，无 mixer，产物齐全）
[2026-10-24 17:38:51] 阶段一结论：GATE=STOP（QC1 logit_mlp 扫描方向反向，归因 mixer；同批硬 token 注入方向正确且强）。阶段二自动启动：B0 no-soft → C1 linear → C2 linear+detach-p
[2026-10-24 17:38:51] ==== pilot(时间压缩版) 开始（PID 3100857）====
[2026-10-24 17:38:51] ==== 阶段二: B0 / C1 训练（串行）====
[2026-10-24 17:38:51] 产物检查 FAIL: /home/z/zhao/Mammography/model/chain_v1_pilot_B0 缺 adapter
[2026-10-24 17:38:51] [B0] 训练尝试 1/3（1 epoch=439 步, seed 42, resume=auto）
[2026-10-24 17:42:42] ==== pilot(时间压缩版) 开始（PID 3101901）====
[2026-10-24 17:42:42] ==== 阶段二: B0 / C1 训练（串行）====
[2026-10-24 17:42:42] 产物检查 FAIL: /home/z/zhao/Mammography/model/chain_v1_pilot_B0 缺 adapter
[2026-10-24 17:42:42] [B0] 训练尝试 1/3（1 epoch=439 步, seed 42, resume=auto）
[2026-10-24 17:47:04] ==== pilot(时间压缩版) 开始（PID 3103017）====
[2026-10-24 17:47:04] ==== 阶段二: B0 / C1 训练（串行）====
[2026-10-24 17:47:04] 产物检查 FAIL: /home/z/zhao/Mammography/model/chain_v1_pilot_B0 缺 adapter
[2026-10-24 17:47:04] [B0] 训练尝试 1/3（1 epoch=439 步, seed 42, resume=auto）
