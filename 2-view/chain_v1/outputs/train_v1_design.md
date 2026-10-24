# 链条侧 v1 · 训练代码（顺序 A：两次前向 + 软证据 KD）

> 状态：**只写代码与单元测试，未启动训练，未占用 GPU**。探针（soft evidence）出结果前不开训练。

## 1. 代码结构

```
chain_v1/
├── train/
│   ├── core.py        # 损失与位置语义（模型无关，可在 CPU 上用极小模型测试）
│   ├── data.py        # 数据：prompt + GT JSON、labels、finding/birads 位置、图像张量
│   ├── trainer.py     # SoftKDTrainer(Trainer)：ρ 调度、compute_loss、训练日志
│   └── run_train.py   # CLI 入口：ChatModel 加载 + LoRA(peft) + Trainer（不自动跑）
├── infer/
│   └── soft_decode.py # soft-decode 解码循环（batch=1）+ 可跑的 CLI
└── tests/
    ├── test_positions.py   # 测试 1：替换位置正确
    ├── test_core_cpu.py    # 测试 2、3：detach-p 梯度 / ρ=0 等价 SFT（含 soft 路径冒烟）
    └── test_soft_decode.py # 测试 4：p∈{0,1} 时 soft-decode == 贪心
```

### 数据与位置（`core.py` 顶部约定）

- 序列 = `[prompt | completion]`，`labels` 在 prompt 段为 `-100`、completion 段为 GT token（连续后缀）。
- `finding_pos` / `birads_pos` 均为**相对 completion 起点**的下标；`logits_to_keep = K = comp_len + 1` 时第 r 行预测 completion 第 r 个 token，行号即 completion 内下标，直接索引。
- **第 2 次前向替换的是取值 token 自身位置**（`seq_pos = prompt_len + comp_idx`），即后续 token 看到的 hidden state；在第 0 个 decoder layer 的 `forward_pre_hook` 里改写，等价于替换输入嵌入。
- 图像输入（pixel_values / image_grid_thw / mm_token_type_ids）两次前向复用同一份张量；**但 vision tower 目前各算一遍**（见「已知限制」）。

### 损失

```
p_j  = sigmoid(l1 - l0)                        # 第 1 次前向在 4 个 finding 位置的 logits
q_T  = softmax(BI-RADS 位置在 '1'..'5' 上)      # detach
q_S  = 同位置、把 finding 位置换成 (1-p)E[0] + p E[1] 后的分布
L    = CE1 + alpha * CE2(birads 位置, GT) + lam * KL(q_T || q_S)
```

第 2 次前向按 ρ 逐样本启用（batch=1 即逐条），ρ 在前 `rho_warmup_frac`（默认 0.2）步数内从 0 线性升到 `rho_max`（默认 0.5）；未启用时 `L = CE1`。

## 2. 开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `--alpha` | 1.0 | CE2 权重 |
| `--lam` | 1.0 | KL 权重 |
| `--rho-max` | 0.5 | 第 2 次前向的最大启用概率 |
| `--rho-warmup-frac` | 0.2 | ρ 线性 warmup 占总步数的比例 |
| `--detach-p` | 关（默认**不** detach） | p 不回传梯度 |
| `--no-kd` | 关 | `lam = 0`（保留第 2 次前向与 CE2） |
| `--no-soft` | 关 | 关闭第 2 次前向，退化为纯 SFT（**B0 基线就用它**） |
| `--soft-mode` | `logit_mlp` | 软证据注入方式：`logit_mlp`（主方案）/ `linear`（对照） |
| `--soft-mixer-init` | 空 | 从已有 `soft_mixer.pt` 恢复 mixer 权重 |
| `--no-reuse-embeds` | 关（默认复用） | 关闭"第 2 次前向复用融合嵌入"，改为重跑视觉塔（对照用） |
| `--seed` | 42 | 与 LLaMA-Factory 默认一致，显式固定 |
| `--freeze-vision-tower` | 关 | 与原配置一致应为 False（视觉塔纳入 LoRA） |
| `--bs` | 1 | 强制 1（位置/ρ 按单条实现） |
| `--grad-ckpt` | 开 | 梯度检查点 |
| `--save-steps` | 200 | 断点粒度（原配置为 100） |

推理侧：`chain_v1/infer/soft_decode.py --no-soft` 关闭软解码；默认开启，`events` 打印 `(step, field, p)`。

训练日志（每 `logging_steps` 步）：`ce1 / ce2 / kl / mean_p / rho / low_p_ratio`，
`low_p_ratio` = 当步「GT 有 finding 但 p<0.5」的比例（只统计 GT 阳性位置）。

## 3. 单元测试结果（全部 CPU 通过）

### 测试 1 · 替换位置（`test_positions.py`）

```
breast_density                     comp_idx= 6 值='C' id= 34   [3]'_density' [4]'":' [5]' "' >>[6]'C' [7]'",' [8]' "' [9]'find'
breast_birads                      comp_idx=54 值='1' id= 16   [51]'ads' [52]'":' [53]' ' >>[54]'1' [55]'}'
findings.mass                      comp_idx=16 值='0' id= 15   [13]'mass' [14]'":' [15]' ' >>[16]'0' [17]',' [18]' "' [19]'s'
findings.suspicious_calcification  comp_idx=26 值='0' id= 15   [23]'ification' [24]'":' [25]' ' >>[26]'0' [27]',' [28]' "' [29]'as'
findings.asymmetry                 comp_idx=34 值='0' id= 15   [31]'metry' [32]'":' [33]' ' >>[34]'0' [35]',' [36]' "' [37]'arch'
findings.architectural_distortion  comp_idx=44 值='0' id= 15   [41]'ortion' [42]'":' [43]' ' >>[44]'0' [45]'},' [46]' "' [47]'bre'
```

### 测试 2 · p 的梯度

`detach=False → p.requires_grad=True`（logits.grad 非零）；`detach=True → False`。

### 测试 3 · ρ=0 与 SFT 完全一致

```
loss=3.245289  vs  SFT CE=3.245289   （差 < 1e-6）
rho_at(0,1000)=0.0, rho_at(200,1000)=0.5
soft 路径冒烟：loss=7.759518 = ce1 3.245289 + ce2 4.462685 + kl 0.051544，q_S 归一化、loss 有限
```

### 测试 4 · soft-decode 与贪心一致

```
p0: tokens 与贪心一致, 替换处 |emb-E|=0.00e+00, events=[(4,'mass',9.36e-14)]
p1: tokens 与贪心一致, 替换处 |emb-E|=0.00e+00, events=[(4,'mass',1.0)]
```

## 3.5 软证据注入的两种模式（`--soft-mode`）

| 模式 | 该位置输入 | 说明 |
|---|---|---|
| `linear`（对照） | `(1-p)·E["0"] + p·E["1"]` | p 的响应近乎阶跃（探针：p≤0.4 时 P(≥3)≈1e-4，p=0.5 跳到 0.81，p≥0.6 稳定 ~0.996） |
| `logit_mlp`（主方案） | `E[base] + MLP([log p, log(1-p)], type)` | `base` = **模型自己的判断**：`p>0.5 → "1"`，否则 `"0"`（训练与 soft-decode 完全一致，**不看 GT**） |

`MLP`: `2 → 64 → hidden_size`，中间 GELU；最后一层默认 **N(0, 1e-3) 小随机初始化**（`--mixer-init small`），
也可用 `--mixer-init zero`（增量恒为 0）。若用 `zero`，第 2 次前向等于"把 finding 取值换成**模型自己 argmax token** 后的硬前向"
（不是 GT teacher forcing —— 因为 base 取模型判断，模型会看到"自己写 0 但仍有怀疑"的输入，与推理一致）；
4 类 finding 共用 MLP，另加 **4×64 可学习类型嵌入**（加到第一层输出上）；`p` 先 clamp 到 `[1e-6, 1-1e-6]` 再取对数。
mixer 参数与 LoRA 一起训练（`create_optimizer` 里追加 param group），每个 checkpoint 保存为 `soft_mixer.pt`
（`SoftKDTrainer._save`），推理时用 `--soft-mixer` 加载；`soft_decode` 同步支持两种模式。

### 测试 5/6 · soft_mixer（`test_soft_mixer.py`）

```
test_zero_init_equals_own_argmax_hard: OK  qS - q_ref max=0.00e+00
      # zero 初始化时，第2次前向 == "把 finding 取值换成模型自己 argmax token" 的硬前向
test_base_uses_model_not_gt:           OK  n=4，GT 全为 1 时 4 个位置的 base 全为 "0"（p<0.5）
      # 且 p=0.15、GT=1 时注入向量恰为 E["0"]
test_small_init_gap:                   OK  q maxdiff=1.176e-04, |Δemb|max=4.592e-03（TinyLM hidden=16）
test_p_gradient_nonzero:               OK  dL/dp(0.05)=-0.097062, dL/dp(0.20)=-0.024903
```

`small` 初始化下 Δ 的尺度 ∝ √hidden_size：TinyLM 里 hidden=16 得到 4.6e-3，换成 hidden=2560 约放大 12.6×（~5.8e-2），
真实模型上建议实测一次"qS 与硬前向的差"，必要时把 std 从 1e-3 降到 3e-4。

## 3.6 第 2 次前向复用融合嵌入（跳视觉塔）

- 第 1 次前向在 **text model 入口**用 `register_forward_pre_hook(with_kwargs=True)` 截获**该次调用的全部实参**
  （真实模型实测为 `input_ids=None / position_ids (3,1,1964) int64 / attention_mask / past_key_values=None /
  inputs_embeds (1,1964,2560) bf16 / use_cache=False`，**没有** `mm_token_type_ids`、没有 deepstack）；
  第 2 次**原样复用这些实参、只替换 `inputs_embeds`**，再 `lm_head(hidden[:, -K:])`，**不再进入视觉塔**。
  注意：不能只传 `inputs_embeds + mm_token_type_ids` 让 LM 自己重算 M-RoPE，那样 position_ids 可能与第 1 次不一致。
- 梯度仍经这份嵌入回传到 `embed_tokens` 与视觉塔（因为第 1 次的计算图被复用，不是 detach 出来的副本）。
- 找不到 text model 或未截获到嵌入时自动回退到"重跑整模型 + hook 注入"，行为与之前一致。

单元测试 `tests/test_embed_reuse.py`（CPU，极小 VL 桩模型：emb → visual 融合 → text → lm_head）：

```
BI-RADS logits 最大差 : 0.000e+00
loss 差              : 0.000e+00
梯度最大差            : 0.000e+00  (model:emb.weight)
分组梯度最大差：emb 0.0 / visual 0.0 / lm_head 0.0 / mixer 0.0
```

真实模型（eval、无 dropout、3 样本）：`position_ids` 逐位相同，density / BI-RADS / 4 个 finding 位置 logits
最大差均为 **0.000e+00**（`chain_v1/outputs/real_parity_reuse.md`）。
训练模式下两次前向共享同一 dropout 掩码，这是复用与不复用**唯一**预期的行为差异。

## 4. 显存估计（基准：**786432 单次前向实测 ≈ 17.3 GB smi**）

以现分辨率（786432，每视角 756 token、样本约 1962 token）单次前向的实测峰值为基准，不再使用原分辨率 memtest 的数字：

```
单次前向 smi 峰值            ≈ 17.3 GB
其中 bf16 权重（4.54B）      ≈  8.7 GB
=> 单次激活 + workspace      ≈  8.6 GB

两次前向（两份计算图共存）   ≈ 8.7 + 2 × 8.6 ≈ 25.9 GB   > 24 GB  → 预计 OOM
一次前向 + ρ_max 0.5 的期望  ≈ 8.7 + 1.5 × 8.6 ≈ 21.6 GB （但峰值看的是"最坏那条"，仍是 25.9 GB）
```

结论：**默认设置（ρ_max=0.5、p 不 detach、两份计算图共存）在 A40-24G 上大概率 OOM**。

**完整方法必须单次 backward、两份图共存**：拆 backward（先 `CE1.backward()` 释放第 1 份图，再跑第 2 次前向）
会让第 1 次前向的图被释放，`p` 因此拿不到梯度 —— 等价于强制 `--detach-p`，**不能**作为完整方法的省显存方案，
只能作为单独的 `--detach-p` 变体。

落地策略（按实测决定）：

1. **先在 A40 上用完整方法实测 20 步显存**（`--epochs 1` + 短跑/`max_steps=20`）；
2. 放得下 → 本地 A40 跑完整方法；
3. 放不下 → 完整方法放 NeSI H100；**A40 只跑 B0（`--no-soft`）与 `--detach-p` 变体**。

实测方法（脚本已具备，照搬 `scripts/bench/train_memtest.py` 的钩子）：

```python
_orig = transformers.Trainer.train
def patched(self, *a, **k):
    torch.cuda.reset_peak_memory_stats()
    out = _orig(self, *a, **k)
    print("PEAK_ALLOCATED", torch.cuda.max_memory_allocated()/1024**3)
    print("PEAK_RESERVED ", torch.cuda.max_memory_reserved()/1024**3)
    return out
transformers.Trainer.train = patched
```

外加 `nvidia-smi --query-gpu=memory.used` 0.5 s 采样取峰值；`s/it` 取第 4–20 步中位数外推 878 步（2 epoch）。

> 注意：这台 A40-24Q（vGPU）**不支持** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，任何 CUDA 分配都会
> `CUDA driver error: operation not supported`，试跑不要带该变量。

## 5. 基线 B0 的定义

**B0 必须用本套新代码训练**（`run_train.py --no-soft`），不与 LLaMA-Factory 训出的历史模型比较：

```
B0 : run_train.py --no-soft   （L = CE1，等价纯 SFT；试验阶段与 A 同为 --epochs 1）
A  : run_train.py             （L = CE1 + α·CE2 + λ·KL，ρ_max=0.5）
```

两者除 `--no-soft` 外参数完全相同（同一 data、同 seed=42、同 `--epochs 1`（试验）/2（正式）、同 LoRA、同 lr/scheduler/batch），
因此 A − B0 的差异只来自软证据 + KD 部分。B0 也是"同一代码路径、同一预处理、同一评估脚本"下的对照，避免历史上 LLaMA-Factory 版本/数据缓存差异带来的混淆。

## 6. 与原 `train_lora_2view_os_sub_vt` 配置对照（`configs/train_lora_2view_os_sub_vt_local.yaml`）

| 项 | 原配置 | 新代码 | 结论 |
|---|---|---|---|
| 数据文件 | `vindr_2view_train_os_sub_local` → `data/llamafactory/train_llamafactory_os_sub_local.json`（Repeat-OS 后 7019 条） | 同一文件（`--data` 默认即此路径） | ✅ 一致 |
| prompt | 数据集内 `instruction` 字段 | 同样取每条 `instruction`（缺失才回落 `dataset_build/prompt.txt`） | ✅ 一致 |
| template | `qwen3_5_nothink` | `qwen3_5_nothink` | ✅ 一致 |
| image_max_pixels | 786432 | 786432 | ✅ 一致 |
| image_min_pixels | 262144 | 262144 | ✅ 一致 |
| LoRA rank | 8 | 8 | ✅ 一致 |
| LoRA alpha | 16 | 16 | ✅ 一致 |
| LoRA dropout | 0.05 | 0.05 | ✅ 一致 |
| LoRA target（含视觉塔） | `lora_target: all` + `freeze_vision_tower: false` → `find_all_linear_modules(model, False)` | **直接调用同一个函数**取目标模块名集合 | ✅ 一致（同一实现） |
| lr | 5.0e-5 | 5e-5 | ✅ 一致 |
| scheduler | cosine | cosine | ✅ 一致 |
| warmup | 0.05（ratio） | 0.05（ratio） | ✅ 一致 |
| 有效 batch | 1 × 16 = 16 | 1 × 16 = 16 | ✅ 一致 |
| epoch | **1.0** | **2.0**（默认） | ❌ 不一致（实验设计如此；如需对齐改 `--epochs 1`） |
| seed | 未写，用 TrainingArguments 默认 42 | 显式 42（`seed`/`data_seed`） | ✅ 等价 |
| gradient checkpointing | true | true | ✅ 一致 |
| bf16 | true | true | ✅ 一致 |
| cutoff_len | 4096 | **未实现截断**（786432 下最长 1962 ≪ 4096） | ⚠️ 行为等价，缺截断逻辑 |
| save | `save_steps: 100`, `save_total_limit: 3` | `save_steps: 200`, limit 2 | ⚠️ 仅 checkpoint 粒度不同 |
| 自动续训 | 有（output_dir 内自动找最近 ckpt） | **无**（需手动 `resume_from_checkpoint`） | ⚠️ 不一致 |
| dataloader workers | 0 | 0 | ✅ 一致 |
| preprocessing workers | 8（HF 缓存） | 无（单进程构建，无缓存） | ⚠️ 预处理更慢，不影响结果 |
| plot_loss | true | 无 | ⚠️ 仅可视化 |
| eval_strategy | no | no | ✅ 一致 |
| position_ids | 由 LLaMA-Factory collator 决定 | 不传，由模型按 attention_mask 内部计算 | ❓ 待核对（可能影响 RoPE 偏移） |

**需要处理的不一致**：epoch（设计差异，明确即可）、截断逻辑缺失（当前输入远小于 cutoff，暂时无影响）、
自动续训缺失（长训练前应补 `resume_from_checkpoint` 逻辑）、`position_ids` 传参待核对。

## 7. 已知限制

1. **vision tower 未复用**：两次前向各跑一遍（ρ_max=0.5 时平均 1.5× 视觉计算）。若显存/速度不可接受，再做「视觉输出缓存 + 只跑 LM」。
2. **batch 固定 1**：ρ、位置索引按单条实现；批量化需要 padding 掩码 + 按启用样本加权。
3. 拆 backward 不适用于完整方法（会使 p 失去梯度，见 §4）；完整方法按 §4 的 A40/H100 策略落地。
4. 自动续训已实现：`--resume auto`（默认）从 `output_dir` 下最新 `checkpoint-<n>` 续训，`off` 表示从头，也可传显式路径。
5. 前向一致性测试（`tests/test_forward_parity.py`，GPU）比较三条路径的 CE 与 BI-RADS/finding 位置 logits：
   A=新代码（不传 position_ids）、B1=LLaMA-Factory collator（含 3D rope position_ids/rope_deltas）、B2=同 B1 但去掉 position_ids；
   结果写 `chain_v1/outputs/forward_parity.md`，若 B1 与 A 差异显著而 B2 与 A 一致，说明必须在新代码里补 position_ids。
4. 未引入新数据格式（无 any_finding），沿用原 `output` JSON 字段。
