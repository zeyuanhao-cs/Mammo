# VinDr-Mammo LoRA 微调（h200 Slurm）

此目录在容器中为 `/mammo/vindr_finetune`。图片位于 `/mammo/images_png`；
训练配置用 `media_dir: /mammo` 解析 JSON 中的相对图片路径，无需修改 20,000 条标注。

| 数据 | 路径 | 条数 |
|---|---|---:|
| 训练 | `/mammo/vindr_finetune/data/direct_train.json` | 16,000 |
| 测试 | `/mammo/vindr_finetune/data/direct_test.json` | 4,000 |

当前 Slurm 规则将 `/mammo` 挂为 `rw`，但 `yu.w` 在容器内对主机目录没有写权限。
训练 adapter 和推理结果因此写入 `yu.w` 可写的 `/data/me/mammo`；
基座模型从已有的 `/data/models/Qwen3.5-4B` 读取。

当前推理使用已有的 `checkpoint-1000`（约 1.11 epoch）。在包含
LLaMA-Factory、PyTorch 和所需依赖的 GPU 容器中运行：

```bash
python3 /mammo/vindr_finetune/infer_lora.py \
  --prompt direct --split test \
  --adapter /data/me/mammo/qwen3.5-4b-lora/checkpoint-1000
```

`trial.yaml` 保留 1 epoch 的独立训练配置；仅在需要重新训练时运行
`llamafactory-cli train /mammo/vindr_finetune/trial.yaml`。

`direct` 推理直接读取本目录的测试 JSON；`icl` 和 `cot` 仍需要同级
`/mammo/vindr_infer/infer_data.<prompt>.test.json`。默认推理输出位于
`/data/me/mammo/outputs/`。`build_data.py` 仅在需要重新生成数据时使用，
其 `--data-root` 必须指向含 `vlm_dataset/{train,test}.json` 的源目录。

## 实验一：Qwen3.5-9B

`trial_9b_balanced.yaml` 沿用合作者 `main@20e980b` 的平衡训练配置：
4657 条训练记录、2 epoch、LoRA r=8/alpha=16、学习率 5e-5、
batch size 2 × 累积 8、786432 图像像素上限、SDPA、关闭思考。
训练集包含重复采样，实际有 2948 张不同图片，与 4000 张测试图无重叠。
测试 JSON 使用该提交更新后的提示词；图像和真值与旧版本一致。

通过 Slurm 在已预约的 H200 GPU 上运行 `bash vindr_finetune/run_exp1_9b.sh`，
并设置 `MAMMO_GPU_PHYSICAL` 为提交时固定的物理卡号。
脚本校验数据哈希、全部图片可读性和模型文件，生成独立的每作业配置，
训练成功后只推理 baseline 相同的前 500 条（索引 0–499，786432 像素上限、256 新 token、关闭思考）。
数据从当前 Git checkout 读取，图片通过 `/mammo/images_png` 读取。
adapter、日志、推理结果和汇总全部写入 `/data/me/mammo` 下带作业 ID 的目录。

## 实验二：Flash Next 思考过程蒸馏

`run_distill_thinking.sh` 通过现有 `http://10.222.10.107:9241/v1` 服务调用
`qwen38-flash-next`，开启 `enable_thinking`。客户端使用 Slurm `download`
队列、零 GPU，不启动或修改模型服务。提交时设置 `MAMMO_COMMIT` 为固定代码提交。

输入为 `train_balanced_2to1.json` 的原始 PNG 和真值标注；只处理 2948 个不同的
图像/标注组合，复用重复采样结果后生成 4657 条训练记录。测试集仅用于校验无重叠。
要求回复正常结束、思考非空、最终 JSON 与输入真值一致；不合格结果记录错误码，
连续 10 条失败停止。未覆盖全部样本时不发布最终训练集。

原始回复和生成数据仅存于 `/data/me/mammo/distillation/flash-next-<JOB_ID>/`。
完成产物为 `train_balanced_2to1_thinking.json`、`dataset_info.json` 和 `summary.json`；
输出保留原始真值，思考写在 `<think>...</think>` 内。这是基于标注的教师生成解释，
尚未经临床正确性验证。`progress.json` 仅含汇总进度。

需要断点恢复时显式设置 `DISTILL_RUN_DIR` 为旧运行目录，保持模型、提示词、代码
提交和参数一致；缓存校验和文件锁避免重复调用与并发覆盖。此脚本只蒸馏，后续
4B 思考模式微调需另行启动。

### 对风险样本二次推理复核

`review_distillation.py` 接收抽查审计 JSON，选取任何一轮被标记为
`recommended_review` 的记录。通过原图、真值和旧解释调用 Flash Next，开启思考，
返回 `accept`、`revised` 或 `reject`，以及修正后的解释；真值必须保持一致。
复核失败最多重试一次，仍不合格的记录排除并保存错误码，不等待人工复核。

二次复核使用独立的零 GPU Slurm 作业，读取原蒸馏目录但不修改它。设置
`--await-final-dataset` 后，风险记录先完成复核，原蒸馏成功结束后自动生成独立的
训练数据与 `summary.json`：保留其他记录，替换通过复核的解释，排除未通过者。
被排除记录仍保留在原始蒸馏产物中，复核目录包含 `excluded.json` 汇总。
这是对已发现风险的同模型复核，不表示全数据经过审计或已获医学正确性验证。

### 蒸馏后的 4B 微调与三方案对比

`trial_4b_thinking.yaml` 使用复核后的 4657 条数据、Qwen3.5-4B、2 epoch，保持
LoRA r=8/alpha=16、学习率 5e-5、batch 2 × 累积 8 与 786432 像素上限。
启用 `qwen3_5` 思考模板，序列长度为 8192；预检会确认思考标签保留、训练/测试
无重叠，图像顺序、重复采样权重和最终标签与 baseline 一致，训练提示词仅按
`thinking_instruction` 转换；文本 token 数加保守的图像 token 预算不会截断。

通过已预约的单卡 Slurm 作业运行 `run_exp2_4b_thinking.sh`，设置
`MAMMO_COMMIT`、`MAMMO_GPU_PHYSICAL`。训练成功后开启思考推理，生成上限为
4096 token。只推理 baseline 同一组前 500 张并产生三方案主表，完成后退出。
`compare_thinking.py` 校验原始 4B 的固定 Git 预测与 9B 作业 12080 的结果版本、
测试图映射和真值，复用相同评测函数。思考内容不参与最终 JSON 评分；未结束的
思考若仅提及候选 JSON，不作为有效答案。表格同时说明生成预算的差异。
训练镜像不包含 Git；启动前由主机从固定提交导出原始 4B 的预测快照至
`/mammo/.cache/mammo-benchmarks/4b-2to1-20e980b-predictions.jsonl`，容器内按
固定 Git blob SHA-1 校验版本，不依赖 Git 或网络。

训练、推理、对比产物均在 `/data/me/mammo` 下按作业 ID 隔离；对比表位于
`runs/exp2-4b-thinking-<JOB_ID>/comparison_500/comparison.md`。
测试源文件有 4000 条，baseline 的 round3.sh 实际只推理前 500 条：357 张有病灶、
143 张正常。训练集为 4657 条（2948 张不同图片），包含 3105 条有病灶和 1552 条正常。
历史 9B 全量输出和蒸馏4B额外输出保留，但对比脚本仅选择 baseline 的 500 个索引；
范围外输出不参与三方案评分，测试源文件行数也不作为推理进度的分母。
原始预测、思考与训练日志保留在服务器，仅回传汇总结果。
