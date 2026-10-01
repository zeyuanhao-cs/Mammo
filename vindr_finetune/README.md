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
训练成功后自动在测试集推理（786432 像素上限、256 新 token、关闭思考）。
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
