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
