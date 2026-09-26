# VinDr-Mammo LoRA 微调

## 数据

| split | 文件 | 条数 |
|---|---|---|
| 训练集 | `data/direct_train.json` | 16,000 |
| 测试集 | `data/direct_test.json` | 4,000 |

- 图片: `images_png/` (5,000 study, 20,000 PNG, 相对路径)
- 格式: alpaca `{instruction, output, images}`
- 划分: train 16,000 / test 4,000

## 一条龙流程 (服务器)

```bash
cd Mammo

# 1. 训练 (LoRA SFT)
llamafactory-cli train vindr_finetune/trial.yaml
#   → LoRA adapter 输出到 Qwen/qwen-4b/lora/

# 2. 推理 (全量 4,000 条 test)
python3 vindr_finetune/infer_lora.py \
    --prompt direct --split test \
    --adapter /hy-tmp/9_9/Qwen/qwen-4b/lora/
#   → 结果输出到 vindr_finetune/outputs/direct_<adapter>_1536_sdpa/predictions.jsonl


## 输出

- 训练 adapter: `/Qwen/qwen-4b/lora/`
- 推理结果: `vindr_finetune/outputs/{prompt}_{adapter_name}_1536_sdpa/predictions.jsonl`
  - 每行一条 JSON: `{index, image_id, prediction_json, eval_json, ground_truth, seconds}`
  - 支持断点续推 (已完成的 index 自动跳过)

## 关键配置

- 基座: `/Qwen3.5-4B`
- 模板: `qwen3_5_nothink` (训练) / `qwen3_5` (推理, enable_thinking=False)
- 图像: 1520×912 全分辨率, bf16, flash_attn=sdpa
- LoRA: rank=8, alpha=16, target=all, lr=1e-4
- `dataset_dir: vindr_finetune` (images 相对路径基准)

