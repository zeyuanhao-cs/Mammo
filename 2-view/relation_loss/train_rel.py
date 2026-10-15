# relation_loss 训练入口: 沿用 os_sub 的 yaml/数据/模板/超参, 只替换 trainer 和 collator。
# 用法:
#   正式: python train_rel.py --lambda-rel 0.5   (或 0.0)
#   冒烟: python train_rel.py --lambda-rel 0.5 --smoke
# 不修改 llamafactory 源码, 不修改任何已有文件。

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import EarlyStoppingCallback, TrainerCallback

SCRIPT_DIR = Path(__file__).resolve().parent          # relation_loss/
PROJ_DIR = SCRIPT_DIR.parent                          # 2-view/
sys.path.insert(0, str(SCRIPT_DIR))

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.ploting import plot_loss
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.callbacks import LogCallback

from rel_core import RelationCollator, RelationTrainer, build_relation_meta

BASE_YAML = PROJ_DIR / "configs" / "train_lora_2view_os_sub.yaml"
VAL_JSON = SCRIPT_DIR / "data" / "val_llamafactory.json"


class RelMetricCallback(TrainerCallback):
    """在每次 train 日志点把区间累计的 r_lm/r1/r2 平均值合并进日志与 log_history。

    只在含 'loss' 的 train 日志时 flush (eval 指标日志不消费累计器)。"""

    def __init__(self):
        self._trainer = None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None or "loss" not in logs:
            return control
        tr = self._trainer or kw.get("trainer")
        stats = tr.flush_rel_stats() if tr is not None and hasattr(tr, "flush_rel_stats") else None
        if not stats:
            return control
        logs.update(stats)
        # 同时写入 log_history (若本条已 append)
        if state.log_history and state.log_history[-1] is not logs:
            for k, v in stats.items():
                state.log_history[-1].setdefault(k, v)
        return control


def build_config(lambda_rel: float, output_dir: str, smoke: bool, patience: int) -> dict:
    with open(BASE_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update({
        "dataset_dir": str(SCRIPT_DIR / "data"),        # 新 dataset_info, 不动原文件
        "dataset": "vindr_2view_train_os_sub",
        "eval_dataset": "vindr_2view_val",
        "output_dir": output_dir,
        "resume_from_checkpoint": False,
        "overwrite_output_dir": True,
        # 每 100 步验证集 LM loss, 只用 LM loss 早停
        "eval_strategy": "steps",
        "eval_steps": 10 if smoke else 100,
        "per_device_eval_batch_size": 4,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "early_stopping_steps": patience,               # finetuning_args: 连续 N 次无改善则停
        "logging_steps": 10,
        "save_steps": 10 if smoke else 100,
        "save_total_limit": 3,
    })
    if smoke:
        cfg["max_steps"] = 20
        cfg["max_samples"] = 200  # 冒烟只用前 200 条 (train/val 各 200), 加速迭代
    return cfg


def run_relation_sft(lambda_rel: float, output_dir: str, smoke: bool, patience: int,
                     rel_seed: int) -> None:
    cfg = build_config(lambda_rel, output_dir, smoke, patience)
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(cfg)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)

    # 值 token 定位预检: 训练前对已分词的全部 train/val 样本跑 build_relation_meta,
    # 提前暴露定位问题 (零额外预处理成本, 失败在加载模型之前)。
    def _precheck(name, ds):
        for i in range(len(ds)):
            build_relation_meta(ds[i], tokenizer)
        print(f"[rel] precheck OK: {name} {len(ds)} 条", flush=True)

    _precheck("train", dataset_module["train_dataset"])
    ev = dataset_module.get("eval_dataset")
    if isinstance(ev, dict):
        for k, v in ev.items():
            _precheck(f"eval[{k}]", v)
    elif ev is not None:
        _precheck("eval", ev)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = RelationCollator(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=-100,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        rel_seed=rel_seed,
        **tokenizer_module,
    )

    rel_cb = RelMetricCallback()
    callbacks = [LogCallback(), rel_cb, EarlyStoppingCallback(early_stopping_patience=patience)]

    trainer = RelationTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=None,
        ref_model=None,
        lambda_rel=lambda_rel,
        **dataset_module,
        **tokenizer_module,
    )
    rel_cb._trainer = trainer

    train_result = trainer.train()
    trainer.save_model()
    if finetuning_args.include_effective_tokens_per_second:
        train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
            dataset_module["train_dataset"], train_result.metrics, stage="sft")
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        keys = ["loss"]
        if isinstance(dataset_module.get("eval_dataset"), dict):
            keys += [[f"eval_{k}_loss"] for k in dataset_module["eval_dataset"].keys()]
        else:
            keys += ["eval_loss"]
        plot_loss(training_args.output_dir, keys=keys)

    # 汇总 rel 日志到 jsonl
    if trainer.is_world_process_zero():
        with open(os.path.join(output_dir, "rel_metrics.jsonl"), "w", encoding="utf-8") as f:
            for h in trainer.state.log_history:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-rel", type=float, required=True, help="0.0 -> 模型(a); 0.5 -> 模型(b)")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="20 步冒烟")
    ap.add_argument("--patience", type=int, default=4, help="早停: 连续 N 次 eval 无改善")
    ap.add_argument("--rel-seed", type=int, default=42)
    args = ap.parse_args()

    tag = "b" if args.lambda_rel > 0 else "a"
    out = args.output_dir
    if out is None:
        if args.smoke:
            out = str(SCRIPT_DIR / f"outputs/smoke_{tag}")
        else:
            out = f"/nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view-rel-{tag}"
    assert VAL_JSON.exists(), f"缺少 {VAL_JSON}, 先运行 make_val.py"
    print(f"[rel] lambda={args.lambda_rel} smoke={args.smoke} output={out}", flush=True)
    run_relation_sft(args.lambda_rel, out, args.smoke, args.patience, args.rel_seed)


if __name__ == "__main__":
    main()
