#!/usr/bin/env python3
"""用 llama-factory 训练入口跑内存测试，并打印 torch 显存峰值。

用法: python scripts/bench/train_memtest.py configs/train_lora_2view_native_memtest_local.yaml
"""
import sys
from pathlib import Path

import torch
import transformers
from transformers import Trainer, Seq2SeqTrainer

PROJ = Path(__file__).resolve().parent.parent.parent


def _wrap(cls):
    orig = cls.train

    def patched(self, *a, **k):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        try:
            out = orig(self, *a, **k)
        except torch.cuda.OutOfMemoryError:
            print(f"[mem] OOM during {cls.__name__}.train "
                  f"(peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.3f} GiB)", flush=True)
            raise
        torch.cuda.synchronize()
        alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        res = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f"[mem] PEAK_ALLOCATED={alloc:.3f} GiB  PEAK_RESERVED={res:.3f} GiB", flush=True)
        return out

    cls.train = patched


_wrap(Trainer)
if Seq2SeqTrainer is not Trainer:
    _wrap(Seq2SeqTrainer)

cfg = sys.argv[1] if len(sys.argv) > 1 else str(PROJ / "configs" / "train_lora_2view_native_memtest_local.yaml")
sys.argv = [sys.argv[0], cfg]  # 模拟 launch() pop("train") 之后的 argv

from llamafactory.train.tuner import run_exp  # noqa: E402

run_exp()
