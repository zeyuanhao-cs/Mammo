#!/usr/bin/env python3
"""
双视图多头视觉 baseline（VLM 专用对照模型）。

- 数据: Run A 划分 (train=train_sub_raw.json / train_sub_repeat_os.json, val=val_raw.json, test=官方 test.json)
  每条样本同侧 CC+MLO, images[0]=CC, images[1]=MLO (make_vlm_splits.py:58 的固定顺序)
- 模型: timm backbone (默认 efficientnet_b2, ImageNet 预训练), CC/MLO 共享 backbone,
  分别 global pool 后拼接 -> 多头: density 4 类 / BI-RADS 5 类 / 4 findings 二分类 logit
- 训练: 原生 912x1520 不缩小, 轻微 affine/brightness 增强(不翻转), AdamW 分层 lr,
  cosine + 5% warmup, bf16, 梯度累积等效 batch 16, 每 epoch 存 checkpoint 可断点续跑
- 损失: CE(density) + CE(birads) + BCE(findings), 等权
- best checkpoint 只按 val total loss 选, test 不参与任何选择
- 输出: val/test 概率 jsonl, 字段与 VLM scoring 文件一致

视图选择 (--views):
  ipsi  同侧 CC+MLO (默认, 原双视图行为)
  cc    仅 CC   /  mlo  仅 MLO (单视图, backbone 输入 1 张)
  four  同一检查的左右乳各 CC+MLO (4 张, 共享 backbone 拼接 4 份特征,
        标签仍取同侧乳房; 对侧缺失时复制同侧图兜底并打印告警)

用法:
  冒烟: python train_vision_baseline.py --version natural --smoke
  正式: python train_vision_baseline.py --version natural
        python train_vision_baseline.py --version repeat_os
  单/四视图: python train_vision_baseline.py --version natural --views cc|mlo|four
  仅推理: python train_vision_baseline.py --version natural --eval-only
  CPU 验证: python train_vision_baseline.py --version natural --views four --cpu --smoke ...
"""

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================== CONFIG ==============================
DENSITY_CLASSES = ["A", "B", "C", "D"]
BIRADS_CLASSES = [1, 2, 3, 4, 5]
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

THIS_DIR = Path(__file__).resolve().parent        # 2-view/scripts/cnn
TWO_VIEW_DIR = THIS_DIR.parent.parent             # 2-view
DATA_DIR = TWO_VIEW_DIR / "data"
OUT_ROOT = TWO_VIEW_DIR / "outputs" / "vision_baseline"

DATA_FILES = {
    "natural": DATA_DIR / "splits" / "train_sub_raw.json",        # 6399 条, 原始分布
    "repeat_os": DATA_DIR / "splits" / "train_sub_repeat_os.json"  # 7019 条, Repeat-OS(t=0.1,cap=4,seed42)
}
VAL_FILE = DATA_DIR / "splits" / "val_raw.json"                            # 1600 条
TEST_FILE = Path("/nesi/project/massey04900/Mammography/test.json")  # 2000 条, 官方 test
# ====================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------ 数据 ------------------------------

VIEWS_N = {"cc": 1, "mlo": 1, "ipsi": 2, "four": 4}


def _study_side(sample_id: str):
    """id = {study}_{L|R} -> (study, side)"""
    study, side = sample_id.rsplit("_", 1)
    return study, side


class TwoViewDataset(Dataset):
    """每条样本: images[0]=CC, images[1]=MLO (构建时固定, 见 make_vlm_splits.py:58)

    views:
      cc/mlo -> 单图; ipsi -> [CC, MLO]; four -> [本侧CC, 本侧MLO, 对侧CC, 对侧MLO]
    __getitem__ 返回 "v0".."v{n-1}" 统一键名 (旧 checkpoint 的模块名/权重不受影响)。
    """

    def __init__(self, path: Path, augment: bool = False, limit: int = 0, name: str = "",
                 views: str = "ipsi", img_size=None):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.views = views
        self.n_views = VIEWS_N[views]

        # four: 对侧索引需建立在完整文件上 (limit 切片只影响训练子集)
        self.contra = {}   # idx -> 对侧样本 (仅 four 使用)
        if views == "four":
            by_study = {}
            for s in raw:
                st, sd = _study_side(s["id"])
                by_study.setdefault(st, {})[sd] = s
            missing = []
            for i, s in enumerate(raw):
                st, sd = _study_side(s["id"])
                other = "L" if sd == "R" else "R"
                c = by_study.get(st, {}).get(other)
                if c is None:
                    missing.append(s["id"])
                self.contra[i] = c
            both = sum(1 for v in by_study.values() if len(v) == 2)
            print(f"[{name}] four-view 完整性: {len(raw)} 乳房 / {len(by_study)} study, "
                  f"双侧齐全 {both} study; 缺对侧 {len(missing)} 乳房 "
                  f"{('例: ' + ', '.join(missing[:3])) if missing else ''} -> 兜底=复制本侧图")

        if limit:
            raw = raw[:limit]
        self.samples = raw
        self.augment = augment
        self.img_size = img_size  # None=原生分辨率; (W,H)=读图后强制 resize (augment 前)

        if augment:
            self.geometric = transforms.RandomAffine(
                degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)
            )  # 轻微 affine, 无翻转
            self.photometric = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # 启动时打印前几条, 确认视图顺序与路径
        print(f"[{name}] N={len(self.samples)} views={views} augment={augment} "
              f"img_size={img_size if img_size else 'native'} src={path}")
        for s in self.samples[:3]:
            with Image.open(s["images"][0]) as im0, Image.open(s["images"][1]) as im1:
                print(f"  id={s['id']}  CC={os.path.basename(s['images'][0])} {im0.size} mode={im0.mode}"
                      f"  MLO={os.path.basename(s['images'][1])} {im1.size} mode={im1.mode}"
                      f"  birads={s['output']['breast_birads']} density={s['output']['breast_density']}"
                      f" findings={[s['output']['findings'][k] for k in FINDINGS]}")

    def _load_views(self, i):
        s = self.samples[i]
        cc, mlo = s["images"][0], s["images"][1]
        if self.views == "cc":
            paths = [cc]
        elif self.views == "mlo":
            paths = [mlo]
        elif self.views == "ipsi":
            paths = [cc, mlo]
        else:  # four
            c = self.contra[i] or s  # 对侧缺失 -> 复制本侧兜底
            paths = [cc, mlo, c["images"][0], c["images"][1]]
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            if self.img_size:
                im = im.resize(self.img_size, Image.BILINEAR)  # 长宽比不变的降采样
            if self.augment:
                im = self.photometric(self.geometric(im))
            imgs.append(self.to_tensor(im))
        return imgs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        o = s["output"]
        out = {
            "id": s["id"],
            "density": DENSITY_CLASSES.index(o["breast_density"]),
            "birads": BIRADS_CLASSES.index(o["breast_birads"]),
            "findings": torch.tensor([float(o["findings"][k]) for k in FINDINGS]),
        }
        for k, im in enumerate(self._load_views(i)):
            out[f"v{k}"] = im
        return out


# ------------------------------ 模型 ------------------------------

class TwoViewMultiHead(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b2", pretrained: bool = True,
                 grad_ckpt: bool = False, n_views: int = 2):
        super().__init__()
        import timm
        self.n_views = n_views
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        d = self.backbone.num_features
        if grad_ckpt:
            try:
                self.backbone.set_grad_checkpointing()
                print(f"gradient checkpointing enabled for {backbone}")
            except Exception as e:
                print(f"WARNING: grad checkpointing not supported: {e}")
        self.head_density = nn.Linear(n_views * d, len(DENSITY_CLASSES))
        self.head_birads = nn.Linear(n_views * d, len(BIRADS_CLASSES))
        self.head_findings = nn.Linear(n_views * d, len(FINDINGS))
        print(f"model: {backbone} feat_dim={d} n_views={n_views} -> "
              f"heads: density={len(DENSITY_CLASSES)}"
              f" birads={len(BIRADS_CLASSES)} findings={len(FINDINGS)}")

    def forward(self, *imgs):
        # 所有视图拼成一个 batch 做一次 forward (BatchNorm 统计量看到 n_views*B 个样本), 再拆特征
        bs = imgs[0].size(0)
        feats = self.backbone(torch.cat(list(imgs), dim=0))
        z = torch.cat([feats[k * bs:(k + 1) * bs] for k in range(self.n_views)], dim=1)
        return self.head_density(z), self.head_birads(z), self.head_findings(z)


def multihead_loss(logit_d, logit_b, logit_f, y_d, y_b, y_f):
    ce_d = F.cross_entropy(logit_d, y_d)
    ce_b = F.cross_entropy(logit_b, y_b)
    bce_f = F.binary_cross_entropy_with_logits(logit_f, y_f)
    total = ce_d + ce_b + bce_f
    return total, {"density": ce_d.item(), "birads": ce_b.item(), "findings": bce_f.item()}


def autocast_ctx(device):
    """cuda 用 bf16 autocast; cpu 用 fp32 nullcontext"""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def batch_views(batch, device):
    """把 "v0".."v{n-1}" 通用键挪到 device, 返回 tensors 列表"""
    i = 0
    imgs = []
    while f"v{i}" in batch:
        imgs.append(batch[f"v{i}"].to(device, non_blocking=True))
        i += 1
    return imgs


# ------------------------------ 训练 ------------------------------

def evaluate(model, loader, device):
    model.eval()
    n, tot = 0, 0.0
    loss_parts = {"density": 0.0, "birads": 0.0, "findings": 0.0}
    d_correct = b_correct = 0
    # findings 混淆: [TP, FP, FN] per class
    fm = [[0, 0, 0] for _ in FINDINGS]
    with torch.no_grad(), autocast_ctx(device):
        for batch in loader:
            imgs = batch_views(batch, device)
            y_d = batch["density"].to(device)
            y_b = batch["birads"].to(device)
            y_f = batch["findings"].to(device)
            logit_d, logit_b, logit_f = model(*imgs)
            loss, parts = multihead_loss(logit_d, logit_b, logit_f, y_d, y_b, y_f)
            bs = imgs[0].size(0)
            tot += loss.item() * bs
            n += bs
            for k in loss_parts:
                loss_parts[k] += parts[k] * bs
            d_correct += (logit_d.argmax(1) == y_d).sum().item()
            b_correct += (logit_b.argmax(1) == y_b).sum().item()
            pred_f = (logit_f.sigmoid() > 0.5).float()
            for j in range(len(FINDINGS)):
                fm[j][0] += ((pred_f[:, j] == 1) & (y_f[:, j] == 1)).sum().item()
                fm[j][1] += ((pred_f[:, j] == 1) & (y_f[:, j] == 0)).sum().item()
                fm[j][2] += ((pred_f[:, j] == 0) & (y_f[:, j] == 1)).sum().item()
    metrics = {
        "val_total_loss": tot / n,
        "val_loss_density": loss_parts["density"] / n,
        "val_loss_birads": loss_parts["birads"] / n,
        "val_loss_findings": loss_parts["findings"] / n,
        "val_density_acc": d_correct / n,
        "val_birads_acc": b_correct / n,
    }
    for j, name in enumerate(FINDINGS):
        tp, fp, fn = fm[j]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        metrics[f"val_{name}_f1"] = f1
        metrics[f"val_{name}_recall"] = r
    model.train()
    return metrics


def predict_jsonl(model, loader, device, out_file: Path):
    """按 VLM scoring 文件格式写概率 jsonl"""
    model.eval()
    results = []
    with torch.no_grad(), autocast_ctx(device):
        for batch in loader:
            imgs = batch_views(batch, device)
            logit_d, logit_b, logit_f = model(*imgs)
            pd_ = logit_d.float().softmax(1).cpu().numpy()
            pb = logit_b.float().softmax(1).cpu().numpy()
            pf = logit_f.float().sigmoid().cpu().numpy()
            for i in range(imgs[0].size(0)):
                gt = {
                    "breast_birads": BIRADS_CLASSES[int(batch["birads"][i])],
                    "breast_density": DENSITY_CLASSES[int(batch["density"][i])],
                    "findings": {k: int(batch["findings"][i][j]) for j, k in enumerate(FINDINGS)},
                }
                pred = {
                    "breast_birads": BIRADS_CLASSES[int(logit_b[i].argmax())],
                    "breast_density": DENSITY_CLASSES[int(logit_d[i].argmax())],
                    "findings": {k: int(pf[i][j] > 0.5) for j, k in enumerate(FINDINGS)},
                }
                probs = {
                    "breast_density": {c: float(pd_[i][j]) for j, c in enumerate(DENSITY_CLASSES)},
                    "breast_birads": {str(c): float(pb[i][j]) for j, c in enumerate(BIRADS_CLASSES)},
                    "findings": {k: {"1": float(pf[i][j])} for j, k in enumerate(FINDINGS)},
                }
                results.append({
                    "id": batch["id"][i],
                    "ground_truth": gt,
                    "greedy_prediction": pred,
                    "probs_norm": probs,
                })
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} records -> {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["natural", "repeat_os"], required=True)
    ap.add_argument("--views", choices=["cc", "mlo", "ipsi", "four"], default="ipsi",
                    help="ipsi=同侧CC+MLO(默认) / cc / mlo 单视图 / four=左右乳各CC+MLO")
    ap.add_argument("--backbone", default="efficientnet_b2",
                    help="timm 模型名, 可切 resnet50")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="等效 batch = batch_size * grad_accum")
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--early-stop-start", type=int, default=8,
                    help="完成第 N 个 epoch 起启用早停检查 (1-indexed)")
    ap.add_argument("--early-stop-patience", type=int, default=4,
                    help="val total loss 连续 N 个 epoch 未刷新最优则停止 (只看 val)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="64 条小子集跑 2 epoch 冒烟")
    ap.add_argument("--eval-only", action="store_true", help="只加载 best.pt 做 val/test 推理")
    ap.add_argument("--smoke-limit", type=int, default=64)
    ap.add_argument("--val-limit", type=int, default=0, help="显式限制 val 条数 (0=不限制; smoke 默认 32)")
    ap.add_argument("--test-limit", type=int, default=0, help="显式限制 test 条数 (0=不限制; smoke 默认 32)")
    ap.add_argument("--cpu", action="store_true", help="CPU 模式 (调试/验证用, fp32)")
    ap.add_argument("--img-size", default="",
                    help="强制 resize 输入, 如 684x1140 (默认空=原生分辨率)")
    ap.add_argument("--out-suffix", default="",
                    help="输出目录名后缀 (如 _lowres -> vision_baseline/repeat_os_lowres/)")
    ap.add_argument("--train-file", default="", help="覆盖训练 json (默认按 --version 取 DATA_FILES)")
    ap.add_argument("--val-file", default="", help="覆盖 val json (默认 val_raw.json)")
    ap.add_argument("--test-file", default="", help="覆盖 test json (默认 NeSI 官方 test.json)")
    args = ap.parse_args()

    img_size = None
    if args.img_size:
        w, h = args.img_size.lower().replace("x", ",").split(",")
        img_size = (int(w), int(h))
        assert img_size[0] > 0 and img_size[1] > 0, f"非法 --img-size: {args.img_size}"

    set_seed(args.seed)
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        assert torch.cuda.is_available(), "需要 GPU"

    tag = f"{args.version}" + ("_smoke" if args.smoke else "")
    if args.views != "ipsi":
        tag += f"_views-{args.views}"
    if args.backbone != "efficientnet_b2":
        tag += f"_{args.backbone}"
    if args.out_suffix:
        tag += args.out_suffix
    work_dir = OUT_ROOT / tag
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== vision baseline [{tag}] ===")
    print(f"work_dir: {work_dir}")
    print(vars(args))

    # ----- 数据 (输入顺序固定 [CC, MLO], Dataset 初始化时打印前几条确认) -----
    train_file = Path(args.train_file) if args.train_file else DATA_FILES[args.version]
    val_file = Path(args.val_file) if args.val_file else VAL_FILE
    test_file = Path(args.test_file) if args.test_file else TEST_FILE
    print(f"[data] train={train_file}\n[data] val={val_file}\n[data] test={test_file}")

    train_ds = TwoViewDataset(
        train_file, augment=True, limit=args.smoke_limit if args.smoke else 0,
        name=f"train_{args.version}", views=args.views, img_size=img_size,
    )
    val_limit = 32 if args.smoke else 0
    if args.val_limit:
        val_limit = args.val_limit
    val_ds = TwoViewDataset(val_file, augment=False, limit=val_limit, name="val", views=args.views,
                            img_size=img_size)
    test_limit = 32 if args.smoke else 0
    if args.test_limit:
        test_limit = args.test_limit
    test_ds = TwoViewDataset(
        test_file, augment=False, limit=test_limit, name="test", views=args.views, img_size=img_size
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- 模型 / 优化器 / 调度器 -----
    model = TwoViewMultiHead(args.backbone, pretrained=True, grad_ckpt=args.grad_ckpt,
                             n_views=VIEWS_N[args.views]).to(device)
    head_params = (
        list(model.head_density.parameters())
        + list(model.head_birads.parameters())
        + list(model.head_findings.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}")

    # ----- 断点续跑 -----
    last_ckpt = work_dir / "last.pt"
    best_ckpt = work_dir / "best.pt"
    start_epoch, best_val, no_improve = 0, float("inf"), 0
    if args.eval_only:
        pass  # best.pt 在下面加载
    elif last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best_val"]
        no_improve = ck.get("no_improve", 0)
        set_seed(args.seed + start_epoch)  # 续跑换 data shuffle 状态
        print(f"resumed from {last_ckpt}: epoch={start_epoch} best_val={best_val:.4f} no_improve={no_improve}")

    metrics_file = work_dir / "metrics.jsonl"

    # ----- 仅推理 -----
    if args.eval_only:
        assert best_ckpt.exists(), f"missing {best_ckpt}"
        ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"eval-only: loaded {best_ckpt} (epoch {ck['epoch']}, val_loss {ck['best_val']:.4f})")
        predict_jsonl(model, val_loader, device, work_dir / "val_probs.jsonl")
        predict_jsonl(model, test_loader, device, work_dir / "test_probs.jsonl")
        return

    # ----- 训练循环 -----
    model.train()
    early_stopped = False
    t_train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        run_loss, nb = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            imgs = batch_views(batch, device)
            y_d = batch["density"].to(device)
            y_b = batch["birads"].to(device)
            y_f = batch["findings"].to(device)
            with autocast_ctx(device):
                loss, _ = multihead_loss(*model(*imgs), y_d, y_b, y_f)
            (loss / args.grad_accum).backward()
            if (it + 1) % args.grad_accum == 0 or (it + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            run_loss += loss.item()
            nb += 1
            step = it + 1
            if step % 10 == 0 or step == len(train_loader):
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} {elapsed:.0f}s", flush=True)
        train_loss = run_loss / max(1, nb)

        # ----- 验证 (best 只按 val total loss 选, test 不参与) -----
        metrics = evaluate(model, val_loader, device)
        metrics.update({"epoch": epoch, "train_loss": train_loss,
                        "lr_backbone": scheduler.get_last_lr()[0],
                        "epoch_sec": round(time.time() - t0, 1)})
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        is_best = metrics["val_total_loss"] < best_val
        if is_best:
            best_val = metrics["val_total_loss"]
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_val": best_val, "args": vars(args)}, best_ckpt)
        else:
            no_improve += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_val": best_val, "no_improve": no_improve,
                    "args": vars(args)}, last_ckpt)
        star = " *best*" if is_best else ""
        print(f"epoch {epoch:3d}/{args.epochs} train={train_loss:.4f} "
              f"val={metrics['val_total_loss']:.4f} d_acc={metrics['val_density_acc']:.4f} "
              f"b_acc={metrics['val_birads_acc']:.4f} "
              f"mass_f1={metrics['val_mass_f1']:.3f} "
              f"({metrics['epoch_sec']}s){star}", flush=True)

        # 早停: 从第 early_stop_start 个 epoch 起检查, val total loss 连续 patience 个 epoch 未刷新最优
        if (epoch + 1) >= args.early_stop_start and no_improve >= args.early_stop_patience:
            print(f"=== early stop: val loss 连续 {no_improve} 个 epoch 未刷新最优 "
                  f"(epoch {epoch}, best_val={best_val:.4f}) ===", flush=True)
            early_stopped = True
            break

    train_hours = (time.time() - t_train_start) / 3600
    print(f"=== training done: {train_hours:.2f}h, best val loss = {best_val:.4f} ===")

    # ----- 用 best.pt 推理 val + test -----
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"best epoch = {ck['epoch']}")
    predict_jsonl(model, val_loader, device, work_dir / "val_probs.jsonl")
    predict_jsonl(model, test_loader, device, work_dir / "test_probs.jsonl")
    with open(work_dir / "DONE.json", "w") as f:
        json.dump({"best_epoch": ck["epoch"], "best_val_loss": best_val,
                   "train_hours": round(train_hours, 3),
                   "early_stopped": early_stopped,
                   "completed_epochs": epoch + 1,
                   "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
