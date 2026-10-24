# 原分辨率 ep2 实验正确性核查（只读，CPU）

核查时间：2026-10-24；未占用 GPU、未打断链条试验。
结论：**5 项全部通过**；其中第 1 项有一处需注明的细节（训练/推理图像 token 数相差 2.1%）。

| # | 项目 | 结论 |
|---|---|---|
| 1 | 训练真的用了原分辨率 | **通过**（附注：训练 1316 token/图 vs 推理 1344 token/图） |
| 2 | 推理真的用了原分辨率 | **通过** |
| 3 | 对照组一致（步数/数据/seed/lr） | **通过** |
| 4 | 训练是否正常（loss 曲线） | **通过** |
| 5 | 冒烟临时目录未污染正式输出 | **通过** |

---

## 1. 训练真的用了原分辨率 —— 通过

**缓存**：`data/tokenized/ep2_native/train/data-00000-of-00001.arrow`（283,868,008 B，mtime `2026-10-23 20:22`，共 7019 行）

抽样 5 条（idx = 0 / 1000 / 3000 / 5000 / 7000）：

| idx | len(input_ids) | 图像 token 总数 | 每图 | 图数 |
|---|---|---|---|---|
| 0 | 3084 | 2632 | 1316 | 2 |
| 1000 | 3084 | 2632 | 1316 | 2 |
| 3000 | 3084 | 2632 | 1316 | 2 |
| 5000 | 3084 | 2632 | 1316 | 2 |
| 7000 | 3084 | 2632 | 1316 | 2 |

- 序列 3084（≈ 1316×2 + 452 文本），与“原分辨率、序列约 3100+”一致
- 对照：用同一 processor 按 `image_max_pixels=786432` 复算同一批真实图片 → `grid=[1,70,42]`、**735 token/图**，序列约 735×2+452 ≈ **1922**（与“约 756/1960”同量级，差异来自取整）

**缓存确为本轮生成、非普通分辨率复用**（`logs/vlm/native_ep2.log`）：
```
[2026-10-23 20:12:01] Loading dataset llamafactory/train_llamafactory_os_sub_local.json...
[2026-10-23 20:22:50] Tokenized dataset is saved at .../data/tokenized/ep2_native.
```
即 20:12 启动时还没有缓存、20:22 由本轮预处理写出；缓存内 token 数（1316/图）远高于普通分辨率的 735/图，不可能是 786432 的缓存。

**附注（需在报告里注明）**：训练侧 1316 token/图（896×1504），推理侧同一图片为 1344 token/图（896×1536），相差 2.1%。CPU 复算（`AutoProcessor` + `image_max_pixels=1376256`，快/慢处理器都试过）一律得到 `[1,96,56]=1344`，说明缓存生成时的对齐取整与推理时不同（向下对齐 32 vs 就近对齐）。不影响“原分辨率”结论，但属于训练/推理预处理不完全一致，建议在实验记录中标注。

## 2. 推理真的用了原分辨率 —— 通过

**实际命令**（`scripts/bench/native_master.sh:126-128`）：
```bash
say "b8 推理开始（2000 条，image_max_pixels=1376256）"
"$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$OUT" \
  --data data/splits/test_local.json --limit 0 --batch-size 8 --image-max-pixels 1376256 ...
# $OUT = /home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep2-native
```

**运行日志**（`logs/vlm/native_master.log`）：
```
:600 Loaded adapter(s): /home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep2-native
:29  b8 推理开始（2000 条，image_max_pixels=1376256）
:30  [data] total=2000 pending=2000 batch=8
:606 [done] cases=2000 batches=250 total=31.8 min, gen=0.76s/case (batch=8)
:609 推理结束：2000/2000
```
adapter 是原分辨率输出目录（不是普通 ep2），像素上限 1376256，batch=8，2000 条齐全。

**CPU 复算 image_grid_thw（2 个测试样本）**：
```
max_pixels=1376256 id=0025a5dc99fd5c742026f0b2b030d3e9_L 原图=(912,1520) grid=[[1,96,56],[1,96,56]] 1344/图
max_pixels=1376256 id=0025a5dc99fd5c742026f0b2b030d3e9_R 原图=(912,1520) grid=[[1,96,56],[1,96,56]] 1344/图
```
与期望的 `[1,96,56]` 完全一致。

产物：`outputs/scoring/vt-ep2-native_test_b8_with_probs.jsonl` 2000 行（2026-10-24 10:26）。

## 3. 对照组一致 —— 通过

| 项 | 普通 ep2 | 原分辨率 ep2 |
|---|---|---|
| 输出目录 | `...-os-sub-vt-ep2` | `...-os-sub-vt-ep2-native` |
| global_step / max_steps | 878 / 878 | 878 / 878 |
| epoch | 2.0 | 2.0 |
| 数据 | `vindr_2view_train_os_sub_local`（7019 条） | 同 |
| lr / schedule | 5e-5 / cosine / warmup_ratio 0.05 | 同 |
| bs × accum | 1 × 16 = 16 | 同 |
| lora | rank 8 / alpha 16 / dropout 0.05 / all-linear，视觉塔解冻 | 同 |
| seed | 未显式设置 | 未显式设置 |

- 两者都用**最终 878 步**的 adapter（`checkpoint-878` 存在，最终目录另存 `adapter_model.safetensors`）
- 配置差异经 `git diff` 确认只有两处：新增 `tokenized_path`、`save_steps: 200 → 100`；`image_max_pixels`、lr、bs、epoch 均未变（配置头注释里的“1386240”是旧注释）
- seed：两个 yaml 都没写 seed → 走 `TrainingArguments` 默认 `seed: int = 42`（`transformers/training_args.py:2185`），llamafactory 在 `parser.py:657` 用 `set_seed(training_args.seed)` 生效 → 两次训练 seed 相同

## 4. 训练是否正常 —— 通过

`trainer_state.json` 的 loss 记录（87 个记录点，取 step 10/220/440/660/870）：

| step | 普通 ep2 | 原分辨率 ep2 |
|---|---|---|
| 10 | 0.5201 | 0.5125 |
| 220 | 0.0308 | 0.0313 |
| 440 | 0.0238 | 0.0231 |
| 660 | 0.0220 | 0.0225 |
| 870 | 0.0195 | 0.0191 |

- `train_loss`：0.032371 vs 0.032220；`total_flos`：6.50e17 vs **1.02e18**（原生分辨率计算量约 1.57×，与 token 数 2632/1470≈1.79、图像更大相符）
- 两条曲线形态、量级、收敛点几乎完全一致，原分辨率**没有** loss 偏高或未收敛

## 5. 冒烟临时目录未污染正式输出 —— 通过

- `scripts/bench/native_train_chain.sh:31,42,74`：冒烟用独立目录 `/model/_smoke_native_ep2`，结束时 `rm -rf "$SMOKE_OUT" "$SMOKE_CFG"`
- 实际检查：`model/_smoke_native_ep2` **不存在**
- 正式目录 `...-ep2-native/` 只有 `checkpoint-800`、`checkpoint-878`（无冒烟的小步数 checkpoint）
- 冒烟 16 条推理产物单独命名 `outputs/scoring/smoke_native_16_with_probs.jsonl`（16 行），正式文件 `vt-ep2-native_test_b8_with_probs.jsonl` 内 **0 次** 出现 smoke 标记，首行 id 与测试集一致

---

## 附：本次核查用到的命令要点

- 缓存抽样：`datasets.load_from_disk` 读 arrow，统计 `image_token_id=248056` 出现次数
- grid 复算：`AutoProcessor.from_pretrained(model/Qwen3.5-4B)`，设 `image_processor.size={'longest_edge':max,'shortest_edge':262144}`
- 全程 CPU、只读；链条试验（tmux `pilot`）未受影响
