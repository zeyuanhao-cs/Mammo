# 原分辨率 ep2 训练/评测状态
[2026-10-23 19:42:07] 冒烟开始：10 步，第 5 步存 ckpt，第 8 步 kill
[2026-10-23 19:54:18] 总控启动：等待冒烟测试（smoke_native）结束
[2026-10-23 19:55:54] 总控启动：等待冒烟测试（smoke_native）结束
[2026-10-23 20:01:12] 第 8 步：kill 训练，随后自动续跑
[2026-10-23 20:01:18] 可用 checkpoint-5
[2026-10-23 20:06:36] 冒烟训练结束；nvidia-smi 峰值=21902 MiB
0
[2026-10-23 20:06:36] 用冒烟 checkpoint 推理 16 条（image_max_pixels=1376256）
[2026-10-23 20:07:24] 推理产出 16/16 条
[2026-10-23 20:07:24] 冒烟 PASS
[2026-10-23 20:07:24] 临时目录已删除
[2026-10-23 20:07:24] 冒烟进程已结束
[2026-10-23 20:07:24] 运行冒烟三项检查
[2026-10-23 20:07:24] STOP：冒烟自检未通过：FAIL: 续跑首步是 0，说明没从 checkpoint-5 接着跑 
[2026-10-23 20:11:05] 总控启动：等待冒烟测试（smoke_native）结束
[2026-10-23 20:11:05] 冒烟进程已结束
[2026-10-23 20:11:05] 运行冒烟三项检查
[2026-10-23 20:11:05] 冒烟自检 PASS：PASS: expandable=0, 续跑=global step 5 起, 到达 10/10, loss 条数=13, probs=16/16, json=16/16
[2026-10-23 20:11:06] 冒烟日志中 tokenizer 渲染次数=10
[2026-10-23 20:11:06] 已写入 tokenized_path=/home/z/zhao/Mammography/Mammo/2-view/data/tokenized/ep2_native
[2026-10-23 20:11:06] 正式训练开始（896x1536, 2 epoch, save 每 100 步, keep 2）
[2026-10-23 20:11:06] 第 1/3 次尝试：从头开始
[2026-10-23 20:11:11] 第 1/3 次尝试失败 rc=1（global_step=err）：  in "/home/z/zhao/Mammography/Mammo/2-view/configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml", line 17, column 1 found duplicate key tokenized_path   in "/home/z/zhao/Mammography/Mammo/2-view/configs/train_lora_2view_os_sub_vt_ep2_native_local.yaml", line 38, column 1 
[2026-10-23 20:11:56] 总控启动：等待冒烟测试（smoke_native）结束
[2026-10-23 20:11:56] 冒烟进程已结束
[2026-10-23 20:11:56] 运行冒烟三项检查
[2026-10-23 20:11:56] 冒烟自检 PASS：PASS: expandable=0, 续跑=global step 5 起, 到达 10/10, loss 条数=13, probs=16/16, json=16/16
[2026-10-23 20:11:56] 冒烟日志中 tokenizer 渲染次数=10
[2026-10-23 20:11:56] tokenized_path 已存在，跳过
[2026-10-23 20:11:56] 正式训练开始（896x1536, 2 epoch, save 每 100 步, keep 2）
[2026-10-23 20:11:56] 第 1/3 次尝试：从头开始
[2026-10-24 09:54:41] 第 1/3 次尝试成功（rc=0, global_step=878, adapter 存在）
[2026-10-24 09:54:41] 训练阶段结束 ok=1；nvidia-smi 峰值=21893 MiB
[2026-10-24 09:54:41] 产物校验通过：adapter 存在，global_step=878
[2026-10-24 09:54:41] b8 推理开始（2000 条，image_max_pixels=1376256）
[2026-10-24 10:26:44] 推理结束：2000/2000
[2026-10-24 10:27:24] metrics 已生成：metrics_probs_vt-ep2-native_test_b8.json
[2026-10-24 10:27:24] 生成 ep2_native_report.md
[2026-10-24 10:27:24] STOP：报告生成失败，见 /home/z/zhao/Mammography/Mammo/2-view/logs/vlm/native_master.log
[2026-10-24 12:13:48] 报告补生成完成（ep2_native_report.py 解包 bug 修复后重跑成功）
