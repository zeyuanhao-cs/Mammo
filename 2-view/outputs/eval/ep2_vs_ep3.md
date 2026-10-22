# ep2 (2 epochs) vs ep3 (3 epochs) — b8 流水线

## 验证集 val (1600)

| run | density AUC | BI-RADS AUC | findings AUROC | density F1 | BI-RADS F1 | findings F1 | density bACC | BI-RADS bACC | BI-RADS ECE | 检出率 |
|---|---|---|---|---|---|---|---|---|---|---|
| ep2-e1 (ckpt439) | (metrics 缺失) | | | | | | | | | |
| ep2-e2 (ckpt878) | (metrics 缺失) | | | | | | | | | |
| ep3-e1 (ckpt439) | 0.9329 | 0.8228 | 0.8022 | 0.4887 | 0.4385 | 0.2744 | 0.4550 | 0.4423 | 0.1180 | 0.2866 (45/157) |
| ep3-e2 (ckpt878) | 0.9439 | 0.8401 | 0.8184 | 0.6432 | 0.5074 | 0.2963 | 0.5881 | 0.5073 | 0.0823 | 0.3567 (56/157) |
| ep3-e3 (ckpt1317) | 0.9426 | 0.8433 | 0.8188 | 0.6252 | 0.5476 | 0.3115 | 0.5742 | 0.5265 | 0.0748 | 0.3248 (51/157) |

## 测试集 test (2000)

| run | density AUC | BI-RADS AUC | findings AUROC | density F1 | BI-RADS F1 | findings F1 | density bACC | BI-RADS bACC | BI-RADS ECE | 检出率 |
|---|---|---|---|---|---|---|---|---|---|---|
| ep2-e1 (ckpt439) | (metrics 缺失) | | | | | | | | | |
| ep2-e2 (ckpt878) | (metrics 缺失) | | | | | | | | | |
| ep3-e1 (ckpt439) | 0.9316 | 0.8283 | 0.8447 | 0.4967 | 0.4334 | 0.2352 | 0.4698 | 0.4216 | 0.1216 | 0.3229 (62/192) |
| ep3-e2 (ckpt878) | 0.9461 | 0.8437 | 0.8567 | 0.6071 | 0.4823 | 0.2863 | 0.5886 | 0.4777 | 0.0879 | 0.3854 (74/192) |
| ep3-e3 (ckpt1317) | 0.9442 | 0.8495 | 0.8589 | 0.6058 | 0.5187 | 0.2863 | 0.5789 | 0.4947 | 0.0810 | 0.3594 (69/192) |

## 说明

- 全部为 batch=8 流水线（`infer_2view_batched.py` + `summarize_probs.py`）。
- ECE 为 BI-RADS top-label ECE；检出率为 GT 有病灶乳房中预测任一 finding 阳性的比例。
- ep2 与 ep3 训练数据/超参完全相同，仅 num_train_epochs 2.0 vs 3.0。
