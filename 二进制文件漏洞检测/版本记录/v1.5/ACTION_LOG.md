# 二进制文件漏洞检测 — 行动日志

## v1.4 (2026-05-08) ✅ 已完成并归档

- 本地验证成绩：**scalar_cwe_macro_f1=0.8739**（v1.3: 0.830，+4.4%）
- 平台提交成绩：**0.901**（v1.3 平台: 0.849，+5.2%，**大幅超 87% 目标**）
- 平台评分标准：Macro-F1 覆盖 87 类（label=0 + 86 CWE），本地仅对真实正例算 CWE macro-F1，两者口径不同
- neural_cwe_macro_f1: 0.8338（v1.3 neural: ~0.59，+41%）
- neural_label_f1: 0.9707
- scalar_label_f1: 0.9740
- fusion 权重：tree/neural label=0.525/0.475, cwe=0.475/0.525 → neural 首次成为主导

源码改进：
  - Focal Loss (gamma=1.5) 替换 CrossEntropyLoss for CWE head
  - LightGBM 替换 RF+ET VotingClassifier for CWE tabular model
  - 更深字节编码器：embedding 24→48，4→5 conv blocks，更宽 MLP
  - MLP 元学习器融合（FusionMLP）替代单标量权重
  - 移除 WeightedRandomSampler，改回标准 shuffle
  - 每 epoch 打印验证分数

归档：
  - 版本记录/v1.4/源码/（8 个 .py 文件）
  - 版本记录/v1.4/模型/（12 个文件，含 neural_bundle_v1.4.pt + fusion_mlp_v1.4.pt）
  - 版本记录/v1.4/提交结果/submission_v1.4.csv

当前工作区产物：
  - 提交结果/submission_v1.4.csv
  - 模型/ 下完整 v1.4 模型文件

**教训**：训练崩溃后必须立刻清理残留进程。三次僵尸进程累计占用 5.2GB 是后续 OOM 根因。

**v1.4 threshold bug 记录**：首版 submission 因 `fusion_config_v1.4.json` 中 `fusion_mode="mlp"` + `fusion_threshold=0.08` 导致平台仅得 0.81（v1.3 为 0.849）。根因是 MLP fusion 输出概率过度集中，0.08 阈值使几乎所有样本判为 label=1，非漏洞样本的 CWE 全部预测错误，所有 CWE 类 precision 崩溃。修复为 `fusion_mode="scalar"` + `fusion_threshold=0.5` 后平台得 0.901。**以后每个版本提交前必须先检查 fusion_config 的 mode 和 threshold。**

**评分标准存档**：竞赛评分 = Macro-F1 覆盖 87 个类别（1 个 "无漏洞" label=0 + 86 个 CWE 类别）。本地 `scalar_cwe_macro_f1` 只在真实正例上算 CWE macro-F1，忽略 label 预测错误，口径偏乐观。参考文档：`D:\桌面\题解\二进制\二进制漏洞检测竞赛完整说明文档.pdf`。
