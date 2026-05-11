# 二进制文件漏洞检测 — 行动日志

## v2.0 (2026-05-10) 🔧 源码就绪，待训练

v1.9 分析结论：seed 42 (emb=48) 达 0.931 超越 v1.5 最佳，异构 emb 无效，种子方差 0.063 是核心瓶颈

**v2.0 三项改动：**
1. **统一 emb=48** — 去掉异构 SEED_CONFIGS，四种子同质。seed 42 证明这是最优配置
2. **Self-attention in byte encoder** — Conv blocks 之后加 MultiheadAttention(4 heads, 320 dim) + residual + LayerNorm，捕捉字节序列中的长程依赖
3. **稀有 CWE 增强** — ≤5 样本的类复制增广（字节翻 1% + 表格加高斯噪声 σ=0.01），补到至少 10 条
4. 训练策略不变：CosineAnnealingWarmRestarts + warmup 3 epoch + SWA + epochs=30

## v1.9 (2026-05-10) ✅ 已完成并归档 — 平台 0.921

- 不改架构，只改**训练策略**，目标降低种子间方差
- **CosineAnnealingWarmRestarts** (T_0=10) 替代 ReduceLROnPlateau — 周期性重启 LR 帮差种子跳出局部最优
- **3-epoch warmup** — LR 从 1% 线性爬升，减少初期震荡
- **SWA** (Stochastic Weight Averaging) — 最后 5 epoch 权重平均
- **epochs 25→30** — v1.8 seed 456/789 在 epoch 25 还在涨
- 其余同 v1.8：加权集成 + scalar fusion + TTA + 同质 emb=48

## v1.8 (2026-05-09) ✅ 已完成并归档 — 平台 0.911

- 变更：epochs 20→25，**去掉逐 CWE 类阈值**（v1.7 平台 0.876 的元凶，砍了 1,387 个 label=1）
- v1.7→v1.8 最小 diff：只动了两处关键参数，其余逻辑不变
- 伪标签同 v1.7（v1.5 的 17,343 条 0.9 阈值高质量伪标签）
- 加权集成 + scalar fusion + TTA 3 窗口保留
- 预期不低于 v1.5 的 0.923

## v1.7 (2026-05-09) ❌ 已完成并归档 — 平台 0.876

- 平台提交成绩：**0.876**（v1.5: 0.923，**-4.7%**）
- 本地 avg scalar_cwe_macro：**0.8862**（v1.5: 0.8980，-1.2%）
- 根因：逐 CWE 类阈值在训练集 val 上拟合，搬到测试集后过度降级 1,387 个 label=1 预测
  - v1.5 label=1 占比 50.2%，v1.7 降到 45.7%
- 次要原因：epochs 降到 20 导致种子质量退化（最佳单种子 0.911 vs v1.5 的 0.928）
- 其他功能正常：加权集成、scalar fusion、TTA 均运行正确

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

---

## v1.5 (2026-05-08) ✅ 已完成并归档

- 平台提交成绩：**0.923**（v1.4: 0.901，+2.2%）
- 本地 avg scalar_cwe_macro：**0.8980**（v1.4: 0.8739，+2.4%）
- 最佳单种子（seed 42）本地 scalar_cwe=0.9281，折算平台 ~0.957，证明模型能力已达 0.95 级别
- 种子间方差：0.928 → 0.893 → 0.873（差距 0.055），集成平均是主要损失来源

**v1.5 改进项：**
- 伪标签扩充：v1.4 模型标注 test.csv，获 17,343 条高置信度伪标签（56%），训练集 39,380 → 56,723
- 3-seed 集成：seeds=[42, 123, 456]，平均预测降低方差
- MLP fusion 修复：正确 train/val/test 划分训练 FusionMLP，但 held-out 分数（0.72）不如 scalar（0.90），最终使用 scalar
- TTA：推理时 3 个字节窗口平均，提升鲁棒性
- 集成配置自动选优：训练完成后自动对比 scalar vs MLP 融合分数，选择最佳模式

**归档（精简模式）：**
- 版本记录/v1.5/源码/（8 个 .py 文件）
- 版本记录/v1.5/提交结果/submission_v1.5.csv
- 版本记录/v1.5/README.md、requirements.txt、ACTION_LOG.md、行动日志.md
- 版本记录/v1.5/模型/复现说明.md（模型文件留在当前工作区 模型/，不重复复制）

**当前工作区模型文件（v1.5）：**
- 3 × label_model_v1.5_seed*.joblib + 3 × cwe_model_v1.5_seed*.joblib
- 3 × neural_bundle_v1.5_seed*.pt + 3 × fusion_mlp_v1.5_seed*.pt
- fusion_config_v1.5.json + tabular_bundle_v1.5.joblib + 缓存文件
- pseudo_train_v1.5.csv（项目根目录）

**新工作流**：归档只保存源码/结果/文档/日志，模型留在工作区，归档处建 `模型/复现说明.md` 记录文件清单和复现步骤。

---

## v1.6 (2026-05-08) ❌ 伪标签噪声导致退化

- 平台提交成绩：**0.904**（v1.5: 0.923，**-1.9%**）
- 本地 avg scalar_cwe_macro：**0.9247**（v1.5: 0.8980，+2.7% 但虚高）
- 本地/平台严重背离：本地 +0.027 但平台 -0.019
- 根因：伪标签阈值从 0.9 降至 0.7，新增 10,878 条 0.7-0.9 置信度样本，其中 56% label=1（vs v1.5 的 43%），大量 CWE 标注错误导致模型过拟合噪声

**v1.6 改进项：**
- 伪标签阈值降至 0.7/0.3，覆盖率 56%→91%（28,216 条）
- 4-seed 异构集成：emb=48/64/48/56，drop=0.22/0.26/0.18/0.24
- 最佳单种子 seed 789（emb=56）本地 0.9338
- 去掉 MLP fusion（只保留 scalar）

**归档（精简模式）：**
- 版本记录/v1.6/源码/、提交结果/、README.md、requirements.txt、ACTION_LOG.md
- 版本记录/v1.6/模型/复现说明.md

**教训**：伪标签不是越多越好。低置信度样本（0.7-0.9）在 86 类任务上 CWE 错误率极高，训练数据质量比数量重要。后续版本回退到 v1.5 的高质量伪标签。

---

## 2026-05-08 会话总结

### 完成事项
- v1.5 平台 0.923，归档完成
- v1.6 平台 0.904（退化），已归档并标注 ❌
- v1.7 代码完成，训练中：回退 v1.5 伪标签（17,343 条，0.9 阈值）+ v1.6 架构（4-seed 异构）+ 加权集成 + 逐 CWE 类阈值 + 跨版本缓存复用
- 版本归档标准工作流建立（`版本归档工作流.md`），模型留工作区、归档放复现说明
- 系统日志赛题优化建议已写入其目录

### 代码改进
- **跨版本特征缓存复用**：train.py 扫描已有缓存匹配 binary_ids，测试缓存去版本号。v1.7 复用了 v1.5 的 tabular 和 byte 缓存（省 70 分钟）
- 确定性推理：test.py 添加 `cudnn.deterministic=True`
- train.py 伪标签函数使用 v1.5 ensemble 模型（跨版本模型路径替换）

### 关键技术发现
1. **伪标签最优阈值**：0.9 有效（+0.022），0.7 反噬（-0.019）。86 类任务低置信度 CWE 错误率高
2. **异构集成有效**：不同 emb_dim/dropout 增加多样性，最佳 seed 789（emb=56）达 0.9338
3. **种子间方差是主要瓶颈**：0.934 vs 0.914，差距 0.02，加权集成可缓解
4. **MLP fusion 对过拟合敏感**：train=val 时虚高，正确划分后仅 0.72
5. **特征缓存跨版本可复用**：binary_ids 匹配即可直接复制，无需重建

### 写入记忆的持久教训
- 训练崩溃后清理残留进程
- 提交前检查 fusion_config 的 mode 和 threshold
- 平台评分 = 87 类 Macro-F1（含 label=0 类）
- 长任务完成立刻通知用户
- 跨版本复用特征缓存（只提取增量）
