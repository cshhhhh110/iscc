# 总行动日志

## 总体规则

- 原始版本统一命名为 `original`
- 新迭代版本按 `v1.0`、`v1.1`、`v2.0` 这样的方式递增
- 版本保留范围覆盖 `源码/`、`模型/`、`提交结果/`
- 当前工作区只保留最新可运行版本，历史版本统一进入 `版本记录/`
- 各项目的项目级日志继续保留，本文件只做跨项目索引与汇总
- 每次记录迭代时，统一写明：当前版本、归档路径、模型文件名、结果文件名、root 清理是否完成

## 项目索引

| 项目 | 当前版本 | 主要产物 | 项目级日志 |
|---|---|---|---|
| powershell恶意脚本检测 | v3.7_align ✅ 平台 0.70831 | `版本记录/v3.7_align/`，exact_key5特征对齐，class1 +46 | `powershell恶意脚本检测/ACTION_LOG.md` |
| 二进制文件漏洞检测 | v1.5 ✅ 平台 0.923 | `提交结果/submission_v1.5.csv`，平台 Macro-F1=0.923，伪标签+3-seed集成+TTA | `二进制文件漏洞检测/ACTION_LOG.md` |
| 系统日志异常检测挑战 | v1.1 | `系统日志异常检测挑战/版本记录/v1.1/模型/model_bundle.joblib`，`系统日志异常检测挑战/版本记录/v1.1/提交结果/submission.csv` | `系统日志异常检测挑战/ACTION_LOG.md` |
| 网络安全智能分类挑战 | v1.3（源码已就绪，模型/提交结果待本机训练） | `版本记录/v1.3/源码/`，`模型/gpu_model_bundle_v1.3.pt`，`提交结果/submission_gpu_v1.3.csv` | `网络安全智能分类挑战/ACTION_LOG.md` |

## 记录模板

- 日期：
- 项目：
- 当前版本：
- 归档路径：
- 模型文件名：
- 结果文件名：
- root 清理：
- 备注：

## 执行记录

- 2026-05-05：创建总行动日志与总要求，建立统一版本保留规则。
- 2026-05-05：为四个项目建立 `版本记录/original/` 快照框架，后续版本将按 `v1.0`、`v1.1` 继续追加。
- 2026-05-05：当前各项目的原始基线与最新结果已纳入版本管理，历史版本不再覆盖当前工作目录。
- 2026-05-06：系统日志异常检测挑战完成 v1.1 迭代并归档到 `系统日志异常检测挑战/版本记录/v1.1/`，OOF score 提升到 `0.922548`。
- 2026-05-06：为 `版本记录/v1.1/` 补充了带版本后缀的源码、模型与结果副本，便于按版本号快速识别文件。
- 2026-05-05：为四个项目的 `README.md` 顶部加入“先读总规则”提示，强制把总要求前置到项目入口。
- 2026-05-06：补充工作区卫生规则与固定日志字段，要求 `源码/` 只保留当前活跃版本，`smoke_*` 和历史产物归档或清理，日志固定记录版本、路径、模型名、结果名和 root 清理状态。
- 2026-05-06：完成四个项目当前工作区的 `smoke_*` 临时件清理，root 与当前 `模型/`、`提交结果/` 仅保留正式产物，版本记录未动。
- 2026-05-06：网络安全智能分类挑战启动 v1.3 纯 GPU 稳健提分迭代，新增 `gpu_v1_3_core.py`、`train_gpu_v1_3.py`、`predict_gpu_v1_3.py`，并归档到 `网络安全智能分类挑战/版本记录/v1.3/`。
- 2026-05-06：v1.3 smoke 在 Codex 沙箱内已完成 2 折 2 epoch 的训练循环并通过编译检查，但 `numpy.save` / bundle 写入被沙箱 Python 写权限拦截；正式模型与提交结果需由你在本机运行 smoke / full training 生成。
- 2026-05-07：计划迁移总目录到 `E:\赛题数据`；方式为复制后确认；源路径 `D:\桌面\赛题数据`，目标路径 `E:\赛题数据`；状态为迁移前记录，待复制校验。

## 项目状态

### powershell恶意脚本检测

- 当前状态：v3.7_align ✅ 平台 0.70831
- 当前结果：`版本记录/v3.7_align/提交结果/submission_v3.7_align.csv`
- 当前指标：平台 Macro-F1=0.70831（v3.3: 0.70441，+0.0039），class1 3308→3354 (+46)
- 关键改动：exact_key5 特征对齐（132→137维），根因修复 teacher/student 表示不匹配
- 版本归档：`版本记录/original/` ~ `版本记录/v3.7_align/`

### 二进制文件漏洞检测

- 当前状态：v1.5 ✅ 平台 Macro-F1=0.923，达标
- 当前结果：本地 avg scalar_cwe=0.8980，平台=0.923
- 当前模型：3-seed 集成（label/cwe/neural/fusion 各 3 份），scalar fusion，TTA 3 窗口
- 当前源码：`源码/`（伪标签+多种子集成+proper MLP fusion+TTA）
- 版本归档：`版本记录/original/`

### 系统日志异常检测挑战

- 当前状态：已完成 v1.1，并保留可复现归档
- 当前结果：`版本记录/v1.1/提交结果/submission.csv`
- 当前指标：OOF score `0.922548`
- 版本归档：`版本记录/v1.1/源码`、`版本记录/v1.1/模型`、`版本记录/v1.1/提交结果`

### 网络安全智能分类挑战

- 当前状态：v1.3 源码已就绪，正式模型与提交结果待本机训练生成
- 当前结果：`版本记录/v1.3/源码/`，根目录保留 v1.3 活跃源码
- 当前指标：v1.2 训练结果已归档；v1.3 smoke 已跑到保存前一步，待本机补全
- 版本归档：`版本记录/original/源码`、`版本记录/original/模型`、`版本记录/original/提交结果`，`版本记录/v1.1`，`版本记录/v1.2`，`版本记录/v1.3/源码`
[2026-05-06 08:47:15] 网络安全智能分类挑战 v1.1 FT-Transformer smoke completed; outputs archived under 版本记录/v1.1/, smoke macro_f1=0.158175, accuracy=0.220833, user will run the full training later.
[2026-05-06 10:04:33] 网络安全智能分类挑战 v1.2 pure GPU dual-model smoke completed; selected=ensemble; smoke macro_f1=0.438022, accuracy=0.459722; results archived under 版本记录/v1.2/.
[2026-05-06 22:15:00] v1.4 FT-Transformer source prepared for 网络安全智能分类挑战: active root source switched to v1.4; archived source snapshot copied to 版本记录/v1.4/源码.
[2026-05-06 22:15:00] v1.4 project structure updated: root source now keeps only the active v1.4 scripts plus common/validate helpers; version record directories created for source/model/result snapshots.
- 2026-05-07 01:22:03 powershell恶意脚本检测 v1.2 versioning applied: active artifacts are model_bundle_v1.2.joblib, validation_report_v1.2.json, and submission_v1.2.csv; OOF Macro-F1=0.755080; snapshot archived under powershell恶意脚本检测/版本记录/v1.2/.
- 2026-05-07 02:27:03 powershell v1.3 source upgrade completed: pattern_lookup candidate added, 4-way fusion simplex enabled, active defaults bumped to v1.3, and smoke-tested; training pending user execution.
- 2026-05-07：总目录已复制迁移到 `E:\赛题数据`，源目录 `D:\桌面\赛题数据` 暂保留作为回滚副本；校验结果为文件数 70801 对 70801、总字节 53456988782 对 53456988782，完全一致；root 清理状态：未改动项目内容，仅迁移目录。
- 2026-05-07 17:08:27 powershell恶意脚本检测 v1.3; selected=fusion; oof_macro_f1=0.755433
- 2026-05-07 20:42:54 powershell恶意脚本检测 v1.4; selected=fusion; oof_macro_f1=0.755713
- 2026-05-07 22:34:15 powershell恶意脚本检测 v1.5; selected=fusion; oof_macro_f1=0.755537
- 2026-05-08：二进制文件漏洞检测 v1.4 完成，本地 scalar_cwe_macro_f1=0.8739（v1.3: 0.830，+4.4%），已完整归档到版本记录/v1.4/（源码/模型/提交结果）。（教训：训练崩溃后未及时清理僵尸进程导致 OOM，浪费数小时；已写入永久记忆和项目日志。）
- 2026-05-08：v1.4 首次平台提交仅得 0.81（因 fusion_mode="mlp" + threshold=0.08 bug），修复为 scalar + threshold=0.5 后重新提交，**平台最终得分 0.901**（v1.3 平台: 0.849，+5.2%），大幅超 87% 目标。评分标准已存档：平台 Macro-F1 覆盖 87 类（label=0 + 86 CWE），本地仅算真实正例 CWE macro，口径偏乐观。
- 2026-05-08：v1.6 完成，4-seed 异构集成 + v1.5 模型伪标签扩充至 28,216 条（阈值 0.7，91% 覆盖）。本地 avg scalar_cwe=0.9247 但**平台仅 0.904**（v1.5: 0.923，-1.9%）。根因：低置信度伪标签（0.7-0.9）含大量 CWE 噪声，导致过拟合。教训：伪标签质量 > 数量。已归档。
- 2026-05-10 22:31:05 powershell恶意脚本检测 v1.6; selected=fusion; oof_macro_f1=0.755472
- 2026-05-11 13:48:27 powershell CMD v1.7; selected=fusion; oof_macro_f1=0.755537
- 2026-05-11 15:29:48 powershell CMD v1.7; selected=fusion; oof_macro_f1=0.755350
- 2026-05-13 13:19:20 powershell v1.12 LGB+XGB 100-model ensemble; oof_macro_f1=0.754355
- 2026-05-13 15:11:18 powershell v1.13 adaptive LGB+XGB 100-model ensemble; oof_macro_f1=0.751889; weighted_oof=0.750297
- 2026-05-15 20:45:00 powershell v3.7_align: exact_key5特征对齐（132→137维），根因修复teacher/student表示不匹配。class1 3308→3354(+46)，平台Macro-F1=0.70831(+0.0039)。自动候选选择+c1 guard。归档到版本记录/v3.7_align/，更新README和项目状态。
