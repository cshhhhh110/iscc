# ACTION_LOG

- 2026-05-04 初始化 PowerShell 恶意脚本检测项目结构，确认任务为 15 个离散数值特征的三分类，指标为 Macro-F1。
- 2026-05-04 升级为自动选型 baseline：ExtraTrees、HistGradientBoosting 与二者融合按 OOF Macro-F1 选最优。
- 2026-05-04 23:53:03 trained PowerShell baseline; selected=blend; oof_macro_f1=0.753027; model=D:\桌面\赛题数据\powershell恶意脚本检测\模型\model_bundle.joblib
- 2026-05-04 23:53:24 generated PowerShell submission with 20000 rows: D:\桌面\赛题数据\powershell恶意脚本检测\提交结果\submission.csv
- 2026-05-05 00:58:50 trained PowerShell baseline; selected=blend; oof_macro_f1=0.753027; model=D:\桌面\赛题数据\powershell恶意脚本检测\模型\model_bundle.joblib
- 2026-05-05 13:32:52 trained PowerShell baseline; selected=blend; class_bias=[1.0, 0.925, 1.4]; oof_macro_f1=0.755080; model=D:\桌面\赛题数据\powershell恶意脚本检测\模型\model_bundle.joblib
- 2026-05-05 13:32:59 generated PowerShell submission with 20000 rows: D:\桌面\赛题数据\powershell恶意脚本检测\提交结果\submission.csv; class_bias=[1.0, 0.925, 1.4]
- 2026-05-05 13:33:00 validated PowerShell submission: rows=20000, label_counts={0: 12729, 1: 3347, 2: 3924}
- 2026-05-05 22:22:51 generated PowerShell submission with 20000 rows: _smoke_v1_1_submission.csv; class_bias=[1.0, 1.0, 1.0]
- 2026-05-05 22:23:12 validated PowerShell submission: rows=20000, label_counts={0: 15696, 1: 366, 2: 3938}
- 2026-05-05 22:32:56 v1.1 deep tabular rebuild source implemented; CUDA smoke train/predict/validate passed with mlp, 2 folds, 1 epoch; snapshot archived under 版本记录/v1.1/ with source, current model, current submission, and smoke artifacts.
- 2026-05-05 22:35:37 added legacy blend-bundle compatibility and silenced loky core warnings; smoke-tested the current official tree bundle with predict/validate.
- 2026-05-05 22:40:31 made total-log writes best-effort so training will not fail if the outer log path is unavailable.
- 2026-05-05 23:03:56 default artifact filenames updated to versioned names: model_bundle_v1.1.joblib, validation_report_v1.1.json, and submission_v1.1.csv.
- 2026-05-05 23:03:56 created versioned artifact copies in 模型/ and 提交结果/ and synced the v1.1 snapshot to the same versioned filenames.
