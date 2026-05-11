# ACTION_LOG

[2026-05-04 20:30:27] Created initial competition statement document.
[2026-05-04 20:35:00] Inspected workspace: train/test/sample CSV files exist in the competition directory.
[2026-05-04 20:36:00] Confirmed actual task contract: numeric tabular 12-class classification with submission columns id,label.
[2026-05-04 20:37:00] Planned complete baseline package with source, models, results, container material, requirements, README, and action log.
[2026-05-04 21:10:00] Upgraded baseline to a stacked ensemble with HGB, ExtraTrees, LogisticRegression, and LDA plus a logistic meta-model.
[2026-05-04 20:40:32] Training started.
[2026-05-04 20:40:33] Loaded data: train_rows=53477, test_rows=19440, features=50, labels=12.
[2026-05-04 20:45:05] Training started.
[2026-05-04 20:45:06] Loaded data: train_rows=53477, test_rows=19440, features=50, labels=12.
[2026-05-04 20:45:26] Fold 1/5 finished: macro_f1=0.922001.
[2026-05-04 20:45:44] Fold 2/5 finished: macro_f1=0.924237.
[2026-05-04 20:46:06] Fold 3/5 finished: macro_f1=0.919412.
[2026-05-04 20:46:25] Fold 4/5 finished: macro_f1=0.918712.
[2026-05-04 20:46:42] Fold 5/5 finished: macro_f1=0.919013.
[2026-05-04 20:47:43] Training completed: macro_f1=0.920678, accuracy=0.925145, submission=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\submission.csv, model=D:\桌面\赛题数据\网络安全智能分类挑战\模型\model_bundle.joblib.
[2026-05-04 23:59:27] Training started.
[2026-05-04 23:59:27] Loaded data: train_rows=53477, test_rows=19440, features=50, labels=12.
[2026-05-05 00:08:46] Training completed: meta_macro_f1=0.931006, accuracy=0.935187, submission=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\submission.csv, model=D:\桌面\赛题数据\网络安全智能分类挑战\模型\model_bundle.joblib.
[2026-05-05 13:22:05] GPU training started: device=cuda, train_rows=53477, test_rows=19440, features=50, labels=12, seeds=[20260504, 20260505], folds=5.
[2026-05-05 13:23:27] GPU seed=20260504 fold=1/5 finished: best_macro_f1=0.896933, best_epoch=91.
[2026-05-05 13:24:39] GPU seed=20260504 fold=2/5 finished: best_macro_f1=0.898584, best_epoch=86.
[2026-05-05 13:25:51] GPU seed=20260504 fold=3/5 finished: best_macro_f1=0.899901, best_epoch=90.
[2026-05-05 13:27:50] GPU training started: device=cuda, train_rows=53477, test_rows=19440, features=50, labels=12, seeds=[20260504, 20260505], folds=5.
[2026-05-05 13:29:18] GPU seed=20260504 fold=1/5 finished: best_macro_f1=0.896933, best_epoch=91.
[2026-05-05 13:30:48] GPU seed=20260504 fold=2/5 finished: best_macro_f1=0.898584, best_epoch=86.
[2026-05-05 13:32:39] GPU seed=20260504 fold=3/5 finished: best_macro_f1=0.899901, best_epoch=90.
[2026-05-05 13:34:41] GPU seed=20260504 fold=4/5 finished: best_macro_f1=0.895293, best_epoch=102.
[2026-05-05 13:36:15] GPU seed=20260504 fold=5/5 finished: best_macro_f1=0.895245, best_epoch=95.
[2026-05-05 13:37:47] GPU seed=20260505 fold=1/5 finished: best_macro_f1=0.893151, best_epoch=85.
[2026-05-05 13:39:14] GPU seed=20260505 fold=2/5 finished: best_macro_f1=0.896719, best_epoch=86.
[2026-05-05 13:40:44] GPU seed=20260505 fold=3/5 finished: best_macro_f1=0.896871, best_epoch=109.
[2026-05-05 13:42:05] GPU seed=20260505 fold=4/5 finished: best_macro_f1=0.897288, best_epoch=118.
[2026-05-05 13:43:20] GPU seed=20260505 fold=5/5 finished: best_macro_f1=0.894440, best_epoch=89.
[2026-05-05 13:43:20] GPU training completed: macro_f1=0.900271, accuracy=0.904389, submission=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu.csv.
[2026-05-06 08:42:18] v1.1 FT-Transformer training started: smoke=True, device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2.
[2026-05-06 08:42:22] v1.1 FT-Transformer seed=20260504 fold=1/2 finished: best_macro_f1=0.142876, best_epoch=2.
[2026-05-06 08:42:22] v1.1 FT-Transformer seed=20260504 fold=2/2 finished: best_macro_f1=0.140592, best_epoch=2.
[2026-05-06 08:44:32] v1.1 FT-Transformer training started: smoke=True, device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2.
[2026-05-06 08:44:34] v1.1 FT-Transformer seed=20260504 fold=1/2 finished: best_macro_f1=0.142876, best_epoch=2.
[2026-05-06 08:44:35] v1.1 FT-Transformer seed=20260504 fold=2/2 finished: best_macro_f1=0.140592, best_epoch=2.
[2026-05-06 08:46:06] v1.1 FT-Transformer training started: smoke=True, device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2.
[2026-05-06 08:46:09] v1.1 FT-Transformer seed=20260504 fold=1/2 finished: best_macro_f1=0.142876, best_epoch=2.
[2026-05-06 08:46:09] v1.1 FT-Transformer seed=20260504 fold=2/2 finished: best_macro_f1=0.140592, best_epoch=2.
[2026-05-06 08:47:12] v1.1 FT-Transformer training started: smoke=True, device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2.
[2026-05-06 08:47:15] v1.1 FT-Transformer seed=20260504 fold=1/2 finished: best_macro_f1=0.142876, best_epoch=2.
[2026-05-06 08:47:15] v1.1 FT-Transformer seed=20260504 fold=2/2 finished: best_macro_f1=0.140592, best_epoch=2.
[2026-05-06 08:47:15] v1.1 FT-Transformer training completed: smoke=True, macro_f1=0.158175, accuracy=0.220833, submission=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_gpu_fttransformer_v1.1.csv.
[2026-05-06 08:55:14] v1.1 FT-Transformer training started: smoke=False, device=cuda, train_rows=53477/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=5.
[2026-05-06 08:59:13] v1.1 FT-Transformer seed=20260504 fold=1/5 finished: best_macro_f1=0.905694, best_epoch=47.
[2026-05-06 09:03:09] v1.1 FT-Transformer seed=20260504 fold=2/5 finished: best_macro_f1=0.907699, best_epoch=44.
[2026-05-06 09:07:11] v1.1 FT-Transformer seed=20260504 fold=3/5 finished: best_macro_f1=0.906365, best_epoch=44.
[2026-05-06 09:11:09] v1.1 FT-Transformer seed=20260504 fold=4/5 finished: best_macro_f1=0.906890, best_epoch=43.
[2026-05-06 09:15:40] v1.1 FT-Transformer seed=20260504 fold=5/5 finished: best_macro_f1=0.904051, best_epoch=51.
[2026-05-06 09:15:40] v1.1 FT-Transformer training completed: smoke=False, macro_f1=0.906203, accuracy=0.911027, submission=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu_fttransformer_v1.1.csv.
[2026-05-06 10:04:00] v1.2 training started: device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2, loss_type=focal.
[2026-05-06 10:04:04] v1.2 ft_transformer seed=20260504 fold=1/2 finished: best_macro_f1=0.184542, best_epoch=2.
[2026-05-06 10:04:04] v1.2 ft_transformer seed=20260504 fold=2/2 finished: best_macro_f1=0.186159, best_epoch=1.
[2026-05-06 10:04:05] v1.2 residual_mlp seed=20260504 fold=1/2 finished: best_macro_f1=0.456837, best_epoch=2.
[2026-05-06 10:04:05] v1.2 residual_mlp seed=20260504 fold=2/2 finished: best_macro_f1=0.413679, best_epoch=2.
[2026-05-06 10:04:05] v1.2 training completed: selected=ensemble; macro_f1=0.438022; accuracy=0.459722; bundle=D:\桌面\赛题数据\网络安全智能分类挑战\模型\gpu_model_bundle_v1.2.pt.
[2026-05-06 10:04:32] v1.2 blend completed: source=ensemble, output=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_blend_v1.2.csv.
[2026-05-06 10:04:33] v1.2 predict completed: source=ensemble, output=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_gpu_v1.2.csv.
[2026-05-06 10:04:33] v1.2 pure GPU dual-model smoke completed; selected=ensemble; smoke macro_f1=0.438022, accuracy=0.459722; outputs: 模型/gpu_model_bundle_v1.2.pt, 模型/gpu_oof_probs_v1.2.npy, 模型/gpu_test_probs_v1.2.npy, 提交结果/submission_gpu_v1.2.csv, 提交结果/submission_blend_v1.2.csv.
[2026-05-06 10:07:50] v1.2 training started: device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2, loss_type=focal.
[2026-05-06 10:07:53] v1.2 ft_transformer seed=20260504 fold=1/2 finished: best_macro_f1=0.184542, best_epoch=2.
[2026-05-06 10:07:54] v1.2 ft_transformer seed=20260504 fold=2/2 finished: best_macro_f1=0.186159, best_epoch=1.
[2026-05-06 10:07:54] v1.2 residual_mlp seed=20260504 fold=1/2 finished: best_macro_f1=0.456837, best_epoch=2.
[2026-05-06 10:07:55] v1.2 residual_mlp seed=20260504 fold=2/2 finished: best_macro_f1=0.413679, best_epoch=2.
[2026-05-06 10:07:55] v1.2 training completed: selected=ensemble; macro_f1=0.438022; accuracy=0.459722; bundle=D:\桌面\赛题数据\网络安全智能分类挑战\模型\smoke_gpu_model_bundle_v1.2.pt.
[2026-05-06 10:08:16] v1.2 blend completed: source=ensemble, output=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_blend_v1.2.csv.
[2026-05-06 10:08:18] v1.2 predict completed: source=ensemble, output=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_gpu_v1.2.csv.
[2026-05-06 12:02:49] v1.2 training started: device=cuda, train_rows=53477/53477, test_rows=19440, features=50, labels=12, seeds=[20260504, 20260505], folds=5, loss_type=focal.
[2026-05-06 12:08:25] v1.2 ft_transformer seed=20260504 fold=1/5 finished: best_macro_f1=0.910755, best_epoch=81.
[2026-05-06 12:13:46] v1.2 ft_transformer seed=20260504 fold=2/5 finished: best_macro_f1=0.909233, best_epoch=64.
[2026-05-06 12:19:17] v1.2 ft_transformer seed=20260504 fold=3/5 finished: best_macro_f1=0.909000, best_epoch=61.
[2026-05-06 12:25:00] v1.2 ft_transformer seed=20260504 fold=4/5 finished: best_macro_f1=0.906785, best_epoch=63.
[2026-05-06 12:31:29] v1.2 ft_transformer seed=20260504 fold=5/5 finished: best_macro_f1=0.911883, best_epoch=75.
[2026-05-06 12:36:27] v1.2 ft_transformer seed=20260505 fold=1/5 finished: best_macro_f1=0.906626, best_epoch=55.
[2026-05-06 12:42:23] v1.2 ft_transformer seed=20260505 fold=2/5 finished: best_macro_f1=0.909862, best_epoch=69.
[2026-05-06 12:48:35] v1.2 ft_transformer seed=20260505 fold=3/5 finished: best_macro_f1=0.910519, best_epoch=80.
[2026-05-06 12:54:07] v1.2 ft_transformer seed=20260505 fold=4/5 finished: best_macro_f1=0.908963, best_epoch=69.
[2026-05-06 12:58:52] v1.2 ft_transformer seed=20260505 fold=5/5 finished: best_macro_f1=0.905520, best_epoch=58.
[2026-05-06 13:00:27] v1.2 residual_mlp seed=20260504 fold=1/5 finished: best_macro_f1=0.883440, best_epoch=72.
[2026-05-06 13:01:26] v1.2 residual_mlp seed=20260504 fold=2/5 finished: best_macro_f1=0.877453, best_epoch=39.
[2026-05-06 13:03:03] v1.2 residual_mlp seed=20260504 fold=3/5 finished: best_macro_f1=0.881498, best_epoch=82.
[2026-05-06 13:03:59] v1.2 residual_mlp seed=20260504 fold=4/5 finished: best_macro_f1=0.881243, best_epoch=44.
[2026-05-06 13:04:35] v1.2 residual_mlp seed=20260504 fold=5/5 finished: best_macro_f1=0.874411, best_epoch=23.
[2026-05-06 13:06:05] v1.2 residual_mlp seed=20260505 fold=1/5 finished: best_macro_f1=0.885172, best_epoch=63.
[2026-05-06 13:07:53] v1.2 residual_mlp seed=20260505 fold=2/5 finished: best_macro_f1=0.888889, best_epoch=85.
[2026-05-06 13:09:54] v1.2 residual_mlp seed=20260505 fold=3/5 finished: best_macro_f1=0.883490, best_epoch=90.
[2026-05-06 13:11:41] v1.2 residual_mlp seed=20260505 fold=4/5 finished: best_macro_f1=0.886278, best_epoch=89.
[2026-05-06 13:12:43] v1.2 residual_mlp seed=20260505 fold=5/5 finished: best_macro_f1=0.876101, best_epoch=33.
[2026-05-06 13:12:45] v1.2 training completed: selected=ensemble; macro_f1=0.920251; accuracy=0.924360; bundle=D:\桌面\赛题数据\网络安全智能分类挑战\模型\gpu_model_bundle_v1.2.pt.
[2026-05-06 15:51:48] v1.2 predict completed: source=ensemble, output=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu_v1.2.csv.
[2026-05-06 20:28:21] v1.3 training started: smoke=True, device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2, loss_type=ce, selection_metric=accuracy.
[2026-05-06 20:28:25] v1.3 compact_resmlp seed=20260504 fold=1/2 finished: best_accuracy=0.297222, best_macro_f1=0.235529, best_epoch=2.
[2026-05-06 20:28:25] v1.3 compact_resmlp seed=20260504 fold=2/2 finished: best_accuracy=0.294444, best_macro_f1=0.239571, best_epoch=2.
[2026-05-06 20:29:23] v1.3 training started: smoke=True, device=cuda, train_rows=720/53477, test_rows=19440, features=50, labels=12, seeds=[20260504], folds=2, loss_type=ce, selection_metric=accuracy.
[2026-05-06 20:29:26] v1.3 compact_resmlp seed=20260504 fold=1/2 finished: best_accuracy=0.297222, best_macro_f1=0.235529, best_epoch=2.
[2026-05-06 20:29:26] v1.3 compact_resmlp seed=20260504 fold=2/2 finished: best_accuracy=0.294444, best_macro_f1=0.239571, best_epoch=2.
[2026-05-06 20:34:39] v1.3 source snapshot copied to 版本记录/v1.3/源码; root source cleaned to the active v1.3 scripts only.
[2026-05-06 20:34:39] v1.3 smoke could not finish artifact saving inside the Codex sandbox because Python writes to the workspace are blocked; user-run smoke/full training is still required to generate the bundle and submission files.
[2026-05-06 20:48:14] v1.3 training started: smoke=False, device=cuda, train_rows=53477/53477, test_rows=19440, features=50, labels=12, seeds=[20260504, 20260505], folds=5, loss_type=ce, selection_metric=accuracy.
[2026-05-06 20:49:27] v1.3 compact_resmlp seed=20260504 fold=1/5 finished: best_accuracy=0.908190, best_macro_f1=0.903913, best_epoch=88.
[2026-05-06 20:50:14] v1.3 compact_resmlp seed=20260504 fold=2/5 finished: best_accuracy=0.902767, best_macro_f1=0.899088, best_epoch=62.
[2026-05-06 20:50:46] v1.3 compact_resmlp seed=20260504 fold=3/5 finished: best_accuracy=0.907059, best_macro_f1=0.903089, best_epoch=36.
[2026-05-06 20:51:32] v1.3 compact_resmlp seed=20260504 fold=4/5 finished: best_accuracy=0.902010, best_macro_f1=0.897244, best_epoch=59.
[2026-05-06 20:52:13] v1.3 compact_resmlp seed=20260504 fold=5/5 finished: best_accuracy=0.903319, best_macro_f1=0.898780, best_epoch=48.
[2026-05-06 20:52:45] v1.3 compact_resmlp seed=20260505 fold=1/5 finished: best_accuracy=0.901739, best_macro_f1=0.898135, best_epoch=37.
[2026-05-06 20:53:22] v1.3 compact_resmlp seed=20260505 fold=2/5 finished: best_accuracy=0.905479, best_macro_f1=0.901441, best_epoch=44.
[2026-05-06 20:54:36] v1.3 compact_resmlp seed=20260505 fold=3/5 finished: best_accuracy=0.906498, best_macro_f1=0.902899, best_epoch=98.
[2026-05-06 20:55:12] v1.3 compact_resmlp seed=20260505 fold=4/5 finished: best_accuracy=0.904815, best_macro_f1=0.900616, best_epoch=34.
[2026-05-06 20:55:53] v1.3 compact_resmlp seed=20260505 fold=5/5 finished: best_accuracy=0.901917, best_macro_f1=0.898426, best_epoch=52.
[2026-05-06 20:55:55] v1.3 training completed: selected=bias_accuracy; accuracy=0.912355; macro_f1=0.908378; bundle=D:\桌面\赛题数据\网络安全智能分类挑战\模型\gpu_model_bundle_v1.3.pt.
[2026-05-06 21:33:05] v1.3 predict completed: selected=bias_accuracy, output=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu_v1.3.csv, candidates=5.
[2026-05-06 22:15:00] v1.4 FT-Transformer source prepared: active root source switched to v1.4; archived source snapshot copied to 版本记录/v1.4/源码.
[2026-05-06 22:15:00] v1.4 project structure updated: root source now keeps only the active v1.4 scripts plus common/validate helpers; version record directories created for source/model/result snapshots.
[2026-05-06 23:17:08] v1.4 FT-Transformer training started: smoke=False, device=cuda, train_rows=53477/53477, test_rows=19440, features=50, labels=12, seeds=[20260504, 20260505], folds=5, selection_metric=accuracy.
[2026-05-06 23:21:20] v1.4 FT-Transformer seed=20260504 fold=1/5 finished: best_accuracy=0.910901, best_macro_f1=0.905694, best_epoch=47.
[2026-05-06 23:25:18] v1.4 FT-Transformer seed=20260504 fold=2/5 finished: best_accuracy=0.912117, best_macro_f1=0.907699, best_epoch=44.
[2026-05-06 23:29:18] v1.4 FT-Transformer seed=20260504 fold=3/5 finished: best_accuracy=0.910986, best_macro_f1=0.906365, best_epoch=44.
[2026-05-06 23:32:57] v1.4 FT-Transformer seed=20260504 fold=4/5 finished: best_accuracy=0.911454, best_macro_f1=0.906890, best_epoch=43.
[2026-05-06 23:36:59] v1.4 FT-Transformer seed=20260504 fold=5/5 finished: best_accuracy=0.909677, best_macro_f1=0.904051, best_epoch=51.
[2026-05-06 23:40:51] v1.4 FT-Transformer seed=20260505 fold=1/5 finished: best_accuracy=0.909873, best_macro_f1=0.905778, best_epoch=48.
[2026-05-06 23:44:34] v1.4 FT-Transformer seed=20260505 fold=2/5 finished: best_accuracy=0.909873, best_macro_f1=0.904110, best_epoch=45.
[2026-05-06 23:47:30] v1.4 FT-Transformer seed=20260505 fold=3/5 finished: best_accuracy=0.911454, best_macro_f1=0.906961, best_epoch=33.
[2026-05-06 23:50:55] v1.4 FT-Transformer seed=20260505 fold=4/5 finished: best_accuracy=0.910332, best_macro_f1=0.905449, best_epoch=41.
[2026-05-06 23:54:07] v1.4 FT-Transformer seed=20260505 fold=5/5 finished: best_accuracy=0.906031, best_macro_f1=0.900930, best_epoch=34.
[2026-05-06 23:54:08] v1.4 FT-Transformer training completed: smoke=False, macro_f1=0.915914, accuracy=0.920564, bundle=D:\桌面\赛题数据\网络安全智能分类挑战\模型\gpu_fttransformer_model_bundle_v1.4.pt.
[2026-05-06 23:56:18] v1.4 predict completed: output=D:\桌面\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu_fttransformer_v1.4.csv, bundle=D:\桌面\赛题数据\网络安全智能分类挑战\模型\gpu_fttransformer_model_bundle_v1.4.pt, folds=10.
[2026-05-10 20:10:44] v1.5 training started: device=cuda, train_rows=720/53477, test_rows=19440, features=47 (dropped=['pattern_change_ratio', 'pattern_mix_density', 'pattern_switch_frequency']), labels=12, seeds=[20260504], folds=2, ft_dropout=(0.12,0.08,0.12,0.2), skip_catboost=False.
[2026-05-10 20:10:47] v1.5 FT seed=20260504 fold=1/2: acc=0.252778 mf1=0.181281 ep=2
[2026-05-10 20:10:49] v1.5 FT seed=20260504 fold=2/2: acc=0.236111 mf1=0.188023 ep=2
[2026-05-10 20:10:51] v1.5 CB fold=1/2: acc=0.541667 mf1=0.491846
[2026-05-10 20:10:53] v1.5 CB fold=2/2: acc=0.511111 mf1=0.452333
[2026-05-10 20:10:53] v1.5 done: ensemble=ft+cb_avg | FT acc=0.2444 mf1=0.2059 | CB acc=0.5264 mf1=0.4740 | ENS acc=0.5069 mf1=0.4510 weak=(class_1:0.0000)
[2026-05-10 20:12:21] v1.5 predict done: ensemble=ft+cb_avg, output=E:\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_gpu_v1.5.csv, ft_folds=2.
[2026-05-10 20:13:08] v1.5 training started: device=cuda, train_rows=53477/53477, test_rows=19440, features=47 (dropped=['pattern_change_ratio', 'pattern_mix_density', 'pattern_switch_frequency']), labels=12, seeds=[20260504, 20260505, 20260506], folds=5, ft_dropout=(0.12,0.08,0.12,0.2), skip_catboost=False.
[2026-05-10 20:21:22] v1.5 FT seed=20260504 fold=1/5: acc=0.908283 mf1=0.903254 ep=52
[2026-05-10 20:29:37] v1.5 FT seed=20260504 fold=2/5: acc=0.906694 mf1=0.902033 ep=53
[2026-05-10 20:37:38] v1.5 FT seed=20260504 fold=3/5: acc=0.913791 mf1=0.908842 ep=52
[2026-05-10 20:45:21] v1.5 FT seed=20260504 fold=4/5: acc=0.909584 mf1=0.904388 ep=46
[2026-05-10 20:53:26] v1.5 FT seed=20260504 fold=5/5: acc=0.910519 mf1=0.905381 ep=49
[2026-05-10 21:02:47] v1.5 FT seed=20260505 fold=1/5: acc=0.909312 mf1=0.904382 ep=62
[2026-05-10 21:07:43] v1.5 FT seed=20260505 fold=2/5: acc=0.906227 mf1=0.901006 ep=44
[2026-05-10 21:11:06] v1.5 FT seed=20260505 fold=3/5: acc=0.910799 mf1=0.905722 ep=36
[2026-05-10 21:20:55] v1.5 FT seed=20260505 fold=4/5: acc=0.911641 mf1=0.906873 ep=66
[2026-05-10 21:30:42] v1.5 FT seed=20260505 fold=5/5: acc=0.906498 mf1=0.902091 ep=45
[2026-05-10 21:39:50] v1.5 FT seed=20260506 fold=1/5: acc=0.911182 mf1=0.906374 ep=53
[2026-05-10 21:46:17] v1.5 FT seed=20260506 fold=2/5: acc=0.909312 mf1=0.904711 ep=40
[2026-05-10 21:55:45] v1.5 FT seed=20260506 fold=3/5: acc=0.907714 mf1=0.902258 ep=61
[2026-05-10 22:05:33] v1.5 FT seed=20260506 fold=4/5: acc=0.907714 mf1=0.901917 ep=63
[2026-05-10 22:13:18] v1.5 FT seed=20260506 fold=5/5: acc=0.907994 mf1=0.903588 ep=47
[2026-05-10 22:16:07] v1.5 CB fold=1/5: acc=0.909873 mf1=0.905805
[2026-05-10 22:19:03] v1.5 CB fold=2/5: acc=0.911088 mf1=0.907626
[2026-05-10 22:21:51] v1.5 CB fold=3/5: acc=0.913417 mf1=0.909481
[2026-05-10 22:24:38] v1.5 CB fold=4/5: acc=0.909771 mf1=0.905819
[2026-05-10 22:27:23] v1.5 CB fold=5/5: acc=0.907994 mf1=0.903808
[2026-05-10 22:27:24] v1.5 done: ensemble=ft+cb_avg | FT acc=0.9225 mf1=0.9178 | CB acc=0.9104 mf1=0.9065 | ENS acc=0.9288 mf1=0.9247 weak=(class_11:0.7735)
[2026-05-10 22:28:19] v1.5 predict done: ensemble=ft+cb_avg, output=E:\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu_v1.5.csv, ft_folds=15.
[2026-05-10 23:53:22] v1.6 training: device=cuda, rows=720/53477, test=19440, features=42 (dropped=['behavior_template_activity', 'behavior_template_control', 'behavior_template_volume', 'pattern_change_ratio', 'pattern_diversity_ratio', 'pattern_mix_density', 'pattern_switch_frequency', 'protocol_variation_level']), labels=12, seeds=[20260504], folds=2, arch=(d=48,blk=3,h=6,ffn=160), drop=(0.15,0.1,0.15,0.25), ls=0.08, wd=0.0005, preproc=quantile.
[2026-05-10 23:53:26] v1.6 FT seed=20260504 fold=1/2: mf1=0.013995 acc=0.091667 ep=1
[2026-05-10 23:53:26] v1.6 FT seed=20260504 fold=2/2: mf1=0.049740 acc=0.141667 ep=2
[2026-05-10 23:53:26] v1.6 done: acc=0.1167 mf1=0.0524 weak=(class_1:0.0000) folds=2
[2026-05-10 23:53:31] v1.6 predict done: output=E:\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_gpu_v1.6.csv, folds=2.
[2026-05-11 00:23:26] v1.6 training: device=cuda, rows=53477/53477, test=19440, features=42 (dropped=['behavior_template_activity', 'behavior_template_control', 'behavior_template_volume', 'pattern_change_ratio', 'pattern_diversity_ratio', 'pattern_mix_density', 'pattern_switch_frequency', 'protocol_variation_level']), labels=12, seeds=[20260504, 20260505], folds=5, arch=(d=48,blk=3,h=6,ffn=160), drop=(0.15,0.1,0.15,0.25), ls=0.08, wd=0.0005, preproc=quantile.
[2026-05-11 00:32:49] v1.6 training: device=cuda, rows=53477/53477, test=19440, features=42 (dropped=['behavior_template_activity', 'behavior_template_control', 'behavior_template_volume', 'pattern_change_ratio', 'pattern_diversity_ratio', 'pattern_mix_density', 'pattern_switch_frequency', 'protocol_variation_level']), labels=12, seeds=[20260504, 20260505], folds=5, arch=(d=48,blk=3,h=6,ffn=160), drop=(0.15,0.1,0.15,0.25), ls=0.08, wd=0.0005, preproc=quantile.
[2026-05-11 09:57:35] v1.6 training: device=cuda, rows=53477/53477, test=19440, features=42 (dropped=['behavior_template_activity', 'behavior_template_control', 'behavior_template_volume', 'pattern_change_ratio', 'pattern_diversity_ratio', 'pattern_mix_density', 'pattern_switch_frequency', 'protocol_variation_level']), labels=12, seeds=[20260504, 20260505], folds=5, arch=(d=48,blk=3,h=6,ffn=160), drop=(0.15,0.1,0.15,0.25), ls=0.08, wd=0.0005, preproc=quantile.
[2026-05-11 09:59:23] v1.6 FT seed=20260504 fold=1/5: mf1=0.832196 acc=0.843119 ep=55
[2026-05-11 10:01:17] v1.6 FT seed=20260504 fold=2/5: mf1=0.835103 acc=0.843493 ep=50
[2026-05-11 10:03:07] v1.6 FT seed=20260504 fold=3/5: mf1=0.847500 acc=0.856381 ep=55
[2026-05-11 10:04:57] v1.6 FT seed=20260504 fold=4/5: mf1=0.841426 acc=0.851052 ep=56
[2026-05-11 10:06:47] v1.6 FT seed=20260504 fold=5/5: mf1=0.840760 acc=0.850771 ep=56
[2026-05-11 10:08:38] v1.6 FT seed=20260505 fold=1/5: mf1=0.835567 acc=0.844428 ep=53
[2026-05-11 10:10:29] v1.6 FT seed=20260505 fold=2/5: mf1=0.831204 acc=0.841062 ep=54
[2026-05-11 10:12:21] v1.6 FT seed=20260505 fold=3/5: mf1=0.841583 acc=0.848714 ep=53
[2026-05-11 10:14:13] v1.6 FT seed=20260505 fold=4/5: mf1=0.841920 acc=0.850678 ep=60
[2026-05-11 10:16:05] v1.6 FT seed=20260505 fold=5/5: mf1=0.836231 acc=0.844133 ep=53
[2026-05-11 10:16:05] v1.6 done: acc=0.8612 mf1=0.8527 weak=(class_10:0.6196) folds=10
[2026-05-11 10:16:14] v1.6 predict done: output=E:\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu_v1.6.csv, folds=10.
[2026-05-11 12:20:15] v1.7 training: rows=720/53477, test=19440, features=50, labels=12, seeds=[20260504], folds=2, xgb=(lr=0.03,d=6,n=50), lgbm=(lr=0.03,d=6,n=50)
[2026-05-11 12:20:17] v1.7 XGB seed=20260504 fold=1/2: mf1=0.6240 | LGBM fold=1: mf1=0.5885
[2026-05-11 12:20:17] v1.7 XGB seed=20260504 fold=2/2: mf1=0.5925 | LGBM fold=2: mf1=0.5508
[2026-05-11 12:20:17] v1.7 done (0.0min): XGB mf1=0.6086 | LGBM mf1=0.5693 | ENS mf1=0.5904 acc=0.6139 weak=(class_5:0.3200)
[2026-05-11 12:20:20] v1.7 predict done: output=E:\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_tree_v1.7.csv, xgb_models=2, lgbm_models=2.
[2026-05-11 12:28:21] v1.7 training: rows=53477/53477, test=19440, features=50, labels=12, seeds=[20260504, 20260505, 20260506], folds=5, xgb=(lr=0.03,d=6,n=3000), lgbm=(lr=0.03,d=6,n=3000)
[2026-05-11 12:30:19] v1.7 XGB seed=20260504 fold=1/5: mf1=0.9301 | LGBM fold=1: mf1=0.9293
[2026-05-11 12:32:23] v1.7 XGB seed=20260504 fold=2/5: mf1=0.9291 | LGBM fold=2: mf1=0.9289
[2026-05-11 12:34:27] v1.7 XGB seed=20260504 fold=3/5: mf1=0.9271 | LGBM fold=3: mf1=0.9262
[2026-05-11 12:36:37] v1.7 XGB seed=20260504 fold=4/5: mf1=0.9259 | LGBM fold=4: mf1=0.9263
[2026-05-11 12:38:55] v1.7 XGB seed=20260504 fold=5/5: mf1=0.9243 | LGBM fold=5: mf1=0.9238
[2026-05-11 12:41:06] v1.7 XGB seed=20260505 fold=1/5: mf1=0.9287 | LGBM fold=1: mf1=0.9281
[2026-05-11 12:43:25] v1.7 XGB seed=20260505 fold=2/5: mf1=0.9305 | LGBM fold=2: mf1=0.9297
[2026-05-11 12:45:52] v1.7 XGB seed=20260505 fold=3/5: mf1=0.9282 | LGBM fold=3: mf1=0.9294
[2026-05-11 12:48:43] v1.7 XGB seed=20260505 fold=4/5: mf1=0.9289 | LGBM fold=4: mf1=0.9294
[2026-05-11 12:51:39] v1.7 XGB seed=20260505 fold=5/5: mf1=0.9277 | LGBM fold=5: mf1=0.9274
[2026-05-11 12:53:42] v1.7 XGB seed=20260506 fold=1/5: mf1=0.9297 | LGBM fold=1: mf1=0.9293
[2026-05-11 12:55:45] v1.7 XGB seed=20260506 fold=2/5: mf1=0.9321 | LGBM fold=2: mf1=0.9317
[2026-05-11 12:57:59] v1.7 XGB seed=20260506 fold=3/5: mf1=0.9287 | LGBM fold=3: mf1=0.9296
[2026-05-11 13:00:07] v1.7 XGB seed=20260506 fold=4/5: mf1=0.9245 | LGBM fold=4: mf1=0.9232
[2026-05-11 13:02:14] v1.7 XGB seed=20260506 fold=5/5: mf1=0.9309 | LGBM fold=5: mf1=0.9332
[2026-05-11 13:02:29] v1.7 done (33.9min): XGB mf1=0.9292 | LGBM mf1=0.9294 | ENS mf1=0.9295 acc=0.9336 weak=(class_10:0.7874)
[2026-05-11 13:03:14] v1.7 predict done: output=E:\赛题数据\网络安全智能分类挑战\提交结果\submission_tree_v1.7.csv, xgb_models=15, lgbm_models=15.
[2026-05-11 13:32:52] v1.8 training: device=cuda, rows=720/53477, test=19440, features=50, labels=12, seeds=[20260504], folds=2, epochs=2, arch=(d=64,blk=4,h=8,ffn=192), swa_start=1(0.6), drop=(0.1,0.05,0.1,0.15), ls=0.03, mf1_early_stop.
[2026-05-11 13:32:58] v1.8 FT seed=20260504 fold=1/2: best_mf1=0.142876 ep=2 swa_updates=2
[2026-05-11 13:32:59] v1.8 FT seed=20260504 fold=2/2: best_mf1=0.140592 ep=2 swa_updates=2
[2026-05-11 13:32:59] v1.8 done (0.1min): best mf1=0.1582 acc=0.2208 | swa mf1=0.1244 acc=0.1806 | selected=best_individual mf1=0.1582 weak=(class_1:0.0000)
[2026-05-11 13:33:05] v1.8 predict done: selected=best_individual, output=E:\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_gpu_v1.8.csv, folds=2.
[2026-05-11 13:52:51] v1.8 training: device=cuda, rows=53477/53477, test=19440, features=50, labels=12, seeds=[20260504, 20260505, 20260506], folds=5, epochs=90, arch=(d=64,blk=4,h=8,ffn=192), swa_start=54(0.6), drop=(0.1,0.05,0.1,0.15), ls=0.03, mf1_early_stop.
[2026-05-11 13:55:40] v1.8 FT seed=20260504 fold=1/5: best_mf1=0.905694 ep=47 swa_updates=8
[2026-05-11 13:58:40] v1.8 FT seed=20260504 fold=2/5: best_mf1=0.907699 ep=44 swa_updates=5
[2026-05-11 14:01:46] v1.8 FT seed=20260504 fold=3/5: best_mf1=0.906365 ep=44 swa_updates=5
[2026-05-11 14:05:07] v1.8 FT seed=20260504 fold=4/5: best_mf1=0.906890 ep=43 swa_updates=4
[2026-05-11 14:09:10] v1.8 FT seed=20260504 fold=5/5: best_mf1=0.904051 ep=51 swa_updates=12
[2026-05-11 14:12:57] v1.8 FT seed=20260505 fold=1/5: best_mf1=0.905778 ep=48 swa_updates=9
[2026-05-11 14:16:34] v1.8 FT seed=20260505 fold=2/5: best_mf1=0.904110 ep=45 swa_updates=6
[2026-05-11 14:20:00] v1.8 FT seed=20260505 fold=3/5: best_mf1=0.906961 ep=33 swa_updates=0
[2026-05-11 14:23:08] v1.8 FT seed=20260505 fold=4/5: best_mf1=0.905449 ep=41 swa_updates=2
[2026-05-11 14:26:15] v1.8 FT seed=20260505 fold=5/5: best_mf1=0.900930 ep=34 swa_updates=0
[2026-05-11 14:29:16] v1.8 FT seed=20260506 fold=1/5: best_mf1=0.905459 ep=32 swa_updates=0
[2026-05-11 14:33:23] v1.8 FT seed=20260506 fold=2/5: best_mf1=0.909145 ep=59 swa_updates=20
[2026-05-11 14:36:32] v1.8 FT seed=20260506 fold=3/5: best_mf1=0.905726 ep=43 swa_updates=4
[2026-05-11 14:40:23] v1.8 FT seed=20260506 fold=4/5: best_mf1=0.902734 ep=50 swa_updates=11
[2026-05-11 14:43:57] v1.8 FT seed=20260506 fold=5/5: best_mf1=0.904311 ep=38 swa_updates=0
[2026-05-11 14:43:58] v1.8 done (51.1min): best mf1=0.9201 acc=0.9247 | swa mf1=0.9197 acc=0.9245 | selected=best_individual mf1=0.9201 weak=(class_11:0.7581)
[2026-05-11 14:44:13] v1.8 predict done: selected=best_individual, output=E:\赛题数据\网络安全智能分类挑战\提交结果\submission_gpu_v1.8.csv, folds=15.
[2026-05-11 15:05:25] v1.9 training: device=cuda, rows=720/53477, test=19440, features=50, labels=12, seeds=[20260504], folds=2, mf1_early_stop, arch=v1.4
[2026-05-11 15:05:29] v1.9 FT seed=20260504 fold=1: mf1=0.142876 acc=0.219444 ep=2
[2026-05-11 15:05:30] v1.9 FT seed=20260504 fold=2: mf1=0.140592 acc=0.222222 ep=2
[2026-05-11 15:05:30] v1.9 done (0.1min): mf1=0.1582 acc=0.2208 weak=(class_1:0.0000) folds=2
[2026-05-11 15:05:35] v1.9 predict done: output=E:\赛题数据\网络安全智能分类挑战\提交结果\smoke_submission_gpu_v1.9.csv, folds=2.
