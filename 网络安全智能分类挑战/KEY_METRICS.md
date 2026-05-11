# KEY_METRICS — v1.5+ 迭代关键数据

| 时间 | 版本 | 类型 | 模型 | features | seeds | folds | local_acc | local_macro_f1 | 弱类F1 | 平台分 | 备注 |
|------|------|------|------|----------|-------|-------|-----------|---------------|--------|--------|------|
| 05-10 20:10 | v1.5 | smoke | ft+cb_avg | 47 | 1 | 2 | 0.5069 | 0.4510 | class_1:0.0000 | - | FT:0.2059 CB:0.4740 |
| 05-10 20:12 | v1.5 | predict | ft+cb_avg | 47 | - | 2 | - | - | - | - | submission=smoke_submission_gpu_v1.5.csv |
| 05-10 22:27 | v1.5 | full | ft+cb_avg | 47 | 3 | 5 | 0.9288 | 0.9247 | class_11:0.7735 | **0.69485** | FT:0.9178 CB:0.9065 |
| 05-10 22:28 | v1.5 | predict | ft+cb_avg | 47 | - | 15 | - | - | - | - | submission=submission_gpu_v1.5.csv |
| 05-10 23:53 | v1.6 | smoke | ft_simplified | 42 | 1 | 2 | 0.1167 | 0.0524 | class_1:0.0000 | - | dropped=8,qt,d=48,blk=3 |
| 05-10 23:53 | v1.6 | predict | ft_simplified | 42 | - | 2 | - | - | - | - | submission=smoke_submission_gpu_v1.6.csv |
| 05-11 10:16 | v1.6 | full | ft_simplified | 42 | 2 | 5 | 0.8612 | 0.8527 | class_10:0.6196 | **0.60658** | dropped=8,qt,d=48,blk=3 |
| 05-11 10:16 | v1.6 | predict | ft_simplified | 42 | - | 10 | - | - | - | - | submission=submission_gpu_v1.6.csv |
| 05-11 12:20 | v1.7 | smoke | xgb+lgbm_avg | 50 | 1 | 2 | 0.6139 | 0.5904 | class_5:0.3200 | - | XGB:0.6086 LGBM:0.5693 |
| 05-11 12:20 | v1.7 | predict | xgb+lgbm_avg | 50 | - | 2 | - | - | - | - | submission=smoke_submission_tree_v1.7.csv |
| 05-11 13:02 | v1.7 | full | xgb+lgbm_avg | 50 | 3 | 5 | 0.9336 | 0.9295 | class_10:0.7874 | **0.68547** | XGB:0.9292 LGBM:0.9294 |
| 05-11 13:03 | v1.7 | predict | xgb+lgbm_avg | 50 | - | 15 | - | - | - | - | submission=submission_tree_v1.7.csv |
| 05-11 13:32 | v1.8 | smoke | ft_swa_best_individual | 50 | 1 | 2 | 0.2208 | 0.1582 | class_1:0.0000 | - | best:0.1582 swa:0.1244->best_individual |
| 05-11 13:33 | v1.8 | predict | ft_swa_best_individual | 50 | - | 2 | - | - | - | - | submission=smoke_submission_gpu_v1.8.csv,best_individual |
| 05-11 14:43 | v1.8 | full | ft_swa_best_individual | 50 | 3 | 5 | 0.9247 | 0.9201 | class_11:0.7581 | **0.70563** | best:0.9201 swa:0.9197->best_individual |
| 05-11 14:44 | v1.8 | predict | ft_swa_best_individual | 50 | - | 15 | - | - | - | - | submission=submission_gpu_v1.8.csv,best_individual |
| 05-11 15:05 | — | blend | v1.1+v1.4 50:50 | 50 | — | — | — | — | — | **0.709** | 零训练，破天花板 |
| 05-11 15:05 | v1.9 | smoke | ft_4seed_ensemble | 50 | 1 | 2 | 0.2208 | 0.1582 | class_1:0.0000 | - | v1.4arch,2fold ensemble |
| 05-11 15:05 | v1.9 | predict | ft_4seed_ensemble | 50 | - | 2 | - | - | - | - | submission=smoke_submission_gpu_v1.9.csv |
