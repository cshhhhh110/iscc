# 先读总规则：请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`，再开始本项目。

根目录的 `源码/` 只放当前活跃版本；历史代码、模型、结果和 `smoke_*` 统一进 `版本记录/<version>/`，文件名也要带版本号。

# ISCC 日志异常检测赛基线包

本目录实现“行级多标签 + 区间解码”的系统日志异常检测基线。训练数据为 `train.csv`，测试数据为 `test.csv`，输出格式与 `sample_submission.csv` 一致。

## 目录结构

- `源码/`：训练、预测和提交校验脚本。
- `模型/`：模型包与验证报告。
- `提交结果/`：提交 CSV。
- `docker容器/`：容器复现材料。
- `requirements.txt`：Python 依赖。
- `ACTION_LOG.md`：执行记录。

## 训练

推荐在赛题目录运行：

```powershell
conda run -n iscc-ml python 源码/train.py
```

默认执行 5 折 OOF 阈值搜索，使用 `SGDClassifier(log_loss)` 训练 1 个文档级异常头和 10 个行级异常类型头。训练和推理循环均带 `tqdm` 进度条。

训练输出：

- `模型/model_bundle.joblib`
- `模型/validation_report.json`
- `提交结果/submission.csv`

## 预测复现

```powershell
conda run -n iscc-ml python 源码/predict.py
```

默认读取 `模型/model_bundle.joblib`，输出 `提交结果/submission_reproduced.csv`。

## 提交校验

```powershell
conda run -n iscc-ml python 源码/validate_submission.py --submission-path 提交结果/submission.csv
```

校验列名、行数、id 顺序、异常字段规则、区间格式和区间边界。

## 打包建议

最终压缩包按竞赛要求包含：

- `源码/`
- `模型/`
- `提交结果/`
- `docker容器/`
- `requirements.txt`
- `README.md`
82%
