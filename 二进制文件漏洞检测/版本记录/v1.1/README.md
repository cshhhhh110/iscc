# 先读总规则：请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`，再开始本项目。

# ISCC Binary Vulnerability Detection

Official package layout for the ISCC data security binary vulnerability task.

## Layout

- `源码/`: training, inference, feature extraction, and helper code
- `模型/`: trained model artifacts and caches
- `提交结果/`: final CSV submissions
- `docker容器/`: Docker reproduction files
- `binaries/`: extracted binary samples used locally

## Output

The final submission file is `提交结果/submission_v1.1.csv` with columns:

```text
binary_id,label,cwe_id
```

`label=0` rows keep `cwe_id` empty.

Versioned model artifacts:

- `模型/tabular_bundle_v1.1.joblib`
- `模型/neural_bundle_v1.1.pt`
- `模型/fusion_config_v1.1.json`
