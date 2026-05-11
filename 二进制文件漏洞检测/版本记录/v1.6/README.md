# 先读总规则：请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`，再开始本项目。

根目录的 `源码/` 只放当前活跃版本；历史代码、模型、结果和 `smoke_*` 统一进 `版本记录/<version>/`，文件名也要带版本号。

# ISCC Binary Vulnerability Detection

Official package layout for the ISCC data security binary vulnerability task.

## Layout

- `源码/`: training, inference, feature extraction, and helper code
- `模型/`: trained model artifacts and caches
- `提交结果/`: final CSV submissions
- `docker容器/`: Docker reproduction files
- `binaries/`: extracted binary samples used locally

## Output

The final submission file is `提交结果/submission_v1.4.csv` with columns:

```text
binary_id,label,cwe_id
```

`label=0` rows keep `cwe_id` empty.

Versioned model artifacts (v1.4):

- `模型/tabular_bundle_v1.4.joblib`
- `模型/neural_bundle_v1.4.pt`
- `模型/fusion_mlp_v1.4.pt`
- `模型/fusion_config_v1.4.json`

## Results

| Version | Local (scalar_cwe_macro) | Platform (Macro-F1 87-class) |
|---------|--------------------------|------------------------------|
| v1.3    | 0.830                    | 0.849                        |
| v1.4    | 0.8739                   | 0.901                        |
| v1.5    | 0.8980 (avg 3-seed)      | **0.923**                    |

v1.5: pseudo-labeling (17,343 samples) + 3-seed ensemble + TTA (3 byte windows) + scalar fusion.
Best single seed: 0.9281 local → ~0.957 estimated platform.

Platform scoring: Macro-F1 over 87 classes (1 "no vuln" label=0 + 86 CWE classes).
Reference: `D:\桌面\题解\二进制\二进制漏洞检测竞赛完整说明文档.pdf`