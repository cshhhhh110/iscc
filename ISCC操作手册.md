# ISCC 数据安全赛道 — 操作手册

> 写给小白看的，照着敲就行。不懂的别乱改，先问。

---

## 一、仓库在哪

你的项目在 `E:\赛题数据`，所有操作都要在这个目录下执行。

```powershell
cd E:\赛题数据
```

仓库地址：https://github.com/cshhhhh110/iscc

---

## 二、日常操作（最常用）

### 2.1 改完代码后，保存到 Git

```powershell
cd E:\赛题数据

# 第1步：看改了什么
git status

# 第2步：把改动加入暂存区
git add .

# 第3步：提交（写清楚这版做了什么）
git commit -m "v1.x: xxx改进xxx"

# 第4步：推到 GitHub
git push
```

### 2.2 四个步骤一条命令搞定

```powershell
git add . && git commit -m "更新xxx" && git push
```

---

## 三、新增文件/代码的完整流程

假设你在做二进制题 v2.1：

```powershell
# 1. 写完代码后看一眼变化
git status

# 2. 如果有新增的大文件（模型、数据），先确认 .gitignore 会排除它
#    看不懂这步就叫我帮你检查

# 3. 提交
git add .
git commit -m "二进制 v2.1: 增大epochs到35, 调整dropout"
git push
```

---

## 四、拉取最新代码（以后多台电脑用）

```powershell
git pull
```

如果提示冲突 → 停手，叫我来处理。

---

## 五、紧急操作（慎用！）

### 5.1 刚才的 commit 写错了，想改

```powershell
git commit --amend -m "正确的内容"
git push --force
```

### 5.2 某个文件改坏了，想回到上次提交的状态

```powershell
git checkout -- 文件路径
```

### 5.3 整个目录回到上次提交的状态

```powershell
git checkout -- .
```

---

## 六、提交记录怎么写（消息规范）

好的：
```
git commit -m "二进制 v2.1: epochs 25→35, seed 42单模达0.954"
git commit -m "修复: fusion threshold 从0.08改为0.5"
git commit -m "新增: 网络分类题 v1.5, 加入FT-Transformer"
```

坏的：
```
git commit -m "改了一下"
git commit -m "123"
git commit -m "asdf"
```

---

## 七、不要做的事

| 不要做 | 原因 |
|--------|------|
| `git push --force` 随便用 | 会覆盖远程历史，丢代码 |
| `git reset --hard` | 会丢掉没提交的修改 |
| 手动改 `.git` 目录里的东西 | 仓库会坏 |
| 提交大文件（`.pt` `.joblib` `.npy`）| 已经帮你忽略了，别手动 `git add -f` |

---

## 八、出问题了怎么办

| 问题 | 操作 |
|------|------|
| push 报错 `rejected` | 先 `git pull`，再 `git push` |
| pull 报错 `conflict` | 停手，叫我 |
| 提示 `not a git repository` | 检查是不是 cd 到了奇怪的地方，应该在 `E:\赛题数据` |
| 文件改坏了想回退 | `git checkout -- 文件名` |
| 不确定能不能做 | 问我 |

---

## 九、版本号速查

| 项目 | 最新版本 | 最佳成绩 |
|------|---------|---------|
| PowerShell 恶意脚本 | v1.7 | OOF ~0.756 |
| 二进制文件漏洞检测 | v2.0 | 平台 0.921 |
| 系统日志异常检测 | v1.3 | OOF ~0.92 |
| 网络安全智能分类 | v1.9 | — |

---

## 十、环境速查

```powershell
# GPU 环境 (二进制、网络分类)
conda activate iscc-gpu

# CPU 环境 (PowerShell、系统日志)
conda activate iscc-ml

# 训练
python 源码/train.py

# 预测
python 源码/predict.py

# 校验提交格式
python 源码/validate_submission.py --submission-path 提交结果/submission_xxx.csv
```

---

> 最后更新：2026-05-11
