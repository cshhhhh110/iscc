#!/bin/bash
set -e
PYTHON="C:/Users/30816/.conda/envs/iscc-gpu/python.exe"
BIN="E:/赛题数据/二进制文件漏洞检测"
LOG="E:/赛题数据/系统日志异常检测挑战"

echo "===== [1/4] 二进制 v1.9 训练开始 $(date) ====="
$PYTHON "$BIN/源码/train.py" --skip-pseudo > "$BIN/train_v1.9.log" 2> "$BIN/train_v1.9_err.log"
echo "===== [1/4] 二进制 v1.9 训练完成 $(date) ====="

echo "===== [2/4] 二进制 v1.9 预测开始 $(date) ====="
$PYTHON "$BIN/源码/test.py" > "$BIN/test_v1.9.log" 2> "$BIN/test_v1.9_err.log"
echo "===== [2/4] 二进制 v1.9 预测完成 $(date) ====="

echo "===== [3/4] 系统日志 训练开始 $(date) ====="
$PYTHON "$LOG/源码/train.py" > "$LOG/train.log" 2> "$LOG/train_err.log"
echo "===== [3/4] 系统日志 训练完成 $(date) ====="

echo "===== [4/4] 系统日志 预测开始 $(date) ====="
$PYTHON "$LOG/源码/predict.py" > "$LOG/predict.log" 2> "$LOG/predict_err.log"
echo "===== [4/4] 系统日志 预测完成 $(date) ====="

echo "===== 全部完成 $(date) ====="
touch "E:/赛题数据/ALL_DONE.marker"
