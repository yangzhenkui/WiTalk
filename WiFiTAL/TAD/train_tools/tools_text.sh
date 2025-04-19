#!/bin/bash

# 检查参数数量（只需要一个 GPU ID）
if [ $# -ne 1 ]; then
    echo "Usage: $0 <CUDA_DEVICE>"
    exit 1
fi

# 检查可用 GPU 数量
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
if [ -z "$GPU_COUNT" ]; then
    echo "Error: Failed to detect GPUs. Check if nvidia-smi is installed and working."
    exit 1
fi

# 确保指定的 GPU ID 有效
if [ $1 -ge $GPU_COUNT ]; then
    echo "Error: Invalid device ordinal. Available GPUs: 0 to $((GPU_COUNT-1))"
    exit 1
fi

# 设置日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 从 YAML 配置文件中提取 embed_type（假设配置文件中有 embed_type 字段）
CONFIG_FILE="configs/WiFiTAD_simple_desc_llama.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found."
    exit 1
fi

# 使用 grep 和 sed 从 YAML 中提取 embed_type 的值
EMBED_TYPE=$(grep "embed_type" $CONFIG_FILE | sed 's/.*: *\(.*\)/\1/' | tr -d ' ')
if [ -z "$EMBED_TYPE" ]; then
    echo "Warning: Could not find embed_type in $CONFIG_FILE, defaulting to 'simple'"
    EMBED_TYPE="simple"
fi

# 设置日志文件，文件名包含 embed_type
LOG_FILE="$LOG_DIR/embedding_llama_7b_${EMBED_TYPE}.log"

echo ""

# 启动训练
echo "start training" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$1 python3 TAD/utils/train_text.py $CONFIG_FILE >> $LOG_FILE 2>&1

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo "Training failed, check $LOG_FILE for details" | tee -a $LOG_FILE
    exit 1
fi

# 启动检测
echo "start detecting..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$1 python3 TAD/utils/test_text.py $CONFIG_FILE >> $LOG_FILE 2>&1

# 检查检测是否成功
if [ $? -ne 0 ]; then
    echo "Detection failed, check $LOG_FILE for details" | tee -a $LOG_FILE
    exit 1
fi

# 启动评估
echo "start eval..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$1 python3 TAD/utils/eval_text.py $CONFIG_FILE >> $LOG_FILE 2>&1

# 检查评估是否成功
if [ $? -ne 0 ]; then
    echo "Evaluation failed, check $LOG_FILE for details" | tee -a $LOG_FILE
    exit 1
fi

echo "All tasks completed successfully!" | tee -a $LOG_FILE