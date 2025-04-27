#!/bin/bash

# 检查参数数量（需要 GPU ID、embed_type 和 embed_model_name）
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <CUDA_DEVICE> <EMBED_TYPE> [<EMBED_MODEL_NAME>]"
    echo "Example: $0 0 simple clip-vit-large-patch14"
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

# 获取 embed_type
EMBED_TYPE=$2

# 如果提供了第三个参数，则使用它作为 embed_model_name
if [ $# -eq 3 ]; then
    EMBED_MODEL_NAME=$3
else
    # 否则尝试从配置文件中提取
    CONFIG_FILE="configs/WiFiTAD_${EMBED_TYPE}.yaml"  # 默认配置文件名称
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file $CONFIG_FILE not found and no embed_model_name specified."
        exit 1
    fi
    
    EMBED_MODEL_NAME=$(grep "embed_model_name" $CONFIG_FILE | sed 's/.*: *\(.*\)/\1/' | tr -d ' ')
    if [ -z "$EMBED_MODEL_NAME" ]; then
        echo "Warning: Could not find embed_model_name in $CONFIG_FILE, defaulting to 'clip-vit-large-patch14'"
        EMBED_MODEL_NAME="clip-vit-large-patch14"
    fi
fi

# 设置最终的配置文件路径
CONFIG_FILE="configs/WiFiTAD_${EMBED_TYPE}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found."
    exit 1
fi

# 设置日志文件，文件名包含 embed_type 和 embed_model_name
LOG_FILE="$LOG_DIR/clip/embedding_${EMBED_TYPE}_${EMBED_MODEL_NAME}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR/clip"

echo ""

# 启动训练
echo "start training with embed_type: $EMBED_TYPE, embed_model_name: $EMBED_MODEL_NAME" | tee -a $LOG_FILE
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