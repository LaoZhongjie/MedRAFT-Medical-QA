#!/bin/bash

# RAFT 知识蒸馏训练启动脚本
# 使用方法: bash run_training.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 训练参数配置
TRAIN_FILE="raft_dataset.json"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="./output"
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=4096
LORA_R=64
LORA_ALPHA=128
SEED=42

# 打印配置信息
echo "=================================="
echo "RAFT 知识蒸馏训练"
echo "=================================="
echo "训练文件: ${TRAIN_FILE}"
echo "模型: ${MODEL_NAME}"
echo "输出目录: ${OUTPUT_DIR}"
echo "训练轮数: ${NUM_EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "梯度累积: ${GRAD_ACCUM}"
echo "学习率: ${LEARNING_RATE}"
echo "=================================="
echo ""

# 检查数据文件是否存在
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "错误: 训练文件 ${TRAIN_FILE} 不存在!"
    echo "提示: 可以使用以下命令生成示例数据集:"
    echo "  python -c 'from utils import create_sample_dataset; create_sample_dataset()'"
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 开始训练
echo "开始训练..."
echo ""

python main.py \
  --train_file ${TRAIN_FILE} \
  --model_name ${MODEL_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate ${LEARNING_RATE} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --seed ${SEED}

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "训练成功完成!"
    echo "=================================="
    echo "模型保存位置: ${OUTPUT_DIR}/final_model"
    echo ""
    echo "使用以下命令进行推理测试:"
    echo "  python test_inference.py --model_path ${OUTPUT_DIR}/final_model"
    echo "=================================="
else
    echo ""
    echo "=================================="
    echo "训练失败,请检查错误信息"
    echo "=================================="
    exit 1
fi