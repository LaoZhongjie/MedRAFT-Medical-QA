# RAFT 知识蒸馏训练系统

一个完整的知识蒸馏训练工程,用于训练学生模型学习教师模型的推理风格。支持 Qwen、LLaMA 等主流大语言模型。

## 📋 项目结构

```
.
├── config.py              # 配置管理模块
├── dataset.py             # 数据集处理模块
├── model.py               # 模型加载与配置模块
├── trainer.py             # 训练器模块
├── inference.py           # 推理模块
├── utils.py               # 工具函数模块
├── main.py                # 主程序入口
├── test_inference.py      # 推理测试脚本
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 如果使用Flash Attention 2(可选,可提升训练速度)
pip install flash-attn --no-build-isolation
```

### 2. 数据准备

准备训练数据集 `raft_dataset.json`,格式如下:

```json
[
  {
    "question": "如果我漏服或多服了治疗2型糖尿病的药,该怎么办?",
    "documents": [
      {
        "content": "2型糖尿病药物治疗指南...",
        "type": "oracle"
      },
      {
        "content": "其他相关文档...",
        "type": "distractor"
      }
    ],
    "teacher_answer": "- 问题: ...\n- 假设/已知信息: ...\n- CoT推理:\n  1) ...\n  2) ...\n  3) ...\n- 初步诊断建议(含不确定度): ...\n- 证据引用: ...\n- 不足信息与后续建议: ...\n- 紧急就医指示(红旗症状): ..."
  }
]
```

**生成示例数据集:**

```python
from utils import create_sample_dataset
create_sample_dataset("sample_dataset.json")
```

### 3. 开始训练

**基础训练命令:**

```bash
python main.py \
  --train_file raft_dataset.json \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4
```

**完整参数示例:**

```bash
python main.py \
  --train_file raft_dataset.json \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./output \
  --validation_split 0.1 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_seq_length 4096 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --seed 42
```

**从检查点恢复训练:**

```bash
python main.py \
  --train_file raft_dataset.json \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./output \
  --resume_from_checkpoint ./output/checkpoint-100
```

### 4. 推理测试

**使用默认测试样本:**

```bash
python test_inference.py \
  --model_path ./output/final_model \
  --base_model Qwen/Qwen2.5-7B-Instruct
```

**使用自定义测试文件:**

```bash
python test_inference.py \
  --model_path ./output/final_model \
  --test_file test_samples.json \
  --base_model Qwen/Qwen2.5-7B-Instruct
```

**使用命令行问题测试:**

```bash
python test_inference.py \
  --model_path ./output/final_model \
  --question "你的问题" \
  --base_model Qwen/Qwen2.5-7B-Instruct
```

## ⚙️ 配置说明

### 模型配置 (ModelConfig)

- `model_name_or_path`: 基础模型名称或路径
- `torch_dtype`: 数据类型 (float16/bfloat16/float32)
- `use_flash_attention_2`: 是否使用Flash Attention 2

### LoRA 配置 (LoRAConfig)

- `lora_r`: LoRA rank (建议 16-128)
- `lora_alpha`: LoRA alpha (通常是 r 的 2 倍)
- `lora_dropout`: Dropout 率
- `lora_target_modules`: 目标模块列表

### 训练配置 (TrainingConfig)

- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 每设备 batch size
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 学习率
- `max_seq_length`: 最大序列长度
- `gradient_checkpointing`: 是否启用梯度检查点(节省显存)

## 💡 核心特性

### 1. LoRA 微调
- 使用 PEFT 库实现高效的 LoRA 微调
- 大幅减少训练参数和显存占用
- 支持自定义 LoRA 配置

### 2. Gradient Checkpointing
- 自动启用梯度检查点以节省显存
- 适合在有限 GPU 资源下训练大模型

### 3. 混合精度训练
- 支持 BF16/FP16 混合精度训练
- 提升训练速度,减少显存占用

### 4. 断点续训
- 自动保存训练检查点
- 支持从任意检查点恢复训练

### 5. 结构化输出
- 训练学生模型生成结构化医疗建议
- 包含 CoT 推理、证据引用等模块

### 6. 自动评估
- 训练过程中自动在验证集上评估
- 保存最佳模型

## 📊 显存需求

以 Qwen2.5-7B 为例:

| 配置 | Batch Size | 梯度累积 | 显存需求 |
|------|-----------|---------|---------|
| BF16 + LoRA64 | 1 | 8 | ~18GB |
| BF16 + LoRA64 | 2 | 8 | ~24GB |
| BF16 + LoRA64 | 4 | 4 | ~32GB |

**节省显存的技巧:**
1. 启用 `gradient_checkpointing`
2. 减小 `per_device_train_batch_size`
3. 增加 `gradient_accumulation_steps`
4. 减小 `max_seq_length`
5. 使用较小的 `lora_r`

## 🔧 高级用法

### 自定义数据处理

修改 `dataset.py` 中的 `_build_prompt()` 方法来自定义 prompt 格式:

```python
def _build_prompt(self, item: Dict[str, Any]) -> str:
    # 自定义你的 prompt 构建逻辑
    question = item['question']
    documents = item['documents']
    # ...
    return prompt
```

### 自定义训练回调

在 `trainer.py` 中扩展 `CustomCallback` 类:

```python
class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 添加自定义逻辑
        pass
```

### 使用不同的基础模型

支持任何 Hugging Face 上的 Causal LM 模型:

```bash
# LLaMA 系列
python main.py --model_name meta-llama/Llama-2-7b-hf ...

# Baichuan 系列
python main.py --model_name baichuan-inc/Baichuan2-7B-Base ...

# ChatGLM 系列
python main.py --model_name THUDM/chatglm3-6b ...
```

## 📝 输出说明

训练过程中会生成以下输出:

```
output/
├── checkpoint-100/          # 训练检查点
├── checkpoint-200/
├── final_model/             # 最终模型
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── logs/                    # TensorBoard 日志
└── training_config.json     # 训练配置备份
```

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案:**
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing`
- 减小 `max_seq_length`

### 2. 训练 Loss 不下降

**检查项:**
- 学习率是否合适
- 数据集是否正确
- 验证 labels 是否正确设置

### 3. 生成结果格式不符合预期

**解决方案:**
- 增加训练数据量
- 延长训练时间
- 调整 prompt 模板
- 在推理时使用更低的 temperature

## 📚 参考资料

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [Qwen 模型](https://github.com/QwenLM/Qwen)

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## ✉️ 联系方式

如有问题,请提交 Issue 或联系项目维护者。