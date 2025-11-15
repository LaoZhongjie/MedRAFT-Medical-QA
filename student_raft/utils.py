"""
工具函数模块
包含各种辅助函数
"""
import os
import json
import random
import numpy as np
import torch
from typing import Dict, Any, List


def set_seed(seed: int):
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✓ 随机种子设置为: {seed}")


def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU信息")
        print("="*60)
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print("="*60 + "\n")
    else:
        print("警告: 未检测到可用的GPU,将使用CPU训练(速度会很慢)")


def check_dataset_format(dataset_path: str) -> bool:
    """
    检查数据集格式是否正确
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        格式是否正确
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"错误: 数据集应该是一个列表,当前类型: {type(data)}")
            return False
        
        if len(data) == 0:
            print("错误: 数据集为空")
            return False
        
        # 检查第一条样本的格式
        sample = data[0]
        required_fields = ['question', 'documents', 'teacher_answer']
        
        for field in required_fields:
            if field not in sample:
                print(f"错误: 样本缺少必需字段: {field}")
                return False
        
        # 检查documents格式
        if not isinstance(sample['documents'], list):
            print("错误: documents字段应该是一个列表")
            return False
        
        if len(sample['documents']) > 0:
            doc = sample['documents'][0]
            if 'content' not in doc:
                print("错误: document缺少content字段")
                return False
        
        print(f"✓ 数据集格式检查通过")
        print(f"  样本数量: {len(data)}")
        print(f"  第一条样本的文档数: {len(sample['documents'])}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析JSON文件: {e}")
        return False
    except FileNotFoundError:
        print(f"错误: 文件不存在: {dataset_path}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def save_training_config(config_dict: Dict[str, Any], output_dir: str):
    """
    保存训练配置
    
    Args:
        config_dict: 配置字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'training_config.json')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 训练配置已保存至: {config_path}")


def load_json_file(file_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        解析后的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Any, file_path: str):
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_training_summary(training_config, model_config, lora_config):
    """
    打印训练配置摘要
    
    Args:
        training_config: 训练配置
        model_config: 模型配置
        lora_config: LoRA配置
    """
    print("\n" + "="*60)
    print("训练配置摘要")
    print("="*60)
    
    print("\n【模型配置】")
    print(f"  模型: {model_config.model_name_or_path}")
    print(f"  数据类型: {model_config.torch_dtype}")
    
    print("\n【LoRA配置】")
    print(f"  使用LoRA: {lora_config.use_lora}")
    if lora_config.use_lora:
        print(f"  LoRA rank: {lora_config.lora_r}")
        print(f"  LoRA alpha: {lora_config.lora_alpha}")
        print(f"  Dropout: {lora_config.lora_dropout}")
    
    print("\n【训练配置】")
    print(f"  训练文件: {training_config.train_file}")
    print(f"  输出目录: {training_config.output_dir}")
    print(f"  训练轮数: {training_config.num_train_epochs}")
    print(f"  Batch size: {training_config.per_device_train_batch_size}")
    print(f"  梯度累积: {training_config.gradient_accumulation_steps}")
    print(f"  学习率: {training_config.learning_rate}")
    print(f"  最大序列长度: {training_config.max_seq_length}")
    print(f"  验证集比例: {training_config.validation_split}")
    
    print("\n【优化配置】")
    print(f"  Gradient checkpointing: {training_config.gradient_checkpointing}")
    print(f"  混合精度: bf16={training_config.bf16}, fp16={training_config.fp16}")
    
    print("="*60 + "\n")


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    seconds_per_step: float = 1.0
) -> str:
    """
    估算训练时间
    
    Args:
        num_samples: 样本数
        batch_size: batch大小
        gradient_accumulation_steps: 梯度累积步数
        num_epochs: 训练轮数
        seconds_per_step: 每步耗时(秒)
        
    Returns:
        估算时间字符串
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * seconds_per_step
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return f"约 {hours} 小时 {minutes} 分钟 (假设每步 {seconds_per_step} 秒)"


def create_sample_dataset(output_path: str = "sample_dataset.json"):
    """
    创建示例数据集
    
    Args:
        output_path: 输出路径
    """
    sample_data = [
        {
            "question": "如果我漏服或多服了治疗2型糖尿病的药,该怎么办?",
            "documents": [
                {
                    "content": "2型糖尿病药物治疗指南:如果漏服一次药物,不应在下次服药时加倍剂量。应按照正常时间服用下一次药物。如果多服药物,应密切监测血糖,必要时就医。",
                    "type": "oracle"
                },
                {
                    "content": "高血压药物使用注意事项:高血压患者应规律服药,不可随意增减剂量。",
                    "type": "distractor"
                }
            ],
            "teacher_answer": """- 问题: 如果我漏服或多服了治疗2型糖尿病的药,该怎么办?
- 假设/已知信息: 患者正在服用2型糖尿病治疗药物,出现了漏服或多服的情况。
- CoT推理:
  1) 漏服药物:根据指南,漏服一次不应加倍补服,应按正常时间服用下一次。
  2) 多服药物:需要密切监测血糖水平,防止低血糖风险。
  3) 如果出现低血糖症状(如头晕、出汗、心悸),需要立即就医。
- 初步诊断建议(含不确定度): 漏服按正常时间继续服药(置信度:高);多服需监测血糖并可能就医(置信度:高)。
- 证据引用: 根据《2型糖尿病药物治疗指南》,漏服不应加倍剂量,多服需监测血糖。
- 不足信息与后续建议: 需要了解具体药物类型、剂量、多服的时间和数量,以便给出更精确建议。
- 紧急就医指示(红旗症状): 如出现严重低血糖症状(意识模糊、抽搐、昏迷)、持续头晕出汗、心悸等,应立即就医。"""
        }
    ]
    
    save_json_file(sample_data, output_path)
    print(f"✓ 示例数据集已创建: {output_path}")


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"