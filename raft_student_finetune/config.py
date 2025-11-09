"""
配置管理模块
包含所有训练相关的超参数和路径配置
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """模型相关配置"""
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    use_flash_attention_2: bool = False
    torch_dtype: str = "bfloat16"  # float16, bfloat16, float32


@dataclass
class LoRAConfig:
    """LoRA微调配置"""
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"


@dataclass
class TrainingConfig:
    """训练相关配置"""
    output_dir: str = "./output"
    train_file: str = "raft_dataset.json"
    validation_split: float = 0.1
    
    # 训练超参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 优化器配置
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    
    # 保存和日志
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    
    # 其他配置
    max_seq_length: int = 4096
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 4
    report_to: str = "tensorboard"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42


@dataclass
class DataConfig:
    """数据处理配置"""
    max_prompt_length: int = 3072
    max_answer_length: int = 1024
    template_prefix: str = """- 问题: {question}
- 假设/已知信息: 
- CoT推理:
  1) 
  2) 
  3) 
- 初步诊断建议(含不确定度): 
- 证据引用: 
- 不足信息与后续建议: 
- 紧急就医指示(红旗症状): """


@dataclass
class InferenceConfig:
    """推理配置"""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True