"""
模型加载与配置模块
负责加载预训练模型、配置LoRA等
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from config import ModelConfig, LoRAConfig

def load_tokenizer(model_config: ModelConfig):
    """
    加载分词器
    
    Args:
        model_config: 模型配置
        
    Returns:
        tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.cache_dir,
        trust_remote_code=model_config.trust_remote_code,
        padding_side='right',  # 训练时padding在右边
        use_fast=True
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print(f"✓ 分词器加载完成")
    print(f"  词表大小: {len(tokenizer)}")
    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    
    return tokenizer


def load_model(model_config: ModelConfig, lora_config: LoRAConfig):
    """
    加载模型并配置LoRA
    
    Args:
        model_config: 模型配置
        lora_config: LoRA配置
        
    Returns:
        model
    """
    # 配置数据类型
    torch_dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    torch_dtype = torch_dtype_map.get(model_config.torch_dtype, torch.bfloat16)
    
    print(f"开始加载模型: {model_config.model_name_or_path}")
    print(f"使用数据类型: {model_config.torch_dtype}")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.cache_dir,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map='auto',
        use_flash_attention_2=model_config.use_flash_attention_2
    )
    
    print(f"✓ 基础模型加载完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # 配置LoRA
    if lora_config.use_lora:
        print("\n配置LoRA...")
        
        # 准备模型进行量化训练(如果需要)
        model = prepare_model_for_kbit_training(model)
        
        # 配置LoRA参数
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.lora_r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.lora_target_modules,
            bias=lora_config.lora_bias
        )
        
        # 应用LoRA
        model = get_peft_model(model, peft_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params
        
        print(f"✓ LoRA配置完成")
        print(f"  LoRA rank: {lora_config.lora_r}")
        print(f"  LoRA alpha: {lora_config.lora_alpha}")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M ({trainable_percent:.2f}%)")
        print(f"  目标模块: {lora_config.lora_target_modules}")
        
        model.print_trainable_parameters()
    
    return model


def prepare_model_for_training(model, training_config):
    """
    为训练准备模型
    
    Args:
        model: 模型
        training_config: 训练配置
        
    Returns:
        准备好的模型
    """
    # 启用gradient checkpointing以节省显存
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ 启用gradient checkpointing")
    
    # 确保embeddings层可训练
    if hasattr(model, 'get_input_embeddings'):
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
    
    return model


def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    """
    保存模型和分词器
    
    Args:
        model: 模型
        tokenizer: 分词器
        output_dir: 输出目录
    """
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
        print(f"✓ 模型已保存至: {output_dir}")
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    print(f"✓ 分词器已保存至: {output_dir}")


def load_trained_model(model_path: str, model_config: ModelConfig):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型路径
        model_config: 模型配置
        
    Returns:
        (model, tokenizer)
    """
    from peft import PeftModel
    
    print(f"从 {model_path} 加载训练好的模型...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # 加载基础模型
    torch_dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    torch_dtype = torch_dtype_map.get(model_config.torch_dtype, torch.bfloat16)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map='auto'
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✓ 模型加载完成")
    
    return model, tokenizer