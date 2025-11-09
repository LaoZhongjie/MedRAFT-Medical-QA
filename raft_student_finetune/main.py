"""
主程序入口
用于训练RAFT知识蒸馏模型
"""
import os
import argparse
import torch
from dataclasses import asdict

# 导入自定义模块
from config import ModelConfig, LoRAConfig, TrainingConfig, DataConfig
from model import load_tokenizer, load_model, prepare_model_for_training, save_model_and_tokenizer
from dataset import RAFTDataset, split_dataset, create_collate_fn
from trainer import create_trainer, resume_training
from utils import (
    set_seed,
    print_gpu_info,
    check_dataset_format,
    save_training_config,
    print_training_summary,
    estimate_training_time
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAFT知识蒸馏训练")
    
    # 数据相关
    parser.add_argument('--train_file', type=str, required=True,
                        help='训练数据文件路径')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='验证集比例')
    
    # 模型相关
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='基础模型名称或路径')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='模型缓存目录')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='数据类型')
    
    # LoRA相关
    parser.add_argument('--lora_r', type=int, default=64,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=128,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # 训练相关
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2,
                        help='每设备训练batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--max_seq_length', type=int, default=4096,
                        help='最大序列长度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 其他
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 打印标题
    print("\n" + "="*60)
    print("RAFT 知识蒸馏训练系统")
    print("="*60 + "\n")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印GPU信息
    print_gpu_info()
    
    # 检查数据集格式
    print("检查数据集格式...")
    if not check_dataset_format(args.train_file):
        print("数据集格式检查失败,请修正后重试")
        return
    
    # 创建配置对象
    model_config = ModelConfig(
        model_name_or_path=args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=args.torch_dtype
    )
    
    lora_config = LoRAConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    training_config = TrainingConfig(
        train_file=args.train_file,
        output_dir=args.output_dir,
        validation_split=args.validation_split,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    data_config = DataConfig()
    
    # 打印训练配置摘要
    print_training_summary(training_config, model_config, lora_config)
    
    # 保存配置
    config_dict = {
        'model_config': asdict(model_config),
        'lora_config': asdict(lora_config),
        'training_config': asdict(training_config),
        'data_config': asdict(data_config)
    }
    save_training_config(config_dict, args.output_dir)
    
    # 加载分词器
    print("\n加载分词器...")
    tokenizer = load_tokenizer(model_config)
    
    # 加载模型
    print("\n加载模型...")
    model = load_model(model_config, lora_config)
    
    # 准备模型用于训练
    print("\n准备模型用于训练...")
    model = prepare_model_for_training(model, training_config)
    
    # 加载数据集
    print("\n加载数据集...")
    full_dataset = RAFTDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        data_config=data_config,
        max_length=args.max_seq_length
    )
    
    # 划分数据集
    print("\n划分训练集和验证集...")
    train_dataset, eval_dataset = split_dataset(
        full_dataset,
        validation_split=args.validation_split,
        seed=args.seed
    )
    
    # 创建数据整理函数
    data_collator = create_collate_fn(tokenizer, args.max_seq_length)
    
    # 估算训练时间
    num_samples = len(train_dataset)
    estimated_time = estimate_training_time(
        num_samples=num_samples,
        batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_train_epochs,
        seconds_per_step=1.5  # 这是一个估算值
    )
    print(f"\n估算训练时间: {estimated_time}")
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        data_collator=data_collator
    )
    
    # 开始训练
    try:
        print("\n" + "="*60)
        print("准备开始训练...")
        print("="*60 + "\n")
        
        resume_training(trainer, args.resume_from_checkpoint)
        
        # 保存最终模型
        print("\n保存最终模型...")
        final_output_dir = os.path.join(args.output_dir, 'final_model')
        save_model_and_tokenizer(model, tokenizer, final_output_dir)
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"模型已保存至: {final_output_dir}")
        print(f"可以使用以下命令进行推理测试:")
        print(f"python test_inference.py --model_path {final_output_dir}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("部分训练结果已保存在检查点中")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()