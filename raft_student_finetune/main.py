"""
主程序入口
用于训练RAFT知识蒸馏模型
"""
import os
import torch
from dataclasses import asdict
import gc

# 清理缓存
torch.cuda.empty_cache()
gc.collect()

# 设置显存分配策略
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# 导入自定义模块
from config import ModelConfig, LoRAConfig, TrainingConfig, DataConfig
from model import load_tokenizer, load_model, prepare_model_for_training, save_model_and_tokenizer
from dataset import RAFTDataset, FilteredDataset, split_dataset, create_collate_fn
from trainer import create_trainer, resume_training
from utils import (
    set_seed,
    print_gpu_info,
    check_dataset_format,
    save_training_config,
    print_training_summary,
    estimate_training_time
)


def main():
    """主函数（不使用命令行参数）"""
    # =====================================================
    # 1️⃣ 创建所有配置对象（直接使用 config.py 的默认值）
    # =====================================================
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()

    # =====================================================
    # 2️⃣ 打印标题与环境信息
    # =====================================================
    print("\n" + "="*60)
    print("RAFT 知识蒸馏训练系统")
    print("="*60 + "\n")

    set_seed(training_config.seed)
    print_gpu_info()

    # =====================================================
    # 3️⃣ 检查数据集格式
    # =====================================================
    print("检查数据集格式...")
    if not check_dataset_format(training_config.train_file):
        print("❌ 数据集格式检查失败，请修正后重试。")
        return

    # =====================================================
    # 4️⃣ 打印训练摘要并保存配置
    # =====================================================
    print_training_summary(training_config, model_config, lora_config)

    config_dict = {
        'model_config': asdict(model_config),
        'lora_config': asdict(lora_config),
        'training_config': asdict(training_config),
        'data_config': asdict(data_config)
    }
    save_training_config(config_dict, training_config.output_dir)

    # =====================================================
    # 5️⃣ 加载分词器和模型
    # =====================================================
    print("\n加载分词器...")
    tokenizer = load_tokenizer(model_config)

    print("\n加载模型...")
    model = load_model(model_config, lora_config)

    print("\n准备模型用于训练...")
    model = prepare_model_for_training(model, training_config)

    # =====================================================
    # 6️⃣ 加载和划分数据集
    # =====================================================
    print("\n加载数据集...")
    full_dataset = RAFTDataset(
        data_path=training_config.train_file,
        tokenizer=tokenizer,
        data_config=data_config,
        max_length=training_config.max_seq_length
    )

    # 过滤无效样本
    filtered_dataset = FilteredDataset(full_dataset)

    print("\n划分训练集和验证集...")
    train_dataset, eval_dataset = split_dataset(
        filtered_dataset,
        validation_split=training_config.validation_split,
        seed=training_config.seed
    )

    data_collator = create_collate_fn(tokenizer, training_config.max_seq_length)

    # =====================================================
    # 7️⃣ 估算训练时间
    # =====================================================
    num_samples = len(train_dataset)
    estimated_time = estimate_training_time(
        num_samples=num_samples,
        batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_epochs=training_config.num_train_epochs,
        seconds_per_step=1.5
    )
    print(f"\n估算训练时间: {estimated_time}")

    # =====================================================
    # 8️⃣ 创建训练器并开始训练
    # =====================================================
    print("\n创建训练器...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        data_collator=data_collator
    )

    try:
        print("\n" + "="*60)
        print("准备开始训练...")
        print("="*60 + "\n")

        resume_training(trainer, training_config.resume_from_checkpoint)

        print("\n保存最终模型...")
        final_output_dir = os.path.join(training_config.output_dir, 'final_model')
        save_model_and_tokenizer(model, tokenizer, final_output_dir)

        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print(f"模型已保存至: {final_output_dir}")
        print(f"可以使用以下命令进行推理测试:")
        print(f"python test_inference.py --model_path {final_output_dir}")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断，部分结果已保存。")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
