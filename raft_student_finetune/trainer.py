"""
训练器模块
封装训练逻辑和回调函数
"""
import os
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from typing import Dict, Optional
from config import TrainingConfig


def create_training_arguments(training_config: TrainingConfig) -> TrainingArguments:
    """
    创建训练参数
    
    Args:
        training_config: 训练配置
        
    Returns:
        TrainingArguments
    """
    args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        
        # 优化器
        optim=training_config.optim,
        lr_scheduler_type=training_config.lr_scheduler_type,
        
        # 保存策略
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        
        # 日志
        logging_steps=training_config.logging_steps,
        logging_dir=os.path.join(training_config.output_dir, 'logs'),
        
        # 评估
        eval_strategy=training_config.eval_strategy,
        eval_steps=training_config.eval_steps,
        
        # 精度
        bf16=training_config.bf16,
        fp16=training_config.fp16,
        
        # 其他
        gradient_checkpointing=training_config.gradient_checkpointing,
        dataloader_num_workers=training_config.dataloader_num_workers,
        report_to=training_config.report_to,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,
        seed=training_config.seed,
        
        # 其他重要参数
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )
    
    return args


class CustomCallback(TrainerCallback):
    """自定义训练回调"""
    
    def __init__(self):
        self.best_eval_loss = float('inf')
        self.best_model_checkpoint = None
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """日志回调"""
        if logs is not None:
            if 'loss' in logs:
                print(f"Step {state.global_step}: train_loss={logs['loss']:.4f}")
            if 'eval_loss' in logs:
                eval_loss = logs['eval_loss']
                print(f"Step {state.global_step}: eval_loss={eval_loss:.4f}")
                
                # 记录最佳模型
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.best_model_checkpoint = f"checkpoint-{state.global_step}"
                    print(f"  ✓ 新的最佳模型! (eval_loss: {eval_loss:.4f})")
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始回调"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        print(f"输出目录: {args.output_dir}")
        print(f"训练轮数: {args.num_train_epochs}")
        print(f"Batch size: {args.per_device_train_batch_size}")
        print(f"梯度累积步数: {args.gradient_accumulation_steps}")
        print(f"有效batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size}")
        print(f"学习率: {args.learning_rate}")
        print("="*60 + "\n")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束回调"""
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"总步数: {state.global_step}")
        print(f"最佳eval_loss: {self.best_eval_loss:.4f}")
        if self.best_model_checkpoint:
            print(f"最佳模型检查点: {self.best_model_checkpoint}")
        print("="*60 + "\n")
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """保存回调"""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        print(f"保存检查点: {checkpoint_folder}")


class RAFTTrainer(Trainer):
    """自定义RAFT训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算损失
        
        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
            
        Returns:
            loss或(loss, outputs)
        """
        # 前向传播
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 如果loss是NaN,打印警告
        if torch.isnan(loss):
            print("警告: 检测到NaN loss!")
            # 可以选择跳过这个batch或采取其他措施
            loss = torch.tensor(0.0, requires_grad=True).to(loss.device)
        
        return (loss, outputs) if return_outputs else loss


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_config: TrainingConfig,
    data_collator=None
):
    """
    创建训练器
    
    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练集
        eval_dataset: 验证集
        training_config: 训练配置
        data_collator: 数据整理函数
        
    Returns:
        Trainer
    """
    # 创建训练参数
    training_args = create_training_arguments(training_config)
    
    # 创建自定义回调
    callback = CustomCallback()
    
    # 创建训练器
    trainer = RAFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[callback]
    )
    
    return trainer


def resume_training(trainer: Trainer, checkpoint_path: Optional[str] = None):
    """
    恢复训练
    
    Args:
        trainer: 训练器
        checkpoint_path: 检查点路径
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从检查点恢复训练: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("开始新的训练")
        trainer.train()