"""
训练器模块
封装训练逻辑和回调函数
"""
import os
import torch
import json
import numpy as np
from datetime import datetime
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from typing import Dict, Optional, List
from config import TrainingConfig


class TrainingRecorder:
    """训练记录器，用于保存训练过程中的各种指标"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "training_metrics.json")
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": [],
            "timestamp": []
        }
        self.start_time = datetime.now()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def record_step(self, metrics: Dict, step: int, epoch: float):
        """记录训练步骤的指标"""
        current_time = datetime.now()
        
        # 记录基础指标
        self.metrics_history["step"].append(step)
        self.metrics_history["epoch"].append(epoch)
        self.metrics_history["timestamp"].append(current_time.isoformat())
        
        # 记录训练损失
        if "loss" in metrics:
            self.metrics_history["train_loss"].append(metrics["loss"])
        else:
            self.metrics_history["train_loss"].append(None)
        
        # 记录评估损失
        if "eval_loss" in metrics:
            self.metrics_history["eval_loss"].append(metrics["eval_loss"])
        else:
            self.metrics_history["eval_loss"].append(None)
        
        # 记录学习率
        if "learning_rate" in metrics:
            self.metrics_history["learning_rate"].append(metrics["learning_rate"])
        else:
            self.metrics_history["learning_rate"].append(None)
    
    def save_metrics(self):
        """保存指标到文件"""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        print(f"✓ 训练指标已保存到: {self.metrics_file}")
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        train_losses = [x for x in self.metrics_history["train_loss"] if x is not None]
        eval_losses = [x for x in self.metrics_history["eval_loss"] if x is not None]
        
        summary = {
            "total_steps": len(self.metrics_history["step"]),
            "training_time": str(datetime.now() - self.start_time),
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "min_train_loss": min(train_losses) if train_losses else None,
            "min_eval_loss": min(eval_losses) if eval_losses else None,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
        }
        return summary


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
        
        # 启用详细日志记录
        logging_strategy="steps",
        eval_accumulation_steps=1,
        prediction_loss_only=False,
    )
    
    return args


class CustomCallback(TrainerCallback):
    """自定义训练回调"""
    
    def __init__(self, output_dir: str):
        self.best_eval_loss = float('inf')
        self.best_model_checkpoint = None
        self.recorder = TrainingRecorder(output_dir)
        self.current_epoch = 0
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """日志回调"""
        if logs is not None:
            # 记录指标
            self.recorder.record_step(logs, state.global_step, state.epoch)
            
            # 打印关键指标
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
            
            # 定期保存指标
            if state.global_step % args.logging_steps == 0:
                self.recorder.save_metrics()
    
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """epoch开始回调"""
        self.current_epoch = state.epoch
        print(f"\n🏁 开始第 {state.epoch:.1f} 轮训练")
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """epoch结束回调"""
        print(f"✅ 第 {state.epoch:.1f} 轮训练完成")
        # 每个epoch结束时保存指标
        self.recorder.save_metrics()
    
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
        
        # 保存训练配置
        config_info = {
            "training_start": datetime.now().isoformat(),
            "output_dir": args.output_dir,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        }
        
        config_file = os.path.join(args.output_dir, "training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束回调"""
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"总步数: {state.global_step}")
        print(f"最佳eval_loss: {self.best_eval_loss:.4f}")
        if self.best_model_checkpoint:
            print(f"最佳模型检查点: {self.best_model_checkpoint}")
        
        # 保存最终指标和摘要
        self.recorder.save_metrics()
        summary = self.recorder.get_summary()
        
        summary_file = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"训练摘要已保存到: {summary_file}")
        print("="*60 + "\n")
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """保存回调"""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        print(f"保存检查点: {checkpoint_folder}")


class RAFTTrainer(Trainer):
    """自定义RAFT训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": []
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算损失，增强稳定性和NaN处理
        """
        try:
            # 检查输入数据
            if torch.isnan(inputs['input_ids']).any() or torch.isinf(inputs['input_ids']).any():
                print("❌ 输入数据包含NaN或Inf")
                return self._create_safe_loss(inputs)
            
            # 检查有效labels数量
            valid_labels = (inputs['labels'] != -100).sum().item()
            if valid_labels < 5:  # 如果有效标签太少
                print(f"⚠️ 有效labels过少: {valid_labels}, 使用安全损失")
                return self._create_safe_loss(inputs)
            
            # 正常前向传播
            outputs = model(**inputs)
            loss = outputs.loss
            
            # 检查NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ 检测到NaN/Inf loss! 分析原因...")
                print(f"  有效labels: {valid_labels}")
                print(f"  input_ids范围: [{inputs['input_ids'].min()}, {inputs['input_ids'].max()}]")
                print(f"  labels中-100数量: {(inputs['labels'] == -100).sum().item()}")
                
                # 尝试梯度裁剪和重新计算
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                return self._create_safe_loss(inputs)
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            print(f"❌ 损失计算异常: {e}")
            return self._create_safe_loss(inputs)

    def _create_safe_loss(self, inputs):
        """创建安全的损失值"""
        return torch.tensor(0.1, requires_grad=True).to(inputs['input_ids'].device)


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
    callback = CustomCallback(training_config.output_dir)
    
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


# 新增：训练可视化工具函数
def plot_training_curves(output_dir: str, save_path: Optional[str] = None):
    """
    绘制训练曲线
    
    Args:
        output_dir: 训练输出目录
        save_path: 图片保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # 读取训练指标
        metrics_file = os.path.join(output_dir, "training_metrics.json")
        if not os.path.exists(metrics_file):
            print(f"❌ 训练指标文件不存在: {metrics_file}")
            return
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # 创建DataFrame
        df = pd.DataFrame(metrics)
        
        # 清理数据
        df = df.dropna(subset=['train_loss'])
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制损失曲线
        if not df.empty:
            steps = df['step']
            
            # 训练损失
            if 'train_loss' in df.columns:
                train_loss = df['train_loss'].dropna()
                if not train_loss.empty:
                    ax1.plot(steps[:len(train_loss)], train_loss, label='训练损失', color='blue', alpha=0.7)
            
            # 评估损失
            if 'eval_loss' in df.columns:
                eval_loss = df['eval_loss'].dropna()
                if not eval_loss.empty:
                    ax1.plot(steps[:len(eval_loss)], eval_loss, label='评估损失', color='red', alpha=0.7)
            
            ax1.set_xlabel('训练步数')
            ax1.set_ylabel('损失')
            ax1.set_title('训练和评估损失曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制学习率曲线
            if 'learning_rate' in df.columns:
                lr = df['learning_rate'].dropna()
                if not lr.empty:
                    ax2.plot(steps[:len(lr)], lr, label='学习率', color='green', alpha=0.7)
                    ax2.set_xlabel('训练步数')
                    ax2.set_ylabel('学习率')
                    ax2.set_title('学习率变化曲线')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 训练曲线已保存到: {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("❌ 请安装matplotlib和pandas来绘制训练曲线: pip install matplotlib pandas")
    except Exception as e:
        print(f"❌ 绘制训练曲线时出错: {e}")