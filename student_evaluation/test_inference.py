"""
推理测试脚本 (简化版)
随机抽取 teacher_dataset.json 中的 100 条样本进行推理测试
"""

import json
import random
from config import ModelConfig, InferenceConfig
from inference import load_and_test_model


def load_random_samples(json_path: str, sample_size: int = 100):
    """从 JSON 文件中随机抽取样本"""
    print(f"正在加载数据集: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("teacher_dataset.json 必须是包含多个样本的列表结构")

    if len(data) < sample_size:
        print(f"⚠️ 数据集样本数 ({len(data)}) 少于 {sample_size}，将全部使用。")
        sample_size = len(data)

    samples = random.sample(data, sample_size)
    print(f"✅ 已随机抽取 {sample_size} 条样本进行测试\n")
    return samples


def main():
    """主函数"""
    # 固定文件路径
    dataset_path = "teacher_dataset.json"
    model_path = "output/final_model"  # 你自己的模型路径
    base_model = "Qwen/Qwen2.5-7B-Instruct"

    print("\n" + "="*60)
    print("RAFT 模型批量推理测试")
    print("="*60 + "\n")

    # 创建配置
    model_config = ModelConfig(
        model_name_or_path=base_model,
        torch_dtype='bfloat16'
    )

    inference_config = InferenceConfig(
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9
    )

    # 加载并抽样
    test_samples = load_random_samples(dataset_path, sample_size=2)

    # 执行推理测试
    for i, sample in enumerate(test_samples, 1):
        print(f"\n🧠 测试样本 {i}/{len(test_samples)} - 问题: {sample['question']}")
        load_and_test_model(
            model_path=model_path,
            test_sample=sample,
            model_config=model_config,
            inference_config=inference_config
        )


if __name__ == "__main__":
    main()
