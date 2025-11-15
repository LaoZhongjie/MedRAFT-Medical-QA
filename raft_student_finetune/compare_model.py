"""
compare_models.py
随机从 teacher_dataset.json 中抽取 100 条样本，
比较基础模型与微调模型的回答，并保存结果到 JSON 文件。
"""

import json
import random
import os
from datetime import datetime
from inference import RAFTInference
from model import load_trained_model, load_base_model
from config import ModelConfig, InferenceConfig

import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

def load_random_samples(json_path: str, sample_size: int = 100):
    """从数据集中随机抽取样本"""
    print(f"📂 正在加载数据集: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("teacher_dataset.json 必须是一个包含多个样本的列表")

    if len(data) < sample_size:
        print(f"⚠️ 数据集样本数 ({len(data)}) 少于 {sample_size}，将全部使用。")
        sample_size = len(data)

    samples = random.sample(data, sample_size)
    print(f"✅ 已随机抽取 {sample_size} 条样本进行对比测试\n")
    return samples


from tqdm import tqdm

def compare_models(base_model_path, fine_tuned_path, test_cases):
    """对比基础模型和微调模型输出，带进度条"""
    
    # 加载基础模型
    print("🔧 加载基础模型...")
    base_model_config = ModelConfig(model_name_or_path=base_model_path)
    base_model, base_tokenizer = load_base_model(base_model_path, base_model_config)
    
    # 加载微调模型
    print("🔧 加载微调模型...")
    tuned_model_config = ModelConfig(model_name_or_path=fine_tuned_path)
    tuned_model, tuned_tokenizer = load_trained_model(fine_tuned_path, tuned_model_config)
    
    # 创建推理器
    inference_config = InferenceConfig(max_new_tokens=1024, temperature=0.7, top_p=0.9)
    base_inferencer = RAFTInference(base_model, base_tokenizer, inference_config)
    tuned_inferencer = RAFTInference(tuned_model, tuned_tokenizer, inference_config)
    
    results = []
    
    # tqdm 进度条
    for i, case in enumerate(tqdm(test_cases, desc="🔄 处理测试样本", unit="sample"), 1):
        print(f"\n{'='*60}")
        print(f"🧠 测试样本 {i}/{len(test_cases)}: {case['question'][:80]}...")
        print(f"{'='*60}")
        
        # 基础模型回答
        print("🤖 基础模型回答中...")
        base_resp = base_inferencer.generate(case['question'], case['documents'])
        # print(base_resp)
        
        # 微调模型回答
        print("🎯 微调模型回答中...")
        tuned_resp = tuned_inferencer.generate(case['question'], case['documents'])
        # print(tuned_resp)
        
        result_item = {
            "id": case.get("id", f"sample_{i}"),
            "question": case["question"],
            "base_response": base_resp,
            "tuned_response": tuned_resp,
            "teacher_answer": case.get("teacher_answer", ""),
            "documents": case.get("documents", [])
        }
        results.append(result_item)
    
    return results


def save_results(results, output_dir="results"):
    """保存对比结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"compare_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 对比结果已保存至: {output_path}")
    return output_path


if __name__ == "__main__":
    dataset_path = "teacher_dataset.json"
    base_model_path = "Qwen/Qwen2.5-7B-Instruct"
    fine_tuned_path = "./output/final_model"

    # 1. 加载随机样本
    test_samples = load_random_samples(dataset_path, sample_size=50)

    # 2. 对比推理
    results = compare_models(base_model_path, fine_tuned_path, test_samples)

    # 3. 保存结果
    save_results(results)
