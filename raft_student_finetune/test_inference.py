"""
推理测试脚本
用于测试训练好的模型
"""
import argparse
import json
from config import ModelConfig, InferenceConfig
from inference import load_and_test_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAFT模型推理测试")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--test_file', type=str, default=None,
                        help='测试数据文件(可选)')
    parser.add_argument('--question', type=str, default=None,
                        help='测试问题(可选,用于单个测试)')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='基础模型名称')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='温度参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p参数')
    
    return parser.parse_args()


def create_test_sample():
    """创建默认测试样本"""
    return {
        "question": "如果我漏服或多服了治疗2型糖尿病的药,该怎么办?",
        "documents": [
            {
                "content": "2型糖尿病药物治疗指南:如果漏服一次药物,不应在下次服药时加倍剂量。应按照正常时间服用下一次药物。如果多服药物,应密切监测血糖,必要时就医。低血糖症状包括头晕、出汗、心悸、手抖等。",
                "type": "oracle"
            },
            {
                "content": "糖尿病患者日常管理:糖尿病患者应规律服药、监测血糖、控制饮食、适量运动。定期复查血糖、糖化血红蛋白等指标。",
                "type": "oracle"
            },
            {
                "content": "高血压药物使用注意事项:高血压患者应规律服药,不可随意增减剂量。漏服后不要加倍补服。",
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
- 不足信息与后续建议: 需要了解具体药物类型、剂量、多服的时间和数量,以便给出更精确建议。"""
    }


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("RAFT 模型推理测试")
    print("="*60 + "\n")
    
    # 创建配置
    model_config = ModelConfig(
        model_name_or_path=args.base_model,
        torch_dtype='bfloat16'
    )
    
    inference_config = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # 准备测试样本
    if args.test_file:
        # 从文件加载
        print(f"从文件加载测试样本: {args.test_file}")
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        test_sample = test_data[0] if isinstance(test_data, list) else test_data
    elif args.question:
        # 使用命令行提供的问题
        test_sample = {
            "question": args.question,
            "documents": [
                {
                    "content": "这是一个示例文档。请根据实际情况提供相关文档内容。",
                    "type": "oracle"
                }
            ]
        }
        print("使用命令行提供的问题进行测试")
    else:
        # 使用默认测试样本
        test_sample = create_test_sample()
        print("使用默认测试样本")
    
    # 执行推理测试
    load_and_test_model(
        model_path=args.model_path,
        test_sample=test_sample,
        model_config=model_config,
        inference_config=inference_config
    )


if __name__ == '__main__':
    main()