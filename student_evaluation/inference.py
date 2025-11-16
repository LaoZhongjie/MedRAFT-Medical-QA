"""
推理模块
用于生成结构化回答
"""
import torch
from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from config import InferenceConfig
import textwrap

class RAFTInference:
    """RAFT推理器"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        inference_config: InferenceConfig
    ):
        """
        初始化推理器
        
        Args:
            model: 训练好的模型
            tokenizer: 分词器
            inference_config: 推理配置
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = inference_config
        
        # 设置为评估模式
        self.model.eval()
        
        # 获取设备
        self.device = next(model.parameters()).device
    
    def build_prompt(self, question: str, documents: List[Dict[str, str]]) -> str:
        """
        构建推理prompt
        
        Args:
            question: 用户问题
            documents: 参考文档列表
            
        Returns:
            格式化的prompt
        """
        # 合并文档
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            source = doc.get('source', '')
            doc_texts.append(f"[文档{i}]\n{content}-来源{source}")
        
        combined_docs = "\n\n".join(doc_texts)
        
        # 构建完整prompt
        prompt = textwrap.dedent(f"""
            请基于以下参考文档回答问题,并按照指定格式输出。

            参考文档:
            {combined_docs}

            问题: {question}

            请按以下格式输出你的回答:
            - 问题: [重复用户问题]
            - 假设/已知信息: [列出从文档中提取的相关信息]
            - CoT推理: [逐步推理过程，可以包含多个步骤]
            - 初步诊断建议(含不确定度): [给出建议及置信度]
            - 证据引用: [引用支持你结论的文档片段+来源]
            - 不足信息与后续建议: [指出缺失的信息]
        """)
        return prompt
    
    @torch.no_grad()
    def generate(
        self,
        question: str,
        documents: List[Dict[str, str]],
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        repetition_penalty: float = None,
        do_sample: bool = None
    ) -> str:
        """
        生成回答
        
        Args:
            question: 问题
            documents: 参考文档
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            
        Returns:
            生成的回答
        """
        # 使用默认配置(如果未提供)
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        # 构建prompt
        prompt = self.build_prompt(question, documents)
        
        # 构建消息
        messages = [
            {"role": "system", "content": "你是一位专业的医疗顾问助手,擅长结构化推理和医疗建议。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # 生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 解码
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def batch_generate(
        self,
        samples: List[Dict[str, Any]],
        **generate_kwargs
    ) -> List[str]:
        """
        批量生成
        
        Args:
            samples: 样本列表,每个包含question和documents
            **generate_kwargs: 生成参数
            
        Returns:
            生成的回答列表
        """
        responses = []
        for sample in samples:
            question = sample['question']
            documents = sample['documents']
            response = self.generate(question, documents, **generate_kwargs)
            responses.append(response)
        
        return responses
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """
        评估单个样本
        
        Args:
            sample: 包含question, documents, teacher_answer的样本
            
        Returns:
            包含student_answer和teacher_answer的字典
        """
        question = sample['question']
        documents = sample['documents']
        teacher_answer = sample.get('teacher_answer', '')
        
        student_answer = self.generate(question, documents)
        
        return {
            'question': question,
            'student_answer': student_answer,
            'teacher_answer': teacher_answer
        }


def load_and_test_model(
    model_path: str,
    test_sample: Dict[str, Any],
    model_config,
    inference_config: InferenceConfig
):
    """
    加载模型并测试
    
    Args:
        model_path: 模型路径
        test_sample: 测试样本
        model_config: 模型配置
        inference_config: 推理配置
    """
    from model import load_trained_model
    
    print("="*60)
    print("开始推理测试")
    print("="*60)
    
    # 加载模型
    model, tokenizer = load_trained_model(model_path, model_config)
    
    # 创建推理器
    inferencer = RAFTInference(model, tokenizer, inference_config)
    
    # 生成回答
    print("\n问题:", test_sample['question'])
    print("\n参考文档:")
    for i, doc in enumerate(test_sample['documents'], 1):
        print(f"  [文档{i}] {doc['content'][:100]}...")
    
    print("\n生成中...")
    response = inferencer.generate(
        test_sample['question'],
        test_sample['documents']
    )
    
    cleaned_response = clean_response(response)

    print("\n" + "="*60)
    print("学生模型生成的回答:")
    print("="*60)
    print(cleaned_response)
    
    if 'teacher_answer' in test_sample:
        print("\n" + "="*60)
        print("教师模型的回答(参考):")
        print("="*60)
        print(test_sample['teacher_answer'])
    
    print("\n" + "="*60)
    print("推理测试完成")
    print("="*60)

import re

def clean_response(text):
    import re
    # 删除各种形式的 “I don't know” （大小写、标点等）
    text = re.sub(r"\bI\s*don'?t\s*know\b[\.\!\,]*", "", text, flags=re.IGNORECASE)
    # 删除“我不知道”字样（包含前后可能的空格或标点）
    text = re.sub(r"[，。,.！!]*\s*我不知道\s*[，。,.！!]*", "", text)
    return text
