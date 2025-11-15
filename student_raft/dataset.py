"""
数据集处理模块
负责加载、预处理RAFT数据集
"""
import json
import torch
from typing import Dict, List, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from config import DataConfig
import textwrap

class RAFTDataset(Dataset):
    """RAFT知识蒸馏数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        data_config: DataConfig,
        max_length: int = 4096
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            data_config: 数据配置
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.max_length = max_length
        self.normal_count = 0
        self.problem_count = 0
    
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        print(f"加载数据集: {len(self.raw_data)} 条样本")
    
    def __len__(self) -> int:
        return len(self.raw_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单条数据，过滤无效样本"""
        try:
            item = self.raw_data[idx]
            
            # 验证数据完整性
            if not all(key in item for key in ['question', 'documents', 'teacher_answer']):
                raise ValueError(f"样本 {idx} 缺少必要字段")
            
            if len(item['teacher_answer'].strip()) < 10:
                raise ValueError(f"样本 {idx} teacher_answer过短")
            
            # 构建输入prompt
            prompt = self._build_prompt(item)
            
            # 获取教师答案
            teacher_answer = item['teacher_answer']
            
            # 构建完整文本用于训练
            messages = [
                {"role": "system", "content": "你是一位专业的医疗顾问助手,擅长结构化推理和医疗建议。"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": teacher_answer}
            ]
            
            # 使用tokenizer的chat_template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # 编码
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            # 创建labels (仅计算assistant回答部分的loss)
            labels = self._create_labels(messages, input_ids)
            
            # 验证labels有效性
            valid_labels_count = sum(1 for x in labels if x != -100)
            
            # 🔴 关键修改：如果有效labels太少，跳过这个样本
            if valid_labels_count < 100:
                self.problem_count += 1
                print(f"❌ 跳过样本 {idx}: 有效labels={valid_labels_count} (第{self.problem_count}个问题样本)")
                raise ValueError(f"有效labels不足: {valid_labels_count}")
            
            # 只有正常样本才会执行到这里
            self.normal_count += 1
            if self.normal_count <= 10:  # 只记录前10个正常样本
                print(f"✅ 正常样本 {idx}: 长度={len(input_ids)}, 有效labels={valid_labels_count}")
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
            
        except Exception as e:
            # 重新抛出异常，让DataLoader跳过这个样本
            raise IndexError(f"跳过样本 {idx}: {e}")
    
    def _build_prompt(self, item: Dict[str, Any]) -> str:
        """
        构建输入prompt
        
        Args:
            item: 单条数据样本
            
        Returns:
            格式化的prompt字符串
        """
        question = item['question']
        documents = item['documents']
        
        # 合并所有文档内容
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            doc_texts.append(f"[文档{i}]{content}")
        
        combined_docs = "\n\n".join(doc_texts)
        
        # 构建完整prompt
        prompt = textwrap.dedent(f"""
            你是一个经验丰富的中文医学专家，基于下面的知识库文档，回答患者的医学问题。要求严格遵守下列格式与规则。

            知识库文档:
            {combined_docs}

            问题: {question}

            请按以下格式输出你的回答:
            - 问题: [重复用户问题]
            - 假设/已知信息: [列出从文档中提取的相关信息]
            - CoT推理: [逐步推理过程，可以包含多个步骤]
            - 初步诊断建议(含不确定度): [给出建议及置信度]
            - 证据引用: [引用支持你结论的文档片段+source]
            - 不足信息与后续建议: [指出缺失的信息]
            """)
        return prompt
    
    def _create_labels(self, messages: List[Dict], input_ids: List[int]) -> List[int]:
        """
        创建labels,只对assistant的回答计算loss
        """
        # 先全部设为-100(忽略)
        labels = [-100] * len(input_ids)
        
        try:
            # 找到assistant回答的内容
            assistant_content = messages[-1]['content']
            assistant_tokens = self.tokenizer.encode(
                assistant_content,
                add_special_tokens=False
            )
            
            # 在input_ids中查找assistant_tokens的位置
            found = False
            for i in range(len(input_ids) - len(assistant_tokens) + 1):
                if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    # 找到了,设置从匹配位置开始往后的所有labels
                    for j in range(i, len(input_ids)):
                        labels[j] = input_ids[j]
                    found = True
                    break
            
            # 如果没找到完全匹配，尝试部分匹配
            if not found and len(assistant_tokens) > 10:
                # 尝试匹配前20个token
                partial_tokens = assistant_tokens[:20]
                for i in range(len(input_ids) - len(partial_tokens) + 1):
                    if input_ids[i:i+len(partial_tokens)] == partial_tokens:
                        # 找到了部分匹配，设置从匹配位置开始往后的所有labels
                        for j in range(i, len(input_ids)):
                            labels[j] = input_ids[j]
                        found = True
                        break
            
            # 如果还是没找到，使用备用方案
            if not found:
                # 设置后1/2的token为有效（基于经验）
                start_pos = len(input_ids) // 2
                for i in range(start_pos, len(input_ids)):
                    labels[i] = input_ids[i]
                    
        except Exception as e:
            print(f"❌ labels创建异常: {e}")
            # 紧急备用：设置后半部分为有效
            start_pos = len(input_ids) // 2
            for i in range(start_pos, len(input_ids)):
                labels[i] = input_ids[i]
        
        return labels

def create_collate_fn(tokenizer: PreTrainedTokenizer, max_length: int):
    """
    创建数据整理函数
    
    Args:
        tokenizer: 分词器
        max_length: 最大长度
        
    Returns:
        collate函数
    """
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """整理batch数据"""
        # 获取batch中的最大长度
        max_len = min(
            max([len(item['input_ids']) for item in batch]),
            max_length
        )
        
        # Padding
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            
            if seq_len > max_len:
                # 截断
                input_ids.append(item['input_ids'][:max_len])
                attention_mask.append(item['attention_mask'][:max_len])
                labels.append(item['labels'][:max_len])
            else:
                # Padding
                pad_len = max_len - seq_len
                input_ids.append(
                    torch.cat([
                        item['input_ids'],
                        torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
                    ])
                )
                attention_mask.append(
                    torch.cat([
                        item['attention_mask'],
                        torch.zeros(pad_len, dtype=torch.long)
                    ])
                )
                labels.append(
                    torch.cat([
                        item['labels'],
                        torch.full((pad_len,), -100, dtype=torch.long)
                    ])
                )
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }
    
    return collate_fn


def split_dataset(dataset: RAFTDataset, validation_split: float = 0.2, seed: int = 42):
    """
    划分训练集和验证集
    
    Args:
        dataset: 完整数据集
        validation_split: 验证集比例
        seed: 随机种子
        
    Returns:
        (train_dataset, eval_dataset)
    """
    from torch.utils.data import random_split
    
    total_size = len(dataset)
    eval_size = int(total_size * validation_split)
    train_size = total_size - eval_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=generator
    )
    
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(eval_dataset)} 条")
    
    return train_dataset, eval_dataset


class FilteredDataset(Dataset):
    """过滤无效样本的数据集包装器"""
    
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.valid_indices = self._find_valid_indices()
    
    def _find_valid_indices(self):
        """找到所有有效样本的索引"""
        valid_indices = []
        print("扫描有效样本...")
        
        for idx in range(len(self.original_dataset)):
            try:
                # 尝试获取样本
                self.original_dataset[idx]
                valid_indices.append(idx)
            except Exception:
                # 跳过无效样本
                continue
        
        print(f"找到 {len(valid_indices)} 个有效样本")
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 映射到原始数据集的索引
        original_idx = self.valid_indices[idx]
        return self.original_dataset[original_idx]