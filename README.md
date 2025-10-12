# MedRAFT: Lightweight Medical QA System

This project implements **MedRAFT**, a Retrieval-Augmented Fine-Tuning (RAFT) framework for medical question answering in Chinese.

## Project Structure
- `src/` : Source code for preprocessing, vector DB building, teacher data generation, fine-tuning, and inference.
- `data_raw/` : Raw medical datasets.
- `data_processed/` : Cleaned and chunked text data.
- `outputs/` : Generated data and model checkpoints.

## Environment
Python 3.10  
PyTorch, HuggingFace Transformers, PEFT, ChromaDB, SentenceTransformers, OpenAI API, Gradio

## To Do
- [ ] Data preprocessing
- [ ] Build ChromaDB vector DB
- [ ] Generate teacher data (GPT-4)
- [ ] Train student model (QLoRA)
- [ ] Gradio interface

## Project Structure
```
MedRAFT-Medical-QA/
│
├── README.md                ← 📘 项目介绍（最重要）
├── requirements.txt         ← 📦 依赖列表
├── .gitignore               ← 🧹 忽略临时文件
│
├── data_raw/                ← 原始医学数据（可先建空文件夹）
│   ├── Huatuo_subset/       
│   └── webMedQA_subset/
│
├── data_processed/          ← 清洗 & 分块后的数据（chunk）
│
├── vector_db/               ← 存放ChromaDB索引（自动生成，不上传）
│
├── src/                     ← 核心代码目录
│   ├── __init__.py
│   ├── 01_data_preprocess.py      ← 清洗 & 分块
│   ├── 02_build_vector_db.py      ← 向量化 + 存入ChromaDB
│   ├── 03_teacher_data_gen.py     ← GPT-4生成教师数据
│   ├── 04_train_student_model.py  ← QLoRA微调代码
│   ├── 05_inference_demo.py       ← Gradio界面
│
├── notebooks/               ← 实验笔记（Colab、调试代码）
│   └── data_exploration.ipynb
│
├── outputs/                 ← 保存生成的教师数据、模型检查点
│   ├── teacher_data/
│   └── model_ckpts/
│
└── LICENSE                  ← 项目许可证（可选，MIT/Apache）
```
