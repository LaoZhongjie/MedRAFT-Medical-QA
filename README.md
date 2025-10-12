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