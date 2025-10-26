# 医学RAG问答系统

一个支持 PDF / TXT / JSON 混合资料的中英文 RAG 系统：
- 预处理与切分：tiktoken 分词驱动、按 token 预算切分
- 向量化：bge-m3 在 GPU（或 CPU）生成嵌入
- 检索：Chroma 持久化向量库
- 生成：接入阿里通义千问 3 API 作为 LLM
- 接口：CLI + FastAPI + Streamlit 可视化界面

## 快速开始

1) 创建并填充 `.env`
```
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY 等
```

2) 安装依赖（建议 Python 3.10+，可用虚拟环境）
```
pip install -r requirements.txt
```

3) 预处理与入库
```
python ingest.py --input ./data --persist_dir ./chroma_store
```
默认会扫描 `./data` 下的 pdf/txt/json 文件，将切分后的文档嵌入入库。

4) 运行服务（FastAPI）
```
uvicorn app:app --reload --port 8000
```

5) CLI 查询示例
```
python query_cli.py -q "偏头痛的患病率与DALYs" -k 5
python query_cli.py -q "Type 2 Diabetes complications" -k 4 -l en
```

## 可视化界面（Streamlit）

启动：
```
streamlit run app_streamlit.py
# 可选端口：streamlit run app_streamlit.py --server.port 8501
```

界面功能概览：
- 侧边栏设置：选择回答语言（`zh`/`en`）、检索条数 `top-k`、Chroma 持久化目录（默认 `CHROMA_DIR` 或 `./chroma_store`）、是否重建向量库；点击“加载/初始化向量库”。
- 上传资料入库：支持多文件上传（JSON / PDF / TXT）。系统自动切分并向量化后写入本地 Chroma 库。
- 知识库资料名：点击“显示资料名”可查看已入库文件及切片数（在可用时以表格展示）。
- 提问与生成：输入问题后点击“检索并生成回答”，先展示生成的 Prompt，再调用千问生成最终答案。

相关环境变量：
- `DASHSCOPE_API_KEY`（必填）：通义千问 API 密钥。
- `QWEN_MODEL`（可选，默认 `qwen2.5-7b-instruct`）：所用的千问模型。
- `CHROMA_DIR`（可选，默认 `./chroma_store`）：向量库持久化目录。
- `HF_ENDPOINT` 或 `HUGGINGFACE_HUB_ENDPOINT`（可选）：Hugging Face 端点，若设置会被 UI 自动同步。

## 目录结构建议
```
DSA2/
  ├── app.py            # FastAPI 服务
  ├── app_streamlit.py  # Streamlit 可视化界面
  ├── ingest.py         # 资料预处理与入库
  ├── query_cli.py      # 命令行查询
  ├── loaders.py        # PDF/TXT/JSON 加载
  ├── splitter.py       # tiktoken 切分
  ├── embedder.py       # bge-m3 嵌入
  ├── rag_chain.py      # RAG 检索-生成链
  ├── requirements.txt
  ├── .env.example
  ├── README.md
  └── data/             # 放原始资料文件
```

## 运行在 GPU
确保安装了支持 GPU 的 PyTorch，并在 `.env` 中设置 `DEVICE=cuda`。若仅 CPU 环境则设为 `cpu`。

## 千问 3 接入
使用 `dashscope` SDK，设置环境变量 `DASHSCOPE_API_KEY`。在 `qwen_llm.py` 中可通过环境变量 `QWEN_MODEL` 选择模型名（如 `qwen2.5-7b-instruct`，根据账号权限与可用性）。

## 语言支持
- bge-m3 与千问 3 均支持中英文；系统在检索时保留原文，并在生成时按查询语言输出。
