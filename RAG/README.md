# 医学RAG问答系统

一个支持 PDF / TXT / JSON 混合资料的中英文RAG系统：
- 预处理与切分：tiktoken 分词驱动、按 token 预算切分
- 向量化：bge-m3 在 GPU（或CPU）生成嵌入
- 检索：Chroma 持久化向量库
- 生成：接入阿里通义千问3 API 作为LLM
- 接口：CLI + FastAPI

## 快速开始

1) 创建并填充 `.env`
```
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY 等
```

2) 安装依赖（建议Python 3.10+，可用虚拟环境）
```
pip install -r requirements.txt
```

3) 预处理与入库
```
python ingest.py --input ./data --persist_dir ./chroma_store
```
默认会扫描 `./data` 下的 pdf/txt/json 文件，将切分后的文档嵌入入库。

4) 运行服务
```
uvicorn app:app --reload --port 8000
```

5) CLI 查询示例
```
python query_cli.py -q "偏头痛的患病率与DALYs" -k 5
python query_cli.py -q "Type 2 Diabetes complications" -k 4 -l en
```

## 目录结构建议
```
DSA2/
  ├── app.py            # FastAPI 服务
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

## 运行在GPU
确保安装了支持GPU的 PyTorch，并在 `.env` 中设置 `DEVICE=cuda`。若仅CPU环境则设为 `cpu`。

## 千问3接入
使用 `dashscope` SDK，设置环境变量 `DASHSCOPE_API_KEY`。在 `rag_chain.py` 中选择 Qwen-3 的模型名，如 `qwen2.5-7b-instruct`（根据账号权限与可用性）。

## 语言支持
- bge-m3 与千问3均支持中英文；系统默认保留原文并在生成时按查询语言输出。