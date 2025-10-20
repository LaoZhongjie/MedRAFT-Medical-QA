# huatuo26m-lite 数据处理与RAG示例（独立目录）

本目录提供独立的数据预处理与基于 ChromaDB 的向量库构建示例，避免与主工程文件冲突。

## 结构
```
contrib/huatuo26m-lite/
├── README.md
├── requirements.txt
└── src/
    ├── config.py
    ├── 01_data_preprocess.py
    └── 02_build_vector_db.py
```

## 使用
- 安装依赖：`pip install -r contrib/huatuo26m-lite/requirements.txt`
- 预处理：
  ```bash
  python contrib/huatuo26m-lite/src/01_data_preprocess.py --chunk_size 500 --chunk_overlap 100
  ```
- 构建向量库与检索：
  ```bash
  python contrib/huatuo26m-lite/src/02_build_vector_db.py --collection huatuo26m-lite --query "糖尿病有什么症状"
  ```

## 说明
- 所有数据与向量库文件均写入到 `contrib/huatuo26m-lite/` 目录下的子目录中，不影响主工程。