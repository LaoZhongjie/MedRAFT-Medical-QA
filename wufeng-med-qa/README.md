# 医学资料疾病问答系统（RAG）

一个仿照 FranzLiszt-1847/LLM 项目搭建的本地 RAG 问答系统，支持以 PDF 或问答对表格（CSV/XLSX）导入医学资料，完成检索与生成并附带引用。

## 功能特性
- 支持上传 `PDF` 与 `CSV/XLSX` 问答表格（列名支持 `question/answer`, `Q/A`, `问题/回答`）
- 文档拆分与向量化（`sentence-transformers/all-MiniLM-L6-v2`）
- 向量检索（FAISS，内积+单位向量=余弦相似度）
- 生成回答与引用（优先使用 OpenAI；无 Key 时走离线抽取回退）
- 持久化存储（`storage/faiss.index` 与 `storage/store.pkl`）

## 快速开始
1. 准备环境：建议 Python 3.10+
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行应用：
   ```bash
   streamlit run app.py
   ```
4. 使用方式：
   - 在侧边栏上传 PDF 或 CSV/XLSX 并点击“添加到知识库”
   - 在主界面输入问题，获取答案与引用出处（文件名与页码/行号）

## OpenAI 可选配置
- 若你有 `OPENAI_API_KEY`，将其加入环境变量即可：
  ```bash
  export OPENAI_API_KEY=你的key
  ```
- 有 Key 时使用 `gpt-4o-mini` 生成；否则使用离线抽取型回答（仅整合检索片段，不进行自由生成）。

## 数据与存储
- 上传文件会保存至 `data/uploads/` 目录
- 向量索引与文本、元数据映射保存至 `storage/` 目录
- 可在侧边栏清空知识库（删除持久化文件）

## 问答表格格式说明
- CSV/XLSX 支持以下任一列名组合：
  - `question` + `answer`
  - `Q` + `A`
  - `问题` + `回答`
- 每行将被转换为一个知识片段：
  ```
  问：<question>
  答：<answer>
  ```

## 免责声明
- 本系统仅用于医学资料的检索与信息提供，不能替代专业医疗建议与诊断。如遇紧急或严重情况，请及时就医并遵循专业医嘱。