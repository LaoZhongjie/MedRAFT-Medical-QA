import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from loaders import load_json_file, load_pdf_file, load_txt_file
from splitter import split_documents
from embedder import build_embeddings
from rag_chain import load_vectorstore, make_rag_chain

# 初始化环境
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
HF_ENDPOINT = os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_HUB_ENDPOINT")
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = HF_ENDPOINT

st.set_page_config(page_title="Medical RAG QA System", layout="wide")
st.title("Medical RAG QA System — Streamlit")

# Sidebar 设置
with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Answer language", options=["zh", "en"], index=0)
    top_k = st.slider("Retriever top-k", min_value=3, max_value=15, value=5, step=1)
    persist_dir = st.text_input("Chroma persist directory", value=CHROMA_DIR)
    reset = st.checkbox("Rebuild vector store (clear then rebuild)", value=False)
    if st.button("Load/Initialize vector store"):
        embeddings = build_embeddings()
        if reset and os.path.exists(persist_dir):
            for root, dirs, files in os.walk(persist_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
        st.session_state["vs"] = load_vectorstore(persist_dir, embeddings)
        st.success("Vector store loaded")

# 上传组件
st.subheader("Upload files and ingest")
uploaded_files = st.file_uploader(
    "Upload JSON / PDF / TXT files (multi-select)", accept_multiple_files=True, type=["json", "pdf", "txt"]
)

if uploaded_files:
    docs: List[Document] = []
    for uf in uploaded_files:
        suffix = os.path.splitext(uf.name)[1].lower()
        # 保存到临时目录，再用原加载器读取，确保 metadata 完整
        tmp_dir = os.path.join("./tmp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, uf.name)
        with open(tmp_path, "wb") as fout:
            fout.write(uf.getbuffer())
        if suffix == ".json":
            try:
                docs.extend(load_json_file(tmp_path))
            except Exception as e:
                st.warning(f"{uf.name} failed to parse, skipped: {e}")
        elif suffix == ".pdf":
            docs.extend(load_pdf_file(tmp_path))
        elif suffix == ".txt":
            docs.extend(load_txt_file(tmp_path))
    if docs:
        chunks = split_documents(docs)
        embeddings = build_embeddings()
        # 增量入库：如果已有 vs 则添加；否则创建
        if "vs" in st.session_state and isinstance(st.session_state["vs"], Chroma):
            st.session_state["vs"].add_documents(chunks)
            st.session_state["vs"].persist()
        else:
            vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
            vs.persist()
            st.session_state["vs"] = vs
        st.success(f"Ingested {len(chunks)} chunks")

# 知识库资料名
st.subheader("Knowledge base file names")
if st.button("Show file names"):
    if "vs" not in st.session_state:
        st.error("Please load/initialize the vector store or ingest files first")
    else:
        try:
            col = st.session_state["vs"]._collection
            res = col.get(include=["metadatas"])  # 仅取元数据
            metas = res.get("metadatas", [])
            # 展平可能的嵌套列表
            flat_metas = []
            for m in metas:
                if isinstance(m, list):
                    flat_metas.extend([mm for mm in m if isinstance(mm, dict)])
                elif isinstance(m, dict):
                    flat_metas.append(m)
            from collections import Counter
            fn_counter = Counter()
            for m in flat_metas:
                fn = m.get("file_name")
                if fn:
                    fn_counter[fn] += 1
            if fn_counter:
                try:
                    import pandas as pd
                    df = pd.DataFrame({
                        "File name": list(fn_counter.keys()),
                        "Chunk count": list(fn_counter.values()),
                    }).sort_values(by="File name")
                    st.table(df)
                except Exception:
                    # 无法使用 pandas 时退化为文本列表
                    for fn, cnt in sorted(fn_counter.items(), key=lambda x: x[0]):
                        st.write(f"- {fn} (chunks {cnt})")
            else:
                st.info("Knowledge base is empty or lacks file name metadata")
        except Exception as e:
            st.error(f"Failed to read file names: {e}")

# 问答
st.subheader("Ask a question")
query = st.text_input("Enter your question:")

if st.button("Retrieve and generate answer"):
    if "vs" not in st.session_state:
        st.error("Please load/initialize the vector store or ingest files first")
    elif not query.strip():
        st.warning("Please enter a question")
    else:
        chain = make_rag_chain(st.session_state["vs"], query_lang=lang, top_k=top_k)
        prompt_str = chain.invoke(query)
        st.text_area("Generated prompt", prompt_str, height=200)
        # 调用千问生成
        from qwen_llm import qwen_complete
        answer = qwen_complete(prompt_str)
        st.markdown("**Answer**")
        st.write(answer)

st.subheader("Teacher Model: Generate Training Data")
teacher_q = st.text_input("Training sample question (teacher model)")
teacher_top_k = st.slider("Retriever top-k (teacher)", min_value=3, max_value=15, value=5, step=1, key="teacher_topk")
if st.button("Generate teacher sample"):
    if "vs" not in st.session_state:
        st.error("Please load/initialize the vector store or ingest files first")
    elif not teacher_q.strip():
        st.warning("Please enter a training sample question")
    else:
        from rag_chain import make_teacher_messages
        from qwen_llm import qwen_chat
        msgs = make_teacher_messages(st.session_state["vs"], teacher_q, query_lang=lang, top_k=teacher_top_k)
        system_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        st.text_area("System prompt", system_msg, height=150)
        st.text_area("User prompt", user_msg, height=300)
        # 调用教师模型
        teacher_ans = qwen_chat(system_msg, user_msg, temperature=0.3)
        st.markdown("**Teacher model answer**")
        st.write(teacher_ans)
        # 组合 JSONL 一条样本
        sample = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": teacher_ans},
            ]
        }
        st.markdown("**Training sample (JSON)**")
        st.code(str(sample), language="json")