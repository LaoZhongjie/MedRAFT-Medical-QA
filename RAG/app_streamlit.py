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

st.set_page_config(page_title="医学RAG问答系统", layout="wide")
st.title("医学RAG问答系统 — Streamlit")

# Sidebar 设置
with st.sidebar:
    st.header("设置")
    lang = st.selectbox("回答语言", options=["zh", "en"], index=0)
    top_k = st.slider("检索条数 (top-k)", min_value=3, max_value=15, value=5, step=1)
    persist_dir = st.text_input("Chroma 持久化目录", value=CHROMA_DIR)
    reset = st.checkbox("重建向量库 (清空后再建)", value=False)
    if st.button("加载/初始化向量库"):
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
        st.success("向量库已加载")

# 上传组件
st.subheader("上传资料并入库")
uploaded_files = st.file_uploader(
    "支持上传 JSON / PDF / TXT 文件 (可多选)", accept_multiple_files=True, type=["json", "pdf", "txt"]
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
            docs.extend(load_json_file(tmp_path))
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
        st.success(f"已入库 {len(chunks)} 个切片")

# 知识库资料名
st.subheader("知识库已有资料名")
if st.button("显示资料名"):
    if "vs" not in st.session_state:
        st.error("请先加载/初始化向量库或上传资料入库")
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
                        "文件名": list(fn_counter.keys()),
                        "切片数": list(fn_counter.values()),
                    }).sort_values(by="文件名")
                    st.table(df)
                except Exception:
                    # 无法使用 pandas 时退化为文本列表
                    for fn, cnt in sorted(fn_counter.items(), key=lambda x: x[0]):
                        st.write(f"- {fn}（切片数 {cnt}）")
            else:
                st.info("知识库为空或未包含文件名元数据")
        except Exception as e:
            st.error(f"读取资料名失败：{e}")

# 问答
st.subheader("提问")
query = st.text_input("请输入你的问题：")

if st.button("检索并生成回答"):
    if "vs" not in st.session_state:
        st.error("请先加载/初始化向量库或上传资料入库")
    elif not query.strip():
        st.warning("请输入问题")
    else:
        chain = make_rag_chain(st.session_state["vs"], query_lang=lang, top_k=top_k)
        prompt_str = chain.invoke(query)
        st.text_area("生成 Prompt", prompt_str, height=200)
        # 调用千问生成
        from qwen_llm import qwen_complete
        answer = qwen_complete(prompt_str)
        st.markdown("**回答**")
        st.write(answer)