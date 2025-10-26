import os
import io
import time
from typing import List, Dict

import numpy as np
import streamlit as st

from src.rag.embedding import Embedder
from src.rag.vector_store import FAISSStore
from src.rag.indexer import pdf_to_chunks, qa_table_to_chunks, json_to_chunks
from src.rag.generator import AnswerGenerator

# ---------------- UI -----------------
st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("Medical RAG Assistant")

# Sidebar: provider status
provider = os.environ.get("LLM_PROVIDER")
if provider == "qwen" and os.environ.get("DASHSCOPE_API_KEY"):
    st.sidebar.success("Generation model: Qwen (DashScope)")
elif provider == "openai" and os.environ.get("OPENAI_API_KEY"):
    st.sidebar.success("Generation model: OpenAI")
else:
    st.sidebar.warning("Generation model: Offline extractive")

# Initialize components
@st.cache_resource
def _init_embedder() -> Embedder:
    return Embedder(model_name=os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

@st.cache_resource
def _init_store(embed_dim: int) -> FAISSStore:
    store = FAISSStore(storage_dir="storage")
    store.load_or_create(embed_dim=embed_dim)
    return store

@st.cache_resource
def _init_generator() -> AnswerGenerator:
    return AnswerGenerator()

embedder = _init_embedder()
store = _init_store(embedder.dim)
generator = _init_generator()

# Uploaders
st.sidebar.header("Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, CSV/XLSX, JSON)",
    type=["pdf", "csv", "xlsx", "json"],
    accept_multiple_files=True,
)

add_btn = st.sidebar.button("Add to Knowledge Base")
clear_btn = st.sidebar.button("Clear Knowledge Base")

if clear_btn:
    store.clear()
    store.load_or_create(embed_dim=embedder.dim)
    st.sidebar.success("Knowledge base cleared.")

if add_btn and uploaded_files:
    # Progress UI
    status = st.empty()
    file_progress = st.progress(0)
    total_files = len(uploaded_files)

    new_texts: List[str] = []
    new_metas: List[Dict] = []

    for idx, uf in enumerate(uploaded_files, start=1):
        name = uf.name
        status.write(f"Indexing {idx}/{total_files}: {name}")
        suffix = os.path.splitext(name)[1].lower()
        tmp_path = os.path.join(st.session_state.get("tmp_dir", os.getcwd()), name)
        with open(tmp_path, "wb") as f:
            f.write(uf.read())
        if suffix == ".pdf":
            texts, metas = pdf_to_chunks(tmp_path, source_name=name)
        elif suffix in (".csv", ".xlsx"):
            texts, metas = qa_table_to_chunks(tmp_path, source_name=name)
        elif suffix == ".json":
            texts, metas = json_to_chunks(tmp_path, source_name=name)
        else:
            st.warning(f"Unsupported file type: {name}")
            texts, metas = [], []
        new_texts.extend(texts)
        new_metas.extend(metas)
        file_progress.progress(int(idx / total_files * 100))

    # Embedding progress (batching)
    if new_texts:
        status.write(f"Embedding {len(new_texts)} chunks...")
        embed_progress = st.progress(0)
        vectors_list = []
        batch_size = 64
        total_chunks = len(new_texts)
        for i in range(0, total_chunks, batch_size):
            batch = new_texts[i : i + batch_size]
            vecs_batch = embedder.embed_texts(batch)
            vectors_list.append(vecs_batch)
            done = min(i + batch_size, total_chunks)
            embed_progress.progress(int(done / total_chunks * 100))
        vecs = np.vstack(vectors_list)
        status.write("Saving to vector store...")
        store.add(vecs, new_texts, new_metas)
        status.write(f"Done. Added {len(new_texts)} chunks from {total_files} file(s).")
        st.sidebar.success(f"Added {len(new_texts)} chunks from {total_files} file(s).")
    else:
        status.write("No chunks were produced from the uploads.")
        st.sidebar.warning("No chunks were produced from the uploads.")

# Chat interface
if "history" not in st.session_state:
    st.session_state["history"] = []

with st.container():
    st.subheader("Ask a question")
    query = st.text_input("Enter your medical question in English or Chinese:")
    if st.button("Search") and query.strip():
        with st.spinner("Retrieving relevant passages..."):
            qvec = embedder.embed_query(query)
            results = store.search(qvec, k=6)
            contexts = [r["text"] for r in results]
            citations = [r["metadata"] for r in results]
        with st.spinner("Generating answer..."):
            answer = generator.generate(query, contexts)
        st.write(answer)
        st.markdown("---")
        st.subheader("Citations")
        for i, meta in enumerate(citations, start=1):
            src = meta.get("source")
            info = ", ".join(f"{k}: {v}" for k, v in meta.items() if k != "source")
            st.write(f"[{i}] {src} ({info})")

# Footer
st.sidebar.info("Tip: You can upload bilingual JSON disease data.")