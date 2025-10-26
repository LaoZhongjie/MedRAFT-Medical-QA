import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from embedder import build_embeddings


SYSTEM_PROMPT_ZH = (
    "你是医学RAG助手。请结合检索到的资料，以中文准确、简洁回答。"
    "若无依据，请说明无法确定；引用资料的疾病名称与来源。"
)
SYSTEM_PROMPT_EN = (
    "You are a medical RAG assistant. Answer accurately and concisely in English"
    " using the retrieved context. If uncertain, say so; cite disease names and sources."
)


def load_vectorstore(persist_directory: str = None, embeddings=None):
    persist_directory = persist_directory or os.getenv("CHROMA_DIR", "./chroma_store")
    embeddings = embeddings or build_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def _select_system_prompt(query_lang: str) -> str:
    return SYSTEM_PROMPT_ZH if query_lang == "zh" else SYSTEM_PROMPT_EN


def make_rag_chain(vs: Chroma, query_lang: str = "zh", top_k: int = 5):
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    def format_docs(docs: List[Document]) -> str:
        blocks = []
        for d in docs:
            meta = d.metadata or {}
            dn = meta.get("disease_name")
            src = meta.get("source")
            cat = meta.get("category")
            sec = meta.get("section")
            fp = meta.get("file_name")
            blocks.append(
                f"[{cat or '-'}] {dn or '-'} | {sec or '-'} | {fp or '-'}\n{d.page_content}\nSource: {src or '-'}"
            )
        return "\n\n".join(blocks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _select_system_prompt(query_lang)),
        ("user", "Query: {question}\n\nContext:\n{context}\n\nConstraints: answer in {lang}, keep citations inline."),
    ])

    def call_llm(inputs):
        return prompt.invoke(inputs).to_string()

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough(), "lang": RunnableLambda(lambda x: query_lang)}
        | RunnableLambda(call_llm)
        | StrOutputParser()
    )
    return chain