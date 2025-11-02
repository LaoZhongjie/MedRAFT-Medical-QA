# 模块级新增：元数据加速检索与重排辅助函数
import os
from typing import List, Optional
import re

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
    # 替换为“加速检索 + 重排 + 格式化”
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
    def build_context(question: str) -> str:
        docs = _retrieve_docs_accel(vs, question, top_k, query_lang)
        return format_docs(docs)
    chain = (
        {"context": RunnableLambda(build_context), "question": RunnablePassthrough(), "lang": RunnableLambda(lambda _: query_lang)}
        | RunnableLambda(call_llm)
        | StrOutputParser()
    )
    return chain


def make_teacher_messages(vs: Chroma, question: str, query_lang: str = "zh", top_k: int = 5):
    # 使用加速检索
    try:
        docs = _retrieve_docs_accel(vs, question, top_k, query_lang)
    except Exception:
        docs = []
    if docs is None:
        docs = []
    if not isinstance(docs, list):
        docs = [docs]

    from langchain_core.documents import Document
    def format_teacher_docs(docs: List[Document]) -> str:
        lines = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            dn = meta.get("disease_name") or "-"
            cat = meta.get("category") or "-"
            sec = meta.get("section") or "-"
            fp = meta.get("file_name") or "-"
            src = meta.get("source") or "-"
            head = f"[Doc{i}] {dn} | {cat} | {sec} | {fp}"
            body = (d.page_content or "").strip()
            lines.append(f"{head}\n摘录:\n{body}\nSource: {src}")
        return "\n\n".join(lines) if lines else "(无检索到的文档)"

    knowledge_docs = format_teacher_docs(docs)

    system_prompt = (
        "你是一位经验丰富的中文医学专家，负责根据知识库文档回答患者问题。你的任务是提供清晰、逐步的推理（可被学生模型学习），引用文档证据，并给出初步诊断建议。请注意：所有回答仅供参考，不能替代面对面医生诊断。"
        if query_lang == "zh"
        else "You are an experienced medical expert. Provide clear, step-by-step reasoning with citations from the knowledge base. All answers are for reference only."
    )

    user_prompt = (
        "你是一个经验丰富的中文医学专家，基于以下知识库文档回答患者的医学问题。请严格遵守以下格式输出。\n\n"
        "输入：\n"
        f"- 问题: {question}\n"
        "- 知识库文档:\n"
        f"{knowledge_docs}\n\n"
        "输出格式（必须严格遵守）:\n"
        "- 问题: {原问题}\n"
        "- 假设/已知信息: {从问题中提取的已知病情要点}\n"
        "- CoT推理:\n"
        "  1) 症状分析: ...\n"
        "  2) 鉴别诊断: ...\n"
        "  3) 推荐检查: ...\n"
        "- 初步诊断建议（含不确定度）: ...\n"
        "- 证据引用: [Doc编号] + 段落摘录\n"
        "- 不足信息与后续建议: ... （若信息不足请以“I don’t know”开头并说明缺哪些关键信息）\n"
        "- 紧急就医指示（红旗症状）: ..."
        if query_lang == "zh"
        else (
            "You are an experienced medical expert. Based on the following knowledge base documents, answer the patient's question. Strictly follow the output format.\n\n"
            "Input:\n"
            f"- Question: {question}\n"
            "- Knowledge Docs:\n"
            f"{knowledge_docs}\n\n"
            "Output format (must follow):\n"
            "- Question: {original question}\n"
            "- Hypotheses/Known info: {key clinical points}\n"
            "- CoT reasoning:\n"
            "  1) Symptom analysis: ...\n"
            "  2) Differential diagnosis: ...\n"
            "  3) Recommended tests: ...\n"
            "- Initial diagnostic advice (with uncertainty): ...\n"
            "- Evidence: [Doc#] + paragraph excerpt\n"
            "- Missing info & next steps: ... (start with “I don’t know” if insufficient)\n"
            "- Red flags (urgent care): ..."
        )
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_keywords(text: str) -> List[str]:
    s = (text or "").strip()
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", s)
    en = re.findall(r"[A-Za-z]{3,}", s)
    # 去重保持顺序，限制最多6个关键词
    seen, out = set(), []
    for t in zh + en:
        if t not in seen:
            seen.add(t)
            out.append(t.lower())
        if len(out) >= 6:
            break
    return out

def _unique_key(d) -> str:
    m = d.metadata or {}
    return f"{m.get('file_name')}-{m.get('index')}-{m.get('section')}"

def _try_filtered_search(vs, query: str, k: int, keywords: List[str]) -> List:
    results = []
    # 先按 file_name 关键词 + abstract 过滤
    for kw in keywords:
        flt = {"section": "abstract", "file_name": {"$contains": kw}}
        try:
            results.extend(vs.similarity_search(query, k=min(3, k), filter=flt))
        except Exception:
            pass
    # 若不足，纯 abstract 过滤补充
    if len(results) < k:
        try:
            results.extend(vs.similarity_search(query, k=min(k * 2, max(k, 8)), filter={"section": "abstract"}))
        except Exception:
            pass
    # 若仍不足，再做一般检索补充
    if len(results) < k:
        try:
            results.extend(vs.similarity_search(query, k=max(k, 8)))
        except Exception:
            pass
    # 去重保持顺序
    dedup, seen = [], set()
    for d in results:
        key = _unique_key(d)
        if key not in seen:
            seen.add(key)
            dedup.append(d)
    return dedup

def _rerank_docs(query: str, docs: List, query_lang: str) -> List:
    kws = _extract_keywords(query)
    def score(d) -> float:
        m = d.metadata or {}
        sc = 0.0
        if m.get("section") == "abstract":
            sc += 1.25
        fn = (m.get("file_name") or "").lower()
        dn = (m.get("disease_name") or "").lower()
        if any(kw in fn for kw in kws):
            sc += 0.85
        if any(kw in dn for kw in kws):
            sc += 0.85
        if (m.get("doc_lang") or "").lower() == (query_lang or "zh").lower():
            sc += 0.3
        return sc
    # 稳定排序：按得分降序，分数相同保留原顺序
    return sorted(docs, key=lambda d: (-score(d)))
    
def _retrieve_docs_accel(vs, question: str, top_k: int, query_lang: str) -> List:
    kws = _extract_keywords(question)
    docs = _try_filtered_search(vs, question, top_k, kws)
    docs = _rerank_docs(question, docs, query_lang)
    return docs[:top_k]