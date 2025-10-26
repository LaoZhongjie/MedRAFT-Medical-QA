import os
import re
import json
from typing import List, Dict, Any, Tuple

import pandas as pd
from pypdf import PdfReader

# Text cleaning tuned for bilingual CN/EN medical content
def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace: convert non-breaking/full-width spaces, strip
    text = text.replace("\u00A0", " ").replace("\u3000", " ")
    # Remove typical bullets and excessive punctuation spacing
    text = re.sub(r"[•●◆■◦]\s*", "", text)
    # Normalize multiple newlines and spaces
    text = re.sub(r"\s+", " ", text)
    # Trim
    return text.strip()

# CN/EN aware chunking that breaks at punctuation when possible
def chunk_text(text: str, max_len: int = 1000, overlap: int = 200) -> List[str]:
    text = _clean_text(text)
    if len(text) <= max_len:
        return [text]
    # Preferred split points: CN punctuation first, then EN sentence enders
    split_chars = "。！？；;.!?\n"
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        # Walk backward to nearest split char within a window
        window = text[start:end]
        idx = max(window.rfind(ch) for ch in split_chars) if window else -1
        if idx == -1 or idx < int(0.5 * len(window)):
            cut = end
        else:
            cut = start + idx + 1
        chunks.append(text[start:cut].strip())
        start = max(0, cut - overlap)
    # Merge last small tail
    merged: List[str] = []
    for c in chunks:
        if merged and len(merged[-1]) + len(c) < max_len * 0.6:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged

# PDF ingestion using pypdf per-page extraction
def pdf_to_chunks(file_path: str, source_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    reader = PdfReader(file_path)
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            content = page.extract_text() or ""
        except Exception:
            content = ""
        content = _clean_text(content)
        if not content:
            continue
        for ch in chunk_text(content, max_len=1000, overlap=200):
            texts.append(ch)
            metas.append({"source": source_name, "type": "pdf", "page": page_idx})
    return texts, metas

# QA table ingestion with bilingual column detection
QA_QUESTION_KEYS = ["Question", "问题", "题目", "问"]
QA_ANSWER_KEYS = ["Answer", "答案", "解答", "答"]

def qa_table_to_chunks(file_path: str, source_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    df = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)
    cols = [c.strip() for c in df.columns]
    q_col = next((c for c in cols if any(k.lower() in c.lower() for k in QA_QUESTION_KEYS)), None)
    a_col = next((c for c in cols if any(k.lower() in c.lower() for k in QA_ANSWER_KEYS)), None)
    if not q_col or not a_col:
        # Fallback: first two columns
        q_col, a_col = cols[0], cols[1] if len(cols) > 1 else (cols[0], cols[0])
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        q = _clean_text(str(row.get(q_col, "")))
        a = _clean_text(str(row.get(a_col, "")))
        content = f"Q: {q}\nA: {a}".strip()
        for ch in chunk_text(content, max_len=800, overlap=120):
            texts.append(ch)
            metas.append({"source": source_name, "type": "qa", "row": int(i)})
    return texts, metas

# JSON ingestion for bilingual disease entries
# Expects objects with keys like: disease_name, category, information (EN), abstract (CN), source

def json_to_chunks(file_path: str, source_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both dict and list
    items = data if isinstance(data, list) else data.get("items") or data.get("data") or []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        disease = str(item.get("disease_name") or item.get("name") or "").strip()
        category = str(item.get("category") or "").strip()
        source = str(item.get("source") or source_name).strip()
        cn = str(item.get("abstract") or item.get("summary_cn") or "").strip()
        en = str(item.get("information") or item.get("summary_en") or "").strip()
        # Build bilingual content blocks if present
        blocks: List[Tuple[str, str]] = []
        if cn:
            blocks.append(("cn", f"Disease: {disease}\nCategory: {category}\nAbstract: {cn}"))
        if en:
            blocks.append(("en", f"Disease: {disease}\nCategory: {category}\nInformation: {en}"))
        for lang, content in blocks:
            for ch in chunk_text(content, max_len=900, overlap=150):
                texts.append(ch)
                metas.append({
                    "source": source,
                    "type": "json",
                    "lang": lang,
                    "disease": disease,
                    "category": category,
                    "item_index": idx,
                })
    return texts, metas