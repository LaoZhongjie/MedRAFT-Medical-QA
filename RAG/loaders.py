import os
import json
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


@dataclass
class LoadResult:
    documents: List[Document]
    files_count: int


def _detect_lang(text: str) -> str:
    # 非严格语言检测：包含汉字则认为中文，否则英文
    for ch in text[:512]:
        if '\u4e00' <= ch <= '\u9fff':
            return "zh"
    return "en"


def load_json_file(path: str) -> List[Document]:
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 尝试去除尾随逗号或不规范JSON（可根据需要增强）
            raise
    if not isinstance(data, list):
        return []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        disease_name = item.get("disease_name")
        category = item.get("category")
        information = item.get("information")
        abstract = item.get("abstract")
        source = item.get("source")

        base_meta = {
            "file_path": path,
            "file_name": os.path.basename(path),
            "disease_name": disease_name,
            "category": category,
            "source": source,
            "index": idx,
            "type": "json",
        }

        if information and isinstance(information, str):
            lang = _detect_lang(information)
            docs.append(
                Document(
                    page_content=information.strip(),
                    metadata={**base_meta, "section": "information", "doc_lang": lang},
                )
            )
        if abstract and isinstance(abstract, str):
            lang = _detect_lang(abstract)
            docs.append(
                Document(
                    page_content=abstract.strip(),
                    metadata={**base_meta, "section": "abstract", "doc_lang": lang},
                )
            )
    return docs


def load_pdf_file(path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(path)
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata.update({
                "file_path": path,
                "file_name": os.path.basename(path),
                "type": "pdf",
                "doc_lang": _detect_lang(d.page_content),
            })
        return pdf_docs
    except Exception:
        return []


def load_txt_file(path: str) -> List[Document]:
    try:
        loader = TextLoader(path, encoding="utf-8")
        txt_docs = loader.load()
        for d in txt_docs:
            d.metadata.update({
                "file_path": path,
                "file_name": os.path.basename(path),
                "type": "txt",
                "doc_lang": _detect_lang(d.page_content),
            })
        return txt_docs
    except Exception:
        return []


def load_documents(input_dir: str, exts: Optional[List[str]] = None) -> LoadResult:
    if exts is None:
        exts = [".json", ".pdf", ".txt"]
    docs: List[Document] = []
    files_count = 0

    for root, _, files in os.walk(input_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            path = os.path.join(root, fn)
            files_count += 1
            if ext == ".json":
                docs.extend(load_json_file(path))
            elif ext == ".pdf":
                docs.extend(load_pdf_file(path))
            elif ext == ".txt":
                docs.extend(load_txt_file(path))

    return LoadResult(documents=docs, files_count=files_count)