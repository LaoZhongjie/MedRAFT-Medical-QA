# function _normalize_text (module-level helper)
import os
import json
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import re
import unicodedata


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
            # 尝试容错：去除BOM、移除尾随逗号，再次解析；失败则返回空
            f.seek(0)
            text = f.read()
            if text.startswith("\ufeff"):
                text = text.lstrip("\ufeff")
            import re
            text = re.sub(r",\s*([}\]])", r"\1", text)
            try:
                data = json.loads(text)
            except Exception:
                return []
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
            norm_info = _normalize_text(information)
            lang = _detect_lang(norm_info)
            docs.append(
                Document(
                    page_content=norm_info,
                    metadata={**base_meta, "section": "information", "doc_lang": lang},
                )
            )
        if abstract and isinstance(abstract, str):
            norm_abs = _normalize_text(abstract)
            lang = _detect_lang(norm_abs)
            docs.append(
                Document(
                    page_content=norm_abs,
                    metadata={**base_meta, "section": "abstract", "doc_lang": lang},
                )
            )
    return docs


def load_pdf_file(path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(path)
        pdf_docs = loader.load()
        for d in pdf_docs:
            content = _normalize_text(d.page_content)
            d.page_content = content
            d.metadata.update({
                "file_path": path,
                "file_name": os.path.basename(path),
                "type": "pdf",
                "doc_lang": _detect_lang(content),
            })
        return pdf_docs
    except Exception:
        return []


def load_txt_file(path: str) -> List[Document]:
    try:
        loader = TextLoader(path, encoding="utf-8")
        txt_docs = loader.load()
        for d in txt_docs:
            content = _normalize_text(d.page_content)
            d.page_content = content
            d.metadata.update({
                "file_path": path,
                "file_name": os.path.basename(path),
                "type": "txt",
                "doc_lang": _detect_lang(content),
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


def _normalize_text(text: str) -> str:
    s = text if isinstance(text, str) else ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\ufeff", "")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # 折叠重复标点（中英文）
    s = re.sub(r"[。]{2,}", "。", s)
    s = re.sub(r"[，]{2,}", "，", s)
    s = re.sub(r"[、]{2,}", "、", s)
    s = re.sub(r"[；]{2,}", "；", s)
    s = re.sub(r"[：]{2,}", "：", s)
    s = re.sub(r"[！]{2,}", "！", s)
    s = re.sub(r"[？]{2,}", "？", s)
    s = re.sub(r"([\.]{2,})", ".", s)
    s = re.sub(r"([,]{2,})", ",", s)
    s = re.sub(r"([;]{2,})", ";", s)
    s = re.sub(r"([:]{2,})", ":", s)
    s = re.sub(r"([!]{2,})", "!", s)
    s = re.sub(r"([?]{2,})", "?", s)
    # 去除中文标点前空格；英文标点后补空格
    s = re.sub(r"\s+([，。；：、！？])", r"\1", s)
    s = re.sub(r"([,;:])([^\s])", r"\1 \2", s)
    # 常见医学缩写扩展（保留缩写并添加中文全称）
    abbr_map = {
        "copd": "慢性阻塞性肺疾病",
        "htn": "高血压",
        "dm": "糖尿病",
        "t2dm": "2型糖尿病",
        "cad": "冠心病",
        "mi": "心肌梗死",
        "chf": "心力衰竭",
        "uti": "泌尿道感染",
        "gerd": "胃食管反流",
        "hbv": "乙型肝炎病毒",
        "hcv": "丙型肝炎病毒",
        "bp": "血压",
        "hr": "心率",
        "wbc": "白细胞",
        "rbc": "红细胞",
        "bmi": "体重指数",
    }
    def _expand(m: re.Match) -> str:
        ab = m.group(0)
        full = abbr_map.get(ab.lower())
        return f"{ab}（{full}）" if full else ab
    pattern = r"\b(" + "|".join(map(re.escape, abbr_map.keys())) + r")\b"
    s = re.sub(pattern, _expand, s)
    return s