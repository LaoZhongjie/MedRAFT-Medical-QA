import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from tqdm import tqdm

from config import (
    ensure_dirs,
    RAW_DATA_DIR,
    PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_Q_LEN,
    MIN_A_LEN,
)


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_json(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            for item in data.get("data", []):
                yield item


def read_csv(path: Path) -> Iterable[Dict]:
    df = pd.read_csv(path)
    for rec in df.to_dict(orient="records"):
        yield rec


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\t]", " ")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_record(item: Dict, idx: int) -> Dict:
    q = (
        item.get("instruction")
        or item.get("query")
        or item.get("question")
        or item.get("input")
        or ""
    )
    a = (
        item.get("output")
        or item.get("answer")
        or item.get("response")
        or item.get("completion")
        or ""
    )
    source = item.get("source") or "huatuo26m-lite"
    specialty = item.get("department") or item.get("topic") or None

    q = clean_text(str(q))
    a = clean_text(str(a))

    if len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
        return {}

    text = f"问题：{q}\n答案：{a}"
    return {
        "id": f"doc_{idx}",
        "source": source,
        "specialty": specialty,
        "question": q,
        "answer": a,
        "text": text,
    }


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = text or ""
    chunks = []
    if not text:
        return chunks
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def auto_locate_input(inp: str) -> Path:
    p = Path(inp) if inp else RAW_DATA_DIR
    if p.is_file():
        return p
    if p.is_dir():
        for ext in (".jsonl", ".json", ".csv"):
            files = list(p.glob(f"*{ext}"))
            if files:
                return files[0]
    raise FileNotFoundError(f"未找到原始数据文件: {p}")


def main():
    parser = argparse.ArgumentParser("huatuo26m-lite 数据清洗与切片（contrib）")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_DIR / "huatuo26m-lite.jsonl"))
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    ensure_dirs()
    input_path = auto_locate_input(args.input)

    if input_path.suffix == ".jsonl":
        it = read_jsonl(input_path)
    elif input_path.suffix == ".json":
        it = read_json(input_path)
    elif input_path.suffix == ".csv":
        it = read_csv(input_path)
    else:
        raise ValueError(f"不支持的文件类型: {input_path.suffix}")

    docs = []
    for i, item in tqdm(enumerate(it), desc="标准化", unit="条"):
        rec = normalize_record(item, i)
        if rec:
            docs.append(rec)

    if not docs:
        raise RuntimeError("无有效记录，请检查原始数据格式或长度过滤阈值！")

    df = pd.DataFrame(docs)
    docs_path = PROCESSED_DIR / "huatuo26m-lite_docs.parquet"
    df.to_parquet(docs_path, index=False)

    rows = []
    for rec in tqdm(docs, desc="切片", unit="文档"):
        pieces = chunk_text(rec["text"], args.chunk_size, args.chunk_overlap)
        for j, ch in enumerate(pieces):
            rows.append({
                "chunk_id": f"{rec['id']}-{j}",
                "doc_id": rec["id"],
                "text": ch,
                "source": rec["source"],
                "specialty": rec["specialty"],
            })

    chunks_df = pd.DataFrame(rows)
    chunks_path = PROCESSED_DIR / "huatuo26m-lite_chunks.parquet"
    chunks_df.to_parquet(chunks_path, index=False)

    print(f"标准化文档: {len(df)} 条 -> {docs_path}")
    print(f"文本块: {len(chunks_df)} 块 -> {chunks_path}")


if __name__ == "__main__":
    main()