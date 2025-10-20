import argparse
from pathlib import Path

import pandas as pd
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import (
    ensure_dirs,
    PROCESSED_DIR,
    VECTORDB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


def build_collection(collection_name: str, recreate: bool = False):
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks_to_collection(collection, df: pd.DataFrame, model: SentenceTransformer, batch_size: int = 256):
    texts = df["text"].astype(str).tolist()
    ids = df["chunk_id"].astype(str).tolist()
    metas = df[["doc_id", "source", "specialty"]].to_dict(orient="records")

    for start in tqdm(range(0, len(texts), batch_size), desc="嵌入并写入", unit="批"):
        end = min(len(texts), start + batch_size)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metas = metas[start:end]
        embs = model.encode(
            batch_texts,
            batch_size=min(batch_size, 64),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        collection.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metas, embeddings=embs)


def query_demo(collection, query: str, top_k: int = 5):
    res = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
    hits = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(docs)):
        hits.append({"rank": i + 1, "distance": dists[i], "meta": metas[i], "text": docs[i]})
    return hits


def main():
    parser = argparse.ArgumentParser("构建ChromaDB向量库与检索示例（contrib）")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument("--recreate", action="store_true", help="若存在则删除并重建")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    ensure_dirs()
    chunks_path = PROCESSED_DIR / "huatuo26m-lite_chunks.parquet"
    if not chunks_path.exists():
        raise FileNotFoundError(f"未找到切片文件: {chunks_path}，请先运行 01_data_preprocess.py")

    df = pd.read_parquet(chunks_path)
    print(f"加载文本块 {len(df)}")

    print(f"加载嵌入模型: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    collection = build_collection(args.collection, recreate=args.recreate)
    add_chunks_to_collection(collection, df, model, batch_size=args.batch_size)
    print(f"集合 {args.collection} 构建完成。向量库目录：{VECTORDB_DIR}")

    if args.query:
        hits = query_demo(collection, args.query, top_k=args.top_k)
        print("\nTop-K 检索结果：")
        for h in hits:
            meta = h["meta"]
            print(f"\n[{h['rank']}] distance={h['distance']:.4f} doc_id={meta.get('doc_id')} source={meta.get('source')} specialty={meta.get('specialty')}")
            print(h["text"][:500])


if __name__ == "__main__":
    main()