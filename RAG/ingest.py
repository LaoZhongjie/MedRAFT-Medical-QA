import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

from loaders import load_documents
from splitter import split_documents
from embedder import build_embeddings


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Preprocess and ingest documents to Chroma")
    parser.add_argument("--input", type=str, default="./data", help="Input directory")
    parser.add_argument("--persist_dir", type=str, default=os.getenv("CHROMA_DIR", "./chroma_store"))
    parser.add_argument("--reset", action="store_true", help="Clear existing Chroma store")
    args = parser.parse_args()

    if args.reset and os.path.exists(args.persist_dir):
        # 清空旧库
        for root, dirs, files in os.walk(args.persist_dir):
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

    # 加载
    result = load_documents(args.input)
    print(f"Loaded {len(result.documents)} chunks from {result.files_count} files")

    # 切分
    chunks = split_documents(result.documents)
    print(f"Split into {len(chunks)} chunks")

    # 嵌入
    embeddings = build_embeddings()

    # 入库并持久化
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=args.persist_dir)
    vs.persist()
    print(f"Persisted to {args.persist_dir}")


if __name__ == "__main__":
    main()