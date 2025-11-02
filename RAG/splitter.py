from typing import List

from langchain_text_splitters import TokenTextSplitter


def build_token_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: List[str] = None,
):
    return TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_documents(docs):
    splitter = build_token_text_splitter()
    return splitter.split_documents(docs)