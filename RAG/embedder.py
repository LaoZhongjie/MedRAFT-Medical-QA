import os
import torch
from typing import List, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
import tiktoken


class TiktokenHashEmbeddings:
    def __init__(self, dim: int = 768, device: str = None, encoding_name: str = "cl100k_base"):
        self.dim = dim
        self.device = device or os.getenv("DEVICE", "cpu")
        self.enc = tiktoken.get_encoding(encoding_name)

    def _encode_vec(self, text: str):
        toks = self.enc.encode(text or "")
        toks = toks[:4096]  # cap per chunk
        vec = torch.zeros(self.dim, device=self.device)
        # simple hashed bag-of-tokens
        for tid in toks:
            h = (tid * 1315423911) % self.dim
            vec[h] += 1.0
        if vec.sum() > 0:
            vec = torch.nn.functional.normalize(vec, p=2.0, dim=0)
        return vec.cpu().tolist()

    def embed_documents(self, texts: List[str]):
        return [self._encode_vec(t) for t in texts]

    def embed_query(self, text: str):
        return self._encode_vec(text)


def build_embeddings(model_name: str = None, device: str = None) -> Any:
    model_name = model_name or os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    device = device or os.getenv("DEVICE", "cuda")

    # HuggingFace endpoint mirror support
    hf_endpoint = os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_HUB_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = hf_endpoint

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except Exception as e:
        # Fallback to local tiktoken-hash embeddings (GPU-capable)
        print(f"[WARN] HF embeddings init failed: {e}. Using TiktokenHashEmbeddings fallback.")
        return TiktokenHashEmbeddings(dim=768, device=device)


def embed_texts(embeddings, texts: List[str]):
    return embeddings.embed_documents(texts)