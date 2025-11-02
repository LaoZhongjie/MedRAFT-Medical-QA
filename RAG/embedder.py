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

    # HuggingFace endpoint mirror + 离线模式支持
    hf_endpoint = os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_HUB_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = hf_endpoint
    hf_offline = os.getenv("HF_HUB_OFFLINE")
    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = hf_offline

    # 如果提供了 EMBED_MODEL_DIR，则尝试确保本地可用（拉取或使用现有）
    local_dir = os.getenv("EMBED_MODEL_DIR")
    model_name_eff = model_name
    if local_dir:
        try:
            # 若目录不存在或为空，尝试本地化下载（尊重镜像与离线设置）
            if not os.path.isdir(local_dir) or not os.listdir(local_dir):
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=model_name,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )
            model_name_eff = local_dir
        except Exception as de:
            print(f"[WARN] Snapshot download failed: {de}. Will try model_name={model_name}.")

    # 若 EMBED_MODEL_NAME 已经是本地路径，则直接用之
    if isinstance(model_name_eff, str) and (model_name_eff.startswith(".") or model_name_eff.startswith("/")):
        if os.path.isdir(model_name_eff):
            pass  # 本地路径可用
        else:
            print(f"[WARN] Local model path not found: {model_name_eff}. Will try remote.")

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name_eff,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except Exception as e:
        print(f"[WARN] HF embeddings init failed: {e}. Using TiktokenHashEmbeddings fallback.")
        return TiktokenHashEmbeddings(dim=768, device=device)


def embed_texts(embeddings, texts: List[str]):
    return embeddings.embed_documents(texts)