import os
import pickle
from typing import List, Dict, Any

import faiss
import numpy as np


class FAISSStore:
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "faiss.index")
        self.store_path = os.path.join(storage_dir, "store.pkl")
        self.index: faiss.Index = None  # type: ignore
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def load_or_create(self, embed_dim: int) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.store_path, "rb") as f:
                data = pickle.load(f)
                self.texts = data.get("texts", [])
                self.metadatas = data.get("metadatas", [])
            # basic sanity check
            if not isinstance(self.index, faiss.IndexFlatIP):
                # For simplicity, enforce IndexFlatIP
                self.index = faiss.IndexFlatIP(embed_dim)
                self.texts = []
                self.metadatas = []
        else:
            self.index = faiss.IndexFlatIP(embed_dim)

    @property
    def size(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def add(self, vectors: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if self.index is None:
            raise RuntimeError("Index not initialized; call load_or_create first.")
        if len(texts) != len(metadatas) or len(texts) != len(vectors):
            raise ValueError("vectors/texts/metadatas 长度不一致")
        # vectors should be normalized float32
        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.save()

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, int(self.index.ntotal))
        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            md = self.metadatas[int(idx)]
            txt = self.texts[int(idx)]
            results.append({"score": float(score), "metadata": md, "text": txt, "index": int(idx)})
        return results

    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump({"texts": self.texts, "metadatas": self.metadatas}, f)

    def clear(self) -> None:
        # remove files and reset in-memory state
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        self.texts = []
        self.metadatas = []
        # re-init index (caller should call load_or_create again)