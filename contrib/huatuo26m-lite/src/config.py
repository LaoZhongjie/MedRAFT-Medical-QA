from pathlib import Path

# 以 contrib/huatuo26m-lite 为根目录，避免影响主工程
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT / "data_raw" / "Huatuo_subset"
PROCESSED_DIR = ROOT / "data_processed"
VECTORDB_DIR = ROOT / "vector_db"

COLLECTION_NAME = "huatuo26m-lite"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_Q_LEN = 4
MIN_A_LEN = 8


def ensure_dirs():
    for d in [RAW_DATA_DIR, PROCESSED_DIR, VECTORDB_DIR]:
        d.mkdir(parents=True, exist_ok=True)