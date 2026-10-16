from pathlib import Path

DATA_DIR = Path("data")
CHROMA_PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "cursor_helper_kb"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K = 4