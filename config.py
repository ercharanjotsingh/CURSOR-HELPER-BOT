from pathlib import Path

DATA_DIR = Path("data")
CHROMA_PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "cursor_helper_kb"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

TOP_K = 6
MAX_DISTANCE = 1.6
IDK_MESSAGE = "I don't know."

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.5-flash-lite"]