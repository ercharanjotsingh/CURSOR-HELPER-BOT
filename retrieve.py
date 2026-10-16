from __future__ import annotations

import os
import sys
from typing import Any

import chromadb

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K


class RetrievalError(RuntimeError):
    """Raised when the vector store is missing or empty."""


def get_collection() -> chromadb.Collection:
    """Open the persisted Chroma collection created by ingest.py."""
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:  # Chroma raises different errors by version
        raise RetrievalError(
            f"Collection '{COLLECTION_NAME}' not found. Run `python ingest.py` first."
        ) from exc
    if collection.count() == 0:
        raise RetrievalError(
            f"Collection '{COLLECTION_NAME}' is empty. Run `python ingest.py` first."
        )
    return collection


def retrieve(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return the top-k most similar chunks for a question, best first.

    Each result has: id, text, source, question_number, question,
    section, distance (lower = more similar).
    """
    k = TOP_K if top_k is None else top_k
    if k < 1:
        raise ValueError("top_k must be at least 1")

    collection = get_collection()
    result = collection.query(
        query_texts=[question],
        n_results=min(k, collection.count()),
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    seen: set[str] = set()
    chunks: list[dict[str, Any]] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        meta = meta or {}
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "source": meta.get("source", ""),
                "question_number": meta.get("question_number", ""),
                "question": meta.get("question", ""),
                "section": meta.get("section", ""),
                "distance": float(distance),
            }
        )

    chunks.sort(key=lambda item: item["distance"])
    return chunks


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "your question"')
        os._exit(1)

    question = " ".join(sys.argv[1:])
    try:
        results = retrieve(question)
    except RetrievalError as exc:
        print(f"Error: {exc}")
        os._exit(1)

    print(f"Question: {question}\n")
    for rank, item in enumerate(results, start=1):
        preview = item["text"][:150].replace("\n", " ")
        print(f"{rank}. {item['id']}  distance={item['distance']:.4f}")
        print(f"   {item['question']}")
        print(f"   {preview}...\n")
    # Chroma's ONNX embedding runtime can leave non-daemon threads running.
    os._exit(0)


if __name__ == "__main__":
    main()
