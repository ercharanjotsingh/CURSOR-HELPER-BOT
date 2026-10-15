from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypedDict

import chromadb

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DATA_DIR,
)

QUESTION_RE = re.compile(r"^Q(\d+)\.\s*(.*)$")
SECTION_RE = re.compile(r"^[A-Z]\.\s+\S")


class QaChunk(TypedDict):
    id: str
    text: str
    source: str
    question_number: str
    question: str
    section: str


def count_tokens(text: str) -> int:
    return len(text.split())


def split_long_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        parts.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return parts


def _flush_current(
    current: dict[str, str] | None,
    body_lines: list[str],
    chunks: list[QaChunk],
    source: str,
    stem: str,
) -> None:
    if current is None:
        return
    text = "\n".join(body_lines).strip()
    if not text:
        return
    chunks.append(
        QaChunk(
            id=f"{stem}-{current['question_number']}",
            text=text,
            source=source,
            question_number=current["question_number"],
            question=current["question"],
            section=current["section"],
        )
    )


def parse_markdown(path: Path) -> list[QaChunk]:
    source = path.name
    stem = path.stem
    section = ""
    chunks: list[QaChunk] = []
    current: dict[str, str] | None = None
    body_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        question_match = QUESTION_RE.match(line)
        if question_match:
            _flush_current(current, body_lines, chunks, source, stem)
            current = {
                "question_number": f"Q{question_match.group(1)}",
                "question": question_match.group(2).strip(),
                "section": section,
            }
            body_lines = [line]
            continue

        if SECTION_RE.match(line):
            section = line.strip()
            continue

        if current is not None:
            body_lines.append(line)

    _flush_current(current, body_lines, chunks, source, stem)
    return chunks


def expand_long_chunks(chunks: list[QaChunk]) -> list[QaChunk]:
    expanded: list[QaChunk] = []
    for chunk in chunks:
        parts = split_long_text(
            chunk["text"], CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
        )
        if len(parts) == 1:
            expanded.append(chunk)
            continue
        for index, part in enumerate(parts):
            split_chunk = QaChunk(chunk)
            split_chunk["id"] = f"{chunk['id']}-p{index}"
            split_chunk["text"] = part
            expanded.append(split_chunk)
    return expanded


def load_markdown_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.md"))


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def delete_chunks_for_source(collection: chromadb.Collection, source: str) -> None:
    existing = collection.get(where={"source": source})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def upsert_chunks(collection: chromadb.Collection, chunks: list[QaChunk]) -> None:
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "source": chunk["source"],
                "question_number": chunk["question_number"],
                "question": chunk["question"],
                "section": chunk["section"],
            }
            for chunk in chunks
        ],
    )


def ingest() -> list[str]:
    stored_ids: list[str] = []
    collection = get_collection()
    for path in load_markdown_files(DATA_DIR):
        chunks = expand_long_chunks(parse_markdown(path))
        delete_chunks_for_source(collection, path.name)
        if chunks:
            upsert_chunks(collection, chunks)
            stored_ids.extend(chunk["id"] for chunk in chunks)
    return stored_ids


def main() -> None:
    ids = ingest()
    print(f"Stored {len(ids)} chunks")
    for chunk_id in ids:
        print(chunk_id)
    # Chroma's ONNX embedding runtime can leave non-daemon threads running.
    os._exit(0)


if __name__ == "__main__":
    main()
