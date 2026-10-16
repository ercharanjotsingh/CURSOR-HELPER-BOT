from __future__ import annotations

import os
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import GEMINI_FALLBACK_MODELS, GEMINI_MODEL, IDK_MESSAGE, MAX_DISTANCE
from retrieve import RetrievalError, retrieve

load_dotenv()  # reads GEMINI_API_KEY from the .env file

MAX_RETRIES = 3  # attempts per model before moving to the next one

CITATION_RE = re.compile(r"\[(Q\d+)\]")

SYSTEM_PROMPT = f"""You are CURSOR-HELPER-BOT, an assistant that answers questions \
about using Cursor to build chatbots.

Rules:
- Answer ONLY using the context chunks provided in the user message.
- If the context does not clearly contain the answer, reply with exactly: {IDK_MESSAGE}
- Do not use outside knowledge, even if you know the answer. Do not guess.
- Keep the answer short and practical.
- At the end of the answer, cite the chunk(s) you used, like [Q5] or [Q5][Q7]."""


class GenerationError(RuntimeError):
    """Raised when the Gemini API cannot be reached or fails."""


def build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Combine the retrieved chunks and the question into one prompt."""
    context = "\n\n".join(f"[{c['question_number']}] {c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send the prompt to Gemini and return its text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError("GEMINI_API_KEY is not set. Add it to your .env file.")
    client = genai.Client(api_key=api_key)
    models = [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
    last_error: Exception | None = None

    for model in models:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                    ),
                )
                # An empty reply (e.g. a blocked response) counts as "I don't know".
                return (response.text or IDK_MESSAGE).strip()
            except Exception as exc:
                last_error = exc
                if _is_model_unavailable(exc):
                    break  # this model is retired or not offered to you: try the next
                if not _is_temporary_error(exc):
                    raise GenerationError(
                        f"Gemini request failed for model '{model}'. Check your API "
                        f"key and the model name. ({exc})"
                    ) from exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))  # wait 2s, then 4s
        # Still overloaded after all retries: try the next model in the list.

    raise GenerationError(
        f"Gemini is busy or rate-limited for all models tried ({', '.join(models)}). "
        f"Wait a minute and try again. ({last_error})"
    )


def _is_model_unavailable(exc: Exception) -> bool:
    """True when the model name is retired, unknown, or not offered to this key."""
    text = str(exc).upper()
    return "404" in text or "NOT_FOUND" in text


def _is_temporary_error(exc: Exception) -> bool:
    """True for overload / rate-limit errors that are worth retrying."""
    text = str(exc).upper()
    return any(
        marker in text
        for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "OVERLOADED")
    )


def _is_idk(text: str) -> bool:
    normalized = text.strip().lower().replace("\u2019", "'")
    return normalized.startswith("i don't know")


def _cited_sources(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return metadata for the chunks the answer cites (falls back to the top chunk)."""
    cited = set(CITATION_RE.findall(answer))
    selected = [c for c in chunks if c["question_number"] in cited]
    if not selected and chunks:
        selected = chunks[:1]
    return [
        {
            "source": c["source"],
            "question_number": c["question_number"],
            "question": c["question"],
        }
        for c in selected
    ]


def answer_question(question: str) -> dict[str, Any]:
    """Retrieve chunks, ask the LLM, and return answer + sources.

    Returns: {"answer": str, "sources": list[dict], "chunks": list[dict]}
    """
    chunks = retrieve(question)

    # Cheap guard: if even the best chunk is far away, don't call the LLM.
    if not chunks or chunks[0]["distance"] > MAX_DISTANCE:
        return {"answer": IDK_MESSAGE, "sources": [], "chunks": chunks}

    raw = _call_llm(SYSTEM_PROMPT, build_user_prompt(question, chunks))
    if _is_idk(raw):
        return {"answer": IDK_MESSAGE, "sources": [], "chunks": chunks}

    return {"answer": raw, "sources": _cited_sources(raw, chunks), "chunks": chunks}


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python generate.py "your question"')
        os._exit(1)

    question = " ".join(sys.argv[1:])
    try:
        result = answer_question(question)
    except (RetrievalError, GenerationError) as exc:
        print(f"Error: {exc}")
        os._exit(1)

    print(f"Question: {question}\n")
    print(f"Answer: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} {s['question_number']}: {s['question']}")
    best = result["chunks"][0]["distance"] if result["chunks"] else None
    if best is not None:
        print(f"\n(best distance: {best:.4f})")
    # Chroma's ONNX embedding runtime can leave non-daemon threads running.
    os._exit(0)


if __name__ == "__main__":
    main()