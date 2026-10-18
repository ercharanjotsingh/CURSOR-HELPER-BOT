from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Iterator
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import GEMINI_FALLBACK_MODELS, GEMINI_MODEL, IDK_MESSAGE, MAX_DISTANCE
from retrieve import RetrievalError, retrieve

load_dotenv()  # reads GEMINI_API_KEY from the .env file

MAX_RETRIES = 3  # attempts per model before moving to the next one
HISTORY_TURNS = 3  # how many previous Q&A pairs are used to resolve follow-ups

CITATION_RE = re.compile(r"\[(Q\d+)\]")

SYSTEM_PROMPT = f"""You are CURSOR-HELPER-BOT, an assistant that answers questions \
about using Cursor to build chatbots.

Rules:
- Answer ONLY using the context chunks provided in the user message.
- If the context does not clearly contain the answer, reply with exactly: {IDK_MESSAGE}
- Do not use outside knowledge, even if you know the answer. Do not guess.
- Keep the answer short and practical.
- At the end of the answer, cite the chunk(s) you used, like [Q5] or [Q5][Q7]."""

CONDENSE_PROMPT = (
    "Rewrite the user's latest message as one standalone question. Use the chat "
    'history only to resolve references such as "it", "that" or "the previous one". '
    "If the message is already standalone, return it unchanged. "
    "Return only the question, with no explanation."
)


class GenerationError(RuntimeError):
    """Raised when the Gemini API cannot be reached or fails."""


# ---------------------------------------------------------------- Gemini calls


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


def _stream_llm(system_prompt: str, user_prompt: str) -> Iterator[str]:
    """Yield the model's reply in small pieces.

    Retries and model fallback only happen before the first piece arrives.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError("GEMINI_API_KEY is not set. Add it to your .env file.")

    client = genai.Client(api_key=api_key)
    models = [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
    last_error: Exception | None = None

    for model in models:
        for attempt in range(MAX_RETRIES):
            started = False
            try:
                for piece in client.models.generate_content_stream(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                    ),
                ):
                    if piece.text:
                        started = True
                        yield piece.text
                return
            except Exception as exc:
                last_error = exc
                if started:
                    raise GenerationError(
                        f"The Gemini reply was cut off midway. ({exc})"
                    ) from exc
                if _is_model_unavailable(exc):
                    break  # retired or not offered to you: try the next model
                if not _is_temporary_error(exc):
                    raise GenerationError(
                        f"Gemini request failed for model '{model}'. Check your API "
                        f"key and the model name. ({exc})"
                    ) from exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))  # wait 2s, then 4s

    raise GenerationError(
        f"Gemini is busy or rate-limited for all models tried ({', '.join(models)}). "
        f"Wait a minute and try again. ({last_error})"
    )


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Return the full reply as one string (an empty reply counts as 'I don't know')."""
    text = "".join(_stream_llm(system_prompt, user_prompt)).strip()
    return text or IDK_MESSAGE


# ------------------------------------------------------------------ RAG logic


def build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Combine the retrieved chunks and the question into one prompt."""
    context = "\n\n".join(f"[{c['question_number']}] {c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def condense_question(question: str, history: list[tuple[str, str]] | None) -> str:
    """Turn a follow-up like 'and how do I undo it?' into a standalone question.

    history is a list of (user_message, bot_answer) pairs, oldest first.
    Falls back to the original question if there is no history or the call fails.
    """
    if not history:
        return question

    transcript = "\n".join(
        f"User: {user}\nAssistant: {bot}" for user, bot in history[-HISTORY_TURNS:]
    )
    prompt = (
        f"Chat history:\n{transcript}\n\n"
        f"Latest message: {question}\n\nStandalone question:"
    )
    try:
        rewritten = _call_llm(CONDENSE_PROMPT, prompt).strip()
    except GenerationError:
        return question
    if not rewritten or rewritten == IDK_MESSAGE:
        return question
    return rewritten


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


def prepare(
    question: str, history: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    """Resolve follow-ups and retrieve chunks (no answer is generated yet).

    Returns {"question": standalone question, "chunks": [...], "too_far": bool}.
    """
    standalone = condense_question(question, history)
    chunks = retrieve(standalone)
    # Cheap guard: if even the best chunk is far away, don't call the LLM.
    too_far = not chunks or chunks[0]["distance"] > MAX_DISTANCE
    return {"question": standalone, "chunks": chunks, "too_far": too_far}


def stream_answer(question: str, chunks: list[dict[str, Any]]) -> Iterator[str]:
    """Yield the answer piece by piece (used by the chat UI)."""
    return _stream_llm(SYSTEM_PROMPT, build_user_prompt(question, chunks))


def finalize(raw: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Clean up a finished reply: normalise 'I don't know' and attach sources."""
    raw = raw.strip() or IDK_MESSAGE
    if _is_idk(raw):
        return {"answer": IDK_MESSAGE, "sources": []}
    return {"answer": raw, "sources": _cited_sources(raw, chunks)}


def answer_question(
    question: str, history: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    """Retrieve chunks, ask the LLM, and return answer + sources.

    Returns: {"answer", "sources", "chunks", "standalone_question"}
    """
    prep = prepare(question, history)
    chunks = prep["chunks"]
    standalone = prep["question"]

    if prep["too_far"]:
        result = {"answer": IDK_MESSAGE, "sources": []}
    else:
        raw = _call_llm(SYSTEM_PROMPT, build_user_prompt(standalone, chunks))
        result = finalize(raw, chunks)

    return {**result, "chunks": chunks, "standalone_question": standalone}


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