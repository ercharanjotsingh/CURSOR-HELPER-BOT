from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import chainlit as cl

from config import IDK_MESSAGE
from generate import (
    HISTORY_TURNS,
    GenerationError,
    finalize,
    prepare,
    stream_answer,
)
from retrieve import RetrievalError

WELCOME = (
    "Hi! I'm **CURSOR-HELPER-BOT**. Ask me how to use Cursor to build a chatbot.\n\n"
    "I answer only from my knowledge base, and I'll say \"I don't know\" "
    "when the answer isn't there. You can ask follow-up questions too."
)


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Where do rules live?",
            message="Where do Cursor project rules live?",
        ),
        cl.Starter(
            label="Undo Agent changes",
            message="How do I undo something the Agent got wrong?",
        ),
        cl.Starter(
            label="RAG build steps",
            message="What are the steps to build a RAG chatbot with Cursor?",
        ),
        cl.Starter(
            label="Using GitHub",
            message="How do I use GitHub with Cursor while building?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])  # list of (user message, bot answer)
    await cl.Message(content=WELCOME).send()


async def _tokens(pieces: Iterator[str]) -> AsyncIterator[str]:
    """Pull pieces from the blocking Gemini stream without freezing the server."""
    sentinel = object()
    while True:
        piece = await cl.make_async(next)(pieces, sentinel)
        if piece is sentinel:
            break
        yield piece


def _build_sources(
    result: dict[str, Any], chunks: list[dict[str, Any]]
) -> tuple[list[cl.Text], list[str]]:
    chunks_by_number = {c["question_number"]: c for c in chunks}
    elements: list[cl.Text] = []
    names: list[str] = []
    for source in result["sources"]:
        chunk = chunks_by_number.get(source["question_number"])
        if chunk is None:
            continue
        name = f"{source['question_number']}: {source['question']}"
        names.append(name)
        elements.append(cl.Text(name=name, content=chunk["text"], display="side"))
    return elements, names


@cl.on_message
async def on_message(message: cl.Message) -> None:
    history: list[tuple[str, str]] = cl.user_session.get("history") or []

    try:
        prep = await cl.make_async(prepare)(message.content, history)
    except (RetrievalError, GenerationError) as exc:
        await cl.Message(content=f"Sorry, something went wrong: {exc}").send()
        return

    chunks = prep["chunks"]
    print(f"[debug] standalone={prep['question']!r} too_far={prep['too_far']} top={[(c['id'], round(c['distance'], 2)) for c in chunks]}")
    reply = cl.Message(content="")

    try:
        if prep["too_far"]:
            raw = IDK_MESSAGE
        else:
            raw = ""
            async for token in _tokens(stream_answer(prep["question"], chunks)):
                raw += token
                await reply.stream_token(token)
    except GenerationError as exc:
        reply.content = f"Sorry, something went wrong: {exc}"
        await reply.send()
        return

    result = finalize(raw, chunks)
    elements, names = _build_sources(result, chunks)

    content = result["answer"]
    if names:
        content += "\n\n**Sources:** " + " | ".join(names)
    reply.content = content
    reply.elements = elements
    await reply.send()

    history.append((message.content, result["answer"]))
    cl.user_session.set("history", history[-HISTORY_TURNS:])