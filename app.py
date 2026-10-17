from __future__ import annotations

import chainlit as cl

from generate import GenerationError, answer_question
from retrieve import RetrievalError

WELCOME = (
    "Hi! I'm **CURSOR-HELPER-BOT**. Ask me how to use Cursor to build a chatbot.\n\n"
    "I answer only from my knowledge base, and I'll say \"I don't know\" "
    "when the answer isn't there."
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
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        # answer_question is blocking, so run it off the event loop.
        result = await cl.make_async(answer_question)(message.content)
    except (RetrievalError, GenerationError) as exc:
        await cl.Message(content=f"Sorry, something went wrong: {exc}").send()
        return

    chunks_by_number = {c["question_number"]: c for c in result["chunks"]}
    elements: list[cl.Text] = []
    source_names: list[str] = []
    for source in result["sources"]:
        chunk = chunks_by_number.get(source["question_number"])
        if chunk is None:
            continue
        name = f"{source['question_number']}: {source['question']}"
        source_names.append(name)
        elements.append(cl.Text(name=name, content=chunk["text"], display="side"))

    content = result["answer"]
    if source_names:
        content += "\n\n**Sources:** " + " | ".join(source_names)

    await cl.Message(content=content, elements=elements).send()
