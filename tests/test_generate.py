from typing import Any

import generate
from config import IDK_MESSAGE
from generate import answer_question, build_user_prompt


def make_chunk(number: str, distance: float = 0.8) -> dict[str, Any]:
    return {
        "id": f"cursor_helper_kb-{number}",
        "text": f"{number}. Sample question? Sample answer text for {number}.",
        "source": "cursor_helper_kb.md",
        "question_number": number,
        "question": "Sample question?",
        "section": "Sample",
        "distance": distance,
    }


def test_far_question_skips_llm(monkeypatch) -> None:
    monkeypatch.setattr(generate, "retrieve", lambda q: [make_chunk("Q1", 1.9)])

    def boom(*args, **kwargs):
        raise AssertionError("LLM should not be called for distant chunks")

    monkeypatch.setattr(generate, "_call_llm", boom)

    result = answer_question("What is the capital of Australia?")
    assert result["answer"] == IDK_MESSAGE
    assert result["sources"] == []


def test_llm_idk_returns_clean_idk_and_no_sources(monkeypatch) -> None:
    monkeypatch.setattr(generate, "retrieve", lambda q: [make_chunk("Q2", 1.2)])
    monkeypatch.setattr(
        generate, "_call_llm", lambda s, u: "I don't know. The context has no pricing."
    )

    result = answer_question("How much does Cursor Pro cost?")
    assert result["answer"] == IDK_MESSAGE
    assert result["sources"] == []


def test_grounded_answer_returns_cited_sources(monkeypatch) -> None:
    chunks = [make_chunk("Q5", 0.7), make_chunk("Q3", 1.1)]
    monkeypatch.setattr(generate, "retrieve", lambda q: chunks)
    monkeypatch.setattr(
        generate, "_call_llm", lambda s, u: "Rules live in .cursor/rules. [Q5]"
    )

    result = answer_question("Where do project rules live?")
    assert "[Q5]" in result["answer"]
    assert [s["question_number"] for s in result["sources"]] == ["Q5"]


def test_missing_citation_falls_back_to_top_chunk(monkeypatch) -> None:
    chunks = [make_chunk("Q10", 0.7), make_chunk("Q7", 1.0)]
    monkeypatch.setattr(generate, "retrieve", lambda q: chunks)
    monkeypatch.setattr(generate, "_call_llm", lambda s, u: "Reject the diff.")

    result = answer_question("How do I undo Agent changes?")
    assert [s["question_number"] for s in result["sources"]] == ["Q10"]


def test_prompt_contains_chunks_and_question() -> None:
    prompt = build_user_prompt("Where are rules?", [make_chunk("Q5")])
    assert "[Q5]" in prompt
    assert "Sample answer text for Q5" in prompt
    assert "Where are rules?" in prompt
