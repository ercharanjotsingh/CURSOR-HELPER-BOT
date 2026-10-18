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


def test_condense_without_history_skips_llm(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("LLM should not be called without history")

    monkeypatch.setattr(generate, "_call_llm", boom)
    assert generate.condense_question("Where are rules?", []) == "Where are rules?"
    assert generate.condense_question("Where are rules?", None) == "Where are rules?"


def test_condense_rewrites_followup(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake(system: str, user: str) -> str:
        seen["user"] = user
        return "How do I undo Agent changes in Cursor?"

    monkeypatch.setattr(generate, "_call_llm", fake)
    result = generate.condense_question(
        "and how do I undo it?", [("What is Agent mode?", "Agent mode edits files.")]
    )
    assert result == "How do I undo Agent changes in Cursor?"
    assert "What is Agent mode?" in seen["user"]


def test_condense_falls_back_on_error(monkeypatch) -> None:
    def fail(system: str, user: str) -> str:
        raise generate.GenerationError("down")

    monkeypatch.setattr(generate, "_call_llm", fail)
    result = generate.condense_question("and undo it?", [("q", "a")])
    assert result == "and undo it?"


def test_answer_question_retrieves_with_standalone_question(monkeypatch) -> None:
    queries: list[str] = []

    def fake_retrieve(q: str) -> list[dict[str, Any]]:
        queries.append(q)
        return [make_chunk("Q10", 0.7)]

    monkeypatch.setattr(generate, "condense_question", lambda q, h: "standalone question")
    monkeypatch.setattr(generate, "retrieve", fake_retrieve)
    monkeypatch.setattr(generate, "_call_llm", lambda s, u: "Reject the diff. [Q10]")

    result = answer_question("and undo it?", history=[("a", "b")])
    assert queries == ["standalone question"]
    assert result["standalone_question"] == "standalone question"


def test_finalize_empty_reply_is_idk() -> None:
    result = generate.finalize("", [make_chunk("Q1")])
    assert result["answer"] == IDK_MESSAGE
    assert result["sources"] == []