import pytest

from retrieve import RetrievalError, get_collection, retrieve

REQUIRED_FIELDS = {
    "id",
    "text",
    "source",
    "question_number",
    "question",
    "section",
    "distance",
}

try:
    get_collection()
    COLLECTION_READY = True
except RetrievalError:
    COLLECTION_READY = False

pytestmark = pytest.mark.skipif(
    not COLLECTION_READY, reason="Run `python ingest.py` first."
)


def test_shortcut_question_finds_expected_chunk() -> None:
    results = retrieve("Which shortcut opens the Chat/Agent panel?", top_k=4)
    ids = [item["id"] for item in results]
    assert "cursor_helper_kb-Q1" in ids or "cursor_helper_kb-Q8" in ids


def test_results_include_all_metadata_fields() -> None:
    results = retrieve("Where do project rules live?")
    assert results
    for item in results:
        assert REQUIRED_FIELDS <= set(item.keys())
        assert item["source"] == "cursor_helper_kb.md"


def test_top_k_is_respected() -> None:
    assert len(retrieve("How do I undo Agent changes?", top_k=2)) == 2
    assert len(retrieve("How do I undo Agent changes?", top_k=5)) == 5


def test_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError):
        retrieve("anything", top_k=0)
