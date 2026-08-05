import asyncio

import pytest

from retrieval import Document
from retrieval.lexical import InMemoryBM25Index, WordTokenizer


def test_retrieve_ranks_documents_by_term_overlap() -> None:
    index = InMemoryBM25Index()
    asyncio.run(
        index.index(
            [
                Document(id="python", text="Python is a programming language."),
                Document(id="berlin", text="Berlin is the capital of Germany."),
            ]
        )
    )

    results = asyncio.run(index.retrieve("programming language"))

    assert [result.document.id for result in results] == ["python"]
    assert results[0].score > 0.0


def test_retrieve_prefers_the_shorter_document_on_equal_term_frequency() -> None:
    index = InMemoryBM25Index()
    asyncio.run(
        index.index(
            [
                Document(id="short", text="dijkstra"),
                Document(id="long", text="dijkstra " + "filler " * 50),
            ]
        )
    )

    results = asyncio.run(index.retrieve("dijkstra"))

    assert [result.document.id for result in results] == ["short", "long"]


def test_retrieve_ignores_documents_without_matching_terms() -> None:
    index = InMemoryBM25Index()
    asyncio.run(
        index.index([Document(id="berlin", text="Berlin is the capital of Germany.")])
    )

    assert asyncio.run(index.retrieve("quantum chromodynamics")) == []


def test_retrieve_returns_nothing_for_a_query_without_terms() -> None:
    index = InMemoryBM25Index()
    asyncio.run(
        index.index([Document(id="berlin", text="Berlin is the capital of Germany.")])
    )

    assert asyncio.run(index.retrieve("!?!")) == []


def test_index_replaces_a_document_with_the_same_id() -> None:
    index = InMemoryBM25Index()

    asyncio.run(index.index([Document(id="document", text="old term")]))
    asyncio.run(index.index([Document(id="document", text="new term")]))

    assert asyncio.run(index.retrieve("old")) == []
    assert len(asyncio.run(index.retrieve("new"))) == 1


def test_index_keeps_document_frequencies_consistent_across_replacements() -> None:
    index = InMemoryBM25Index()

    asyncio.run(index.index([Document(id="first", text="shared unique")]))
    asyncio.run(index.index([Document(id="first", text="shared")]))
    asyncio.run(index.index([Document(id="second", text="unique")]))

    results = asyncio.run(index.retrieve("unique"))
    assert [result.document.id for result in results] == ["second"]


def test_retrieve_rejects_non_positive_limit() -> None:
    index = InMemoryBM25Index()

    with pytest.raises(ValueError, match="limit must be at least 1"):
        asyncio.run(index.retrieve("query", limit=0))


def test_index_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="k1 must not be negative"):
        InMemoryBM25Index(k1=-1.0)
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        InMemoryBM25Index(b=1.5)


def test_word_tokenizer_case_folds_and_drops_punctuation() -> None:
    assert WordTokenizer().tokenize("Grüße, Welt! 42") == ["grüsse", "welt", "42"]


def test_delete_removes_documents_and_their_term_statistics() -> None:
    index = InMemoryBM25Index()
    asyncio.run(
        index.index(
            [
                Document(id="kept", text="shared term"),
                Document(id="removed", text="shared term"),
            ]
        )
    )

    asyncio.run(index.delete(["removed"]))
    kept_alone = asyncio.run(index.retrieve("shared"))

    asyncio.run(index.index([Document(id="reference", text="unrelated")]))
    index_without_history = InMemoryBM25Index()
    asyncio.run(
        index_without_history.index(
            [
                Document(id="kept", text="shared term"),
                Document(id="reference", text="unrelated"),
            ]
        )
    )
    reference = asyncio.run(index_without_history.retrieve("shared"))

    assert [result.document.id for result in kept_alone] == ["kept"]
    assert asyncio.run(index.retrieve("shared"))[0].score == pytest.approx(
        reference[0].score
    )


def test_delete_ignores_unknown_ids() -> None:
    index = InMemoryBM25Index()
    asyncio.run(index.index([Document(id="kept", text="shared term")]))

    asyncio.run(index.delete(["never-indexed"]))

    assert [result.document.id for result in asyncio.run(index.retrieve("shared"))] == [
        "kept"
    ]
