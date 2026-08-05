import asyncio
from collections.abc import Sequence

import pytest

from retrieval import Document, Embedding, EmbeddingProvider
from retrieval.vector import InMemoryVectorIndex


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embeddings: dict[str, Embedding]) -> None:
        self._embeddings = embeddings

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return [self._embeddings[text] for text in texts]


def test_retrieve_returns_most_similar_documents_first() -> None:
    provider = FakeEmbeddingProvider(
        {
            "Python is a programming language.": [1.0, 0.0],
            "Berlin is the capital of Germany.": [0.0, 1.0],
            "Which text is about software?": [0.9, 0.1],
        }
    )
    index = InMemoryVectorIndex(provider)
    asyncio.run(
        index.index(
            [
                Document(id="python", text="Python is a programming language."),
                Document(id="berlin", text="Berlin is the capital of Germany."),
            ]
        )
    )

    results = asyncio.run(index.retrieve("Which text is about software?", limit=1))

    assert results[0].document.id == "python"
    assert results[0].score == pytest.approx(0.9939, abs=0.0001)


def test_index_replaces_a_document_with_the_same_id() -> None:
    provider = FakeEmbeddingProvider(
        {
            "old": [1.0, 0.0],
            "new": [0.0, 1.0],
            "query": [1.0, 0.0],
        }
    )
    index = InMemoryVectorIndex(provider)

    asyncio.run(index.index([Document(id="document", text="old")]))
    asyncio.run(index.index([Document(id="document", text="new")]))

    results = asyncio.run(index.retrieve("query"))
    assert len(results) == 1
    assert results[0].document.text == "new"
    assert results[0].score == 0.0


def test_retrieve_rejects_non_positive_limit() -> None:
    index = InMemoryVectorIndex(FakeEmbeddingProvider({}))

    with pytest.raises(ValueError, match="limit must be at least 1"):
        asyncio.run(index.retrieve("query", limit=0))


def test_index_rejects_inconsistent_dimensions() -> None:
    provider = FakeEmbeddingProvider({"first": [1.0], "second": [1.0, 0.0]})
    index = InMemoryVectorIndex(provider)

    with pytest.raises(ValueError, match="same dimensions"):
        asyncio.run(
            index.index(
                [
                    Document(id="first", text="first"),
                    Document(id="second", text="second"),
                ]
            )
        )


def test_delete_removes_documents_from_the_rankings() -> None:
    provider = FakeEmbeddingProvider(
        {"first": [1.0, 0.0], "second": [0.0, 1.0], "query": [1.0, 0.0]}
    )
    index = InMemoryVectorIndex(provider)
    asyncio.run(
        index.index(
            [Document(id="first", text="first"), Document(id="second", text="second")]
        )
    )

    asyncio.run(index.delete(["first", "never-indexed"]))

    assert [result.document.id for result in asyncio.run(index.retrieve("query"))] == [
        "second"
    ]


def test_delete_leaves_an_empty_index_queryable() -> None:
    provider = FakeEmbeddingProvider({"only": [1.0, 0.0], "query": [1.0, 0.0]})
    index = InMemoryVectorIndex(provider)
    asyncio.run(index.index([Document(id="only", text="only")]))

    asyncio.run(index.delete(["only"]))

    assert asyncio.run(index.retrieve("query")) == []
