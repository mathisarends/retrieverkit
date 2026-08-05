import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from retrieval import Embedding, EmbeddingProvider
from retrieval.sqlite import SQLiteEmbeddingCache


class CountingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embeddings: dict[str, Embedding]) -> None:
        self._embeddings = embeddings
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls.append(list(texts))
        return [self._embeddings[text] for text in texts]


def _cache(
    database: Path, provider: EmbeddingProvider, *, namespace: str = "model"
) -> SQLiteEmbeddingCache:
    return SQLiteEmbeddingCache(database, provider, namespace=namespace)


@contextmanager
def _closing(cache: SQLiteEmbeddingCache) -> Iterator[SQLiteEmbeddingCache]:
    try:
        yield cache
    finally:
        asyncio.run(cache.close())


def test_embed_only_asks_the_provider_for_uncached_texts(tmp_path: Path) -> None:
    provider = CountingEmbeddingProvider({"first": [1.0, 0.0], "second": [0.0, 1.0]})
    with _closing(_cache(tmp_path / "cache.db", provider)) as cache:
        first = asyncio.run(cache.embed(["first"]))
        both = asyncio.run(cache.embed(["first", "second"]))

    assert first == [[1.0, 0.0]]
    assert both == [[1.0, 0.0], [0.0, 1.0]]
    assert provider.calls == [["first"], ["second"]]


def test_embed_survives_reopening_the_database(tmp_path: Path) -> None:
    database = tmp_path / "persistent.db"
    provider = CountingEmbeddingProvider({"text": [0.5, 0.5]})
    with _closing(_cache(database, provider)) as cache:
        asyncio.run(cache.embed(["text"]))

    with _closing(_cache(database, provider)) as reopened:
        embeddings = asyncio.run(reopened.embed(["text"]))

    assert embeddings == [[0.5, 0.5]]
    assert provider.calls == [["text"]]


def test_embed_returns_one_embedding_per_input_including_duplicates(
    tmp_path: Path,
) -> None:
    provider = CountingEmbeddingProvider({"repeated": [1.0, 0.0], "other": [0.0, 1.0]})
    with _closing(_cache(tmp_path / "duplicates.db", provider)) as cache:
        embeddings = asyncio.run(cache.embed(["repeated", "other", "repeated"]))

    assert embeddings == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert provider.calls == [["repeated", "other"]]


def test_namespaces_do_not_share_embeddings(tmp_path: Path) -> None:
    database = tmp_path / "namespaced.db"
    small = CountingEmbeddingProvider({"text": [1.0, 0.0]})
    large = CountingEmbeddingProvider({"text": [0.0, 1.0, 0.0]})

    with _closing(_cache(database, small, namespace="small")) as cache:
        small_embedding = asyncio.run(cache.embed(["text"]))
    with _closing(_cache(database, large, namespace="large")) as cache:
        large_embedding = asyncio.run(cache.embed(["text"]))

    assert small_embedding == [[1.0, 0.0]]
    assert large_embedding == [[0.0, 1.0, 0.0]]


def test_embed_without_texts_does_not_reach_the_provider(tmp_path: Path) -> None:
    provider = CountingEmbeddingProvider({})
    with _closing(_cache(tmp_path / "empty.db", provider)) as cache:
        assert asyncio.run(cache.embed([])) == []

    assert provider.calls == []


def test_delete_forgets_only_requested_texts_in_its_namespace(tmp_path: Path) -> None:
    database = tmp_path / "delete.db"
    provider = CountingEmbeddingProvider({"kept": [1.0], "removed": [0.0]})
    with _closing(_cache(database, provider)) as cache:
        asyncio.run(cache.embed(["kept", "removed"]))
        asyncio.run(cache.delete(["removed", "removed"]))
        asyncio.run(cache.embed(["kept", "removed"]))

    assert provider.calls == [["kept", "removed"], ["removed"]]


def test_cache_requires_a_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="namespace must not be empty"):
        _cache(tmp_path / "invalid.db", CountingEmbeddingProvider({}), namespace="")
