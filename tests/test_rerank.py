import asyncio
from collections.abc import Sequence

import pytest

from retrieval import Document, Embedding, EmbeddingProvider, RetrievalResult, Retriever
from retrieval.rerank import MaximalMarginalRelevance

# Both candidates sit equally close to the query, so only their similarity to
# an already-selected document can separate them: "near-duplicate" repeats
# "best" exactly, while "different" leans on an axis "best" does not use.
_OFF_AXIS = 0.43589  # sqrt(1 - 0.9**2), keeping the vectors unit length
_EMBEDDINGS: dict[str, Embedding] = {
    "query": [1.0, 0.0, 0.0],
    "best": [0.9, _OFF_AXIS, 0.0],
    "near-duplicate": [0.9, _OFF_AXIS, 0.0],
    "different": [0.9, 0.0, _OFF_AXIS],
}


class FakeRetriever(Retriever):
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.requested_limit: int | None = None

    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        self.requested_limit = limit
        return self._results[:limit]


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embeddings: dict[str, Embedding] | None = None) -> None:
        self._embeddings = embeddings or _EMBEDDINGS
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls += 1
        return [self._embeddings[text] for text in texts]


def _result(text: str, score: float) -> RetrievalResult:
    return RetrievalResult(document=Document(id=text, text=text), score=score)


def _candidates() -> list[RetrievalResult]:
    return [
        _result("best", 1.0),
        _result("near-duplicate", 0.99),
        _result("different", 0.1),
    ]


def test_retrieve_drops_a_near_duplicate_in_favour_of_a_different_document() -> None:
    reranker = MaximalMarginalRelevance(
        FakeRetriever(_candidates()), FakeEmbeddingProvider()
    )

    results = asyncio.run(reranker.retrieve("query", limit=2))

    assert [result.document.id for result in results] == ["best", "different"]


def test_full_relevance_keeps_the_underlying_relevance_order() -> None:
    reranker = MaximalMarginalRelevance(
        FakeRetriever(_candidates()), FakeEmbeddingProvider(), relevance=1.0
    )

    results = asyncio.run(reranker.retrieve("query", limit=2))

    assert [result.document.id for result in results] == ["best", "near-duplicate"]


def test_scores_describe_the_ranking_the_reranker_produced() -> None:
    reranker = MaximalMarginalRelevance(
        FakeRetriever(_candidates()), FakeEmbeddingProvider(), relevance=0.5
    )

    results = asyncio.run(reranker.retrieve("query", limit=2))

    assert [result.score for result in results] == pytest.approx(
        [0.45, 0.045], abs=0.0001
    )


def test_retrieve_over_fetches_relative_to_the_limit() -> None:
    retriever = FakeRetriever(_candidates())
    reranker = MaximalMarginalRelevance(
        retriever, FakeEmbeddingProvider(), candidates=1, candidate_multiplier=4
    )

    asyncio.run(reranker.retrieve("query", limit=3))

    assert retriever.requested_limit == 12


def test_retrieve_embeds_the_query_and_candidates_in_one_call() -> None:
    provider = FakeEmbeddingProvider()
    reranker = MaximalMarginalRelevance(FakeRetriever(_candidates()), provider)

    asyncio.run(reranker.retrieve("query", limit=2))

    assert provider.calls == 1


def test_retrieve_passes_a_single_candidate_through_untouched() -> None:
    provider = FakeEmbeddingProvider()
    reranker = MaximalMarginalRelevance(FakeRetriever([_result("best", 1.0)]), provider)

    results = asyncio.run(reranker.retrieve("query", limit=2))

    assert results == [_result("best", 1.0)]
    assert provider.calls == 0


def test_retrieve_without_candidates_returns_nothing() -> None:
    reranker = MaximalMarginalRelevance(FakeRetriever([]), FakeEmbeddingProvider())

    assert asyncio.run(reranker.retrieve("query")) == []


def test_reranker_rejects_invalid_parameters() -> None:
    retriever = FakeRetriever([])
    provider = FakeEmbeddingProvider()

    with pytest.raises(ValueError, match="relevance must be between 0 and 1"):
        MaximalMarginalRelevance(retriever, provider, relevance=1.5)
    with pytest.raises(ValueError, match="candidates must be at least 1"):
        MaximalMarginalRelevance(retriever, provider, candidates=0)
    with pytest.raises(ValueError, match="candidate_multiplier must be at least 1"):
        MaximalMarginalRelevance(retriever, provider, candidate_multiplier=0)


def test_retrieve_rejects_non_positive_limit() -> None:
    reranker = MaximalMarginalRelevance(FakeRetriever([]), FakeEmbeddingProvider())

    with pytest.raises(ValueError, match="limit must be at least 1"):
        asyncio.run(reranker.retrieve("query", limit=0))
