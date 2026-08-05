import asyncio

import pytest

from retrieval import Document, RetrievalResult, Retriever
from retrieval.fusion import ReciprocalRankFusion


class FakeRetriever(Retriever):
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.requested_limit: int | None = None

    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        self.requested_limit = limit
        return self._results[:limit]


def _result(document_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        document=Document(id=document_id, text=document_id), score=score
    )


def test_retrieve_prefers_documents_ranked_well_by_every_retriever() -> None:
    dense = FakeRetriever([_result("shared", 0.9), _result("dense-only", 0.8)])
    lexical = FakeRetriever([_result("lexical-only", 42.0), _result("shared", 12.0)])
    fusion = ReciprocalRankFusion([dense, lexical])

    results = asyncio.run(fusion.retrieve("query"))

    assert results[0].document.id == "shared"
    assert len(results) == 3


def test_retrieve_ignores_incomparable_score_scales() -> None:
    dense = FakeRetriever([_result("dense-only", 0.51), _result("shared", 0.5)])
    lexical = FakeRetriever([_result("shared", 99.0)])
    fusion = ReciprocalRankFusion([dense, lexical])

    results = asyncio.run(fusion.retrieve("query"))

    assert [result.document.id for result in results] == ["shared", "dense-only"]


def test_retrieve_requests_more_candidates_than_it_returns() -> None:
    retriever = FakeRetriever([_result("only", 1.0)])
    fusion = ReciprocalRankFusion([retriever], candidates_per_retriever=30)

    asyncio.run(fusion.retrieve("query", limit=2))

    assert retriever.requested_limit == 30


def test_retrieve_honours_a_limit_above_the_candidate_depth() -> None:
    retriever = FakeRetriever([_result("first", 1.0), _result("second", 0.5)])
    fusion = ReciprocalRankFusion([retriever], candidates_per_retriever=1)

    results = asyncio.run(fusion.retrieve("query", limit=2))

    assert retriever.requested_limit == 2
    assert len(results) == 2


def test_retrieve_rejects_non_positive_limit() -> None:
    fusion = ReciprocalRankFusion([FakeRetriever([])])

    with pytest.raises(ValueError, match="limit must be at least 1"):
        asyncio.run(fusion.retrieve("query", limit=0))


def test_fusion_requires_at_least_one_retriever() -> None:
    with pytest.raises(ValueError, match="At least one retriever is required"):
        ReciprocalRankFusion([])


def test_weights_decide_between_otherwise_equal_rankings() -> None:
    dense = FakeRetriever([_result("dense-only", 0.9)])
    lexical = FakeRetriever([_result("lexical-only", 42.0)])

    balanced = asyncio.run(ReciprocalRankFusion([dense, lexical]).retrieve("query"))
    dense_favoured = asyncio.run(
        ReciprocalRankFusion([dense, lexical], weights=[0.7, 0.3]).retrieve("query")
    )

    assert [result.score for result in balanced] == pytest.approx([1 / 61, 1 / 61])
    assert [result.document.id for result in dense_favoured] == [
        "dense-only",
        "lexical-only",
    ]


def test_a_weighted_retriever_cannot_outrank_agreement_on_its_own() -> None:
    dense = FakeRetriever([_result("dense-only", 0.9), _result("shared", 0.8)])
    lexical = FakeRetriever([_result("shared", 42.0)])
    fusion = ReciprocalRankFusion([dense, lexical], weights=[0.7, 0.3])

    results = asyncio.run(fusion.retrieve("query"))

    assert [result.document.id for result in results] == ["shared", "dense-only"]


def test_candidate_multiplier_scales_the_over_fetch_with_the_limit() -> None:
    retriever = FakeRetriever([])
    fusion = ReciprocalRankFusion(
        [retriever], candidates_per_retriever=1, candidate_multiplier=4
    )

    asyncio.run(fusion.retrieve("query", limit=10))

    assert retriever.requested_limit == 40


def test_candidate_multiplier_never_fetches_below_the_candidate_floor() -> None:
    retriever = FakeRetriever([])
    fusion = ReciprocalRankFusion(
        [retriever], candidates_per_retriever=60, candidate_multiplier=4
    )

    asyncio.run(fusion.retrieve("query", limit=5))

    assert retriever.requested_limit == 60


def test_fusion_rejects_weights_that_do_not_match_the_retrievers() -> None:
    with pytest.raises(
        ValueError, match="weights must contain one value per retriever"
    ):
        ReciprocalRankFusion([FakeRetriever([])], weights=[0.7, 0.3])


def test_fusion_rejects_non_positive_weights() -> None:
    with pytest.raises(ValueError, match="weights must be positive"):
        ReciprocalRankFusion([FakeRetriever([])], weights=[0.0])


def test_fusion_rejects_a_non_positive_candidate_multiplier() -> None:
    with pytest.raises(ValueError, match="candidate_multiplier must be at least 1"):
        ReciprocalRankFusion([FakeRetriever([])], candidate_multiplier=0)
