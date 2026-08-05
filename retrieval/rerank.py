from retrieval import Embedding, EmbeddingProvider, RetrievalResult, Retriever
from retrieval._similarity import cosine, normalize


class MaximalMarginalRelevance(Retriever):
    """Re-rank a retriever's results to trade relevance against diversity.

    A ranking by relevance alone tends to fill its top places with near
    duplicates — the same passage from three revisions of a file, or three
    chunks of the same section. Each pick here is scored against how similar
    it is to the query and how similar it already is to what was picked,
    so a document earns its place by adding something.
    """

    def __init__(
        self,
        retriever: Retriever,
        embedding_provider: EmbeddingProvider,
        *,
        relevance: float = 0.5,
        candidates: int = 30,
        candidate_multiplier: int = 4,
    ) -> None:
        if not 0.0 <= relevance <= 1.0:
            raise ValueError("relevance must be between 0 and 1")
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")

        self._retriever = retriever
        self._embedding_provider = embedding_provider
        self._relevance = relevance
        self._candidates = candidates
        self._candidate_multiplier = candidate_multiplier

    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        depth = max(limit * self._candidate_multiplier, self._candidates)
        results = await self._retriever.retrieve(query, limit=depth)
        if len(results) <= 1:
            return results[:limit]

        embeddings = await self._embedding_provider.embed(
            [query, *(result.document.text for result in results)]
        )
        if len(embeddings) != len(results) + 1:
            raise ValueError(
                "Embedding provider returned an unexpected number of embeddings"
            )

        query_embedding = normalize(embeddings[0])
        candidates = [normalize(embedding) for embedding in embeddings[1:]]
        selected = _select(query_embedding, candidates, self._relevance, limit)
        return [
            RetrievalResult(document=results[position].document, score=score)
            for position, score in selected
        ]


def _select(
    query: Embedding,
    candidates: list[Embedding],
    relevance: float,
    limit: int,
) -> list[tuple[int, float]]:
    """Greedily pick positions, returning each with the score that won it.

    Redundancy is the similarity to the nearest already-selected candidate,
    carried forward as a running maximum so each candidate is compared only
    against the newest pick rather than against the whole selection again.
    """
    relevances = [cosine(query, candidate) for candidate in candidates]
    redundancies = [0.0] * len(candidates)
    remaining = list(range(len(candidates)))
    selected: list[tuple[int, float]] = []

    while remaining and len(selected) < limit:
        scores = {
            position: relevance * relevances[position]
            - (1.0 - relevance) * redundancies[position]
            for position in remaining
        }
        best = max(remaining, key=lambda position: scores[position])
        selected.append((best, scores[best]))
        remaining.remove(best)

        for position in remaining:
            redundancies[position] = max(
                redundancies[position], cosine(candidates[position], candidates[best])
            )

    return selected
