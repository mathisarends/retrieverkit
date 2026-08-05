from collections.abc import Sequence

from retrieval import Document, Embedding, EmbeddingProvider, RetrievalResult, TextIndex
from retrieval._similarity import cosine, normalize


class InMemoryVectorIndex(TextIndex):
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding_provider = embedding_provider
        self._documents: dict[str, tuple[Document, Embedding]] = {}

    async def index(self, documents: Sequence[Document]) -> None:
        items = list(documents)
        if not items:
            return

        embeddings = await self._embedding_provider.embed(
            [document.text for document in items]
        )
        if len(embeddings) != len(items):
            raise ValueError(
                "Embedding provider returned an unexpected number of embeddings"
            )

        expected_dimensions = (
            len(next(iter(self._documents.values()))[1])
            if self._documents
            else len(embeddings[0])
        )
        if any(len(embedding) != expected_dimensions for embedding in embeddings):
            raise ValueError("Embedding vectors must have the same dimensions")

        indexed_documents = {
            document.id: (document, normalize(embedding))
            for document, embedding in zip(items, embeddings, strict=True)
        }
        self._documents.update(indexed_documents)

    async def delete(self, document_ids: Sequence[str]) -> None:
        for document_id in document_ids:
            self._documents.pop(document_id, None)

    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not self._documents:
            return []

        embeddings = await self._embedding_provider.embed([query])
        if len(embeddings) != 1:
            raise ValueError(
                "Embedding provider returned an unexpected number of embeddings"
            )

        query_embedding = normalize(embeddings[0])
        results = [
            RetrievalResult(document=document, score=cosine(query_embedding, embedding))
            for document, embedding in self._documents.values()
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]
