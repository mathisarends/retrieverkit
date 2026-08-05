from abc import ABC, abstractmethod
from collections.abc import Sequence

from retrieval.types import Document, Embedding, RetrievalResult


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        raise NotImplementedError


class LexicalTokenizer(ABC):
    """Split text into the comparable terms a lexical index matches on."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError


class TextIndex(Retriever):
    @abstractmethod
    async def index(self, documents: Sequence[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, document_ids: Sequence[str]) -> None:
        """Remove documents by id, ignoring ids the index does not hold."""
        raise NotImplementedError
