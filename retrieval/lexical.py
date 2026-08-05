import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from retrieval import Document, LexicalTokenizer, RetrievalResult, TextIndex


class InMemoryBM25Index(TextIndex):
    """Okapi BM25 over documents held in memory.

    Matches on exact terms, so it complements a vector index rather than
    replacing it: rare words, identifiers and numbers score well here even
    when no embedding model has ever seen them.
    """

    def __init__(
        self,
        tokenizer: LexicalTokenizer | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must not be negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")

        self._tokenizer = tokenizer or WordTokenizer()
        self._k1 = k1
        self._b = b
        self._documents: dict[str, _IndexedDocument] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._total_length = 0

    async def index(self, documents: Sequence[Document]) -> None:
        for document in documents:
            self._forget(document.id)

            term_frequencies = Counter(self._tokenizer.tokenize(document.text))
            indexed = _IndexedDocument(
                document=document,
                term_frequencies=term_frequencies,
                length=sum(term_frequencies.values()),
            )
            self._documents[document.id] = indexed
            self._document_frequencies.update(term_frequencies.keys())
            self._total_length += indexed.length

    async def delete(self, document_ids: Sequence[str]) -> None:
        for document_id in document_ids:
            self._forget(document_id)

    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not self._documents:
            return []

        query_terms = set(self._tokenizer.tokenize(query))
        if not query_terms:
            return []

        average_length = self._total_length / len(self._documents)
        results = [
            RetrievalResult(document=indexed.document, score=score)
            for indexed in self._documents.values()
            if (score := self._score(query_terms, indexed, average_length)) > 0.0
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _forget(self, document_id: str) -> None:
        indexed = self._documents.pop(document_id, None)
        if indexed is None:
            return

        self._total_length -= indexed.length
        for term in indexed.term_frequencies:
            if self._document_frequencies[term] > 1:
                self._document_frequencies[term] -= 1
            else:
                del self._document_frequencies[term]

    def _score(
        self, query_terms: set[str], indexed: "_IndexedDocument", average_length: float
    ) -> float:
        length_penalty = self._k1 * (
            1 - self._b + self._b * indexed.length / average_length
        )
        return sum(
            self._inverse_document_frequency(term)
            * frequency
            * (self._k1 + 1)
            / (frequency + length_penalty)
            for term in query_terms
            if (frequency := indexed.term_frequencies[term])
        )

    def _inverse_document_frequency(self, term: str) -> float:
        document_frequency = self._document_frequencies[term]
        total = len(self._documents)
        return math.log(
            1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
        )


class WordTokenizer(LexicalTokenizer):
    """Case-folded Unicode word tokens, without stemming or stop words."""

    _WORDS = re.compile(r"\w+", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        return self._WORDS.findall(text.casefold())


@dataclass(frozen=True, slots=True)
class _IndexedDocument:
    document: Document
    term_frequencies: Counter[str]
    length: int
