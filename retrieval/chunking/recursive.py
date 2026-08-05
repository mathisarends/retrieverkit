from collections.abc import Sequence

from retrieval import Chunker
from retrieval.chunking._common import (
    largest_prefix,
    overlap_start,
    token_count,
    validate_sizes,
)
from retrieval.chunking.tokenizers import CharacterTokenizer, TextTokenizer

_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


class RecursiveChunker(Chunker):
    """Prefer structural separators, falling back to a hard token boundary."""

    def __init__(
        self,
        chunk_size: int,
        *,
        chunk_overlap: int = 0,
        separators: Sequence[str] = _DEFAULT_SEPARATORS,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        validate_sizes(chunk_size, chunk_overlap)
        if any(not separator for separator in separators):
            raise ValueError("separators must not contain empty strings")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = tuple(separators)
        self._tokenizer = tokenizer or CharacterTokenizer()

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        remaining = text
        while remaining:
            if token_count(remaining, self._tokenizer) <= self._chunk_size:
                chunks.append(remaining)
                break

            prefix_end = largest_prefix(remaining, self._chunk_size, self._tokenizer)
            chunk_end = _preferred_break(remaining[:prefix_end], self._separators)
            if (
                chunk_end == 0
                or token_count(remaining[:chunk_end], self._tokenizer)
                <= self._chunk_overlap
            ):
                chunk_end = prefix_end

            chunk = remaining[:chunk_end]
            chunks.append(chunk)
            next_start = overlap_start(chunk, self._chunk_overlap, self._tokenizer)
            if next_start == 0:
                next_start = chunk_end
            remaining = remaining[next_start:]

        return chunks


def _preferred_break(text: str, separators: Sequence[str]) -> int:
    for separator in separators:
        position = text.rfind(separator)
        if position >= 0:
            return position + len(separator)
    return 0
