from retrieval import Chunker
from retrieval.chunking._common import validate_sizes
from retrieval.chunking.tokenizers import CharacterTokenizer, TextTokenizer


class FixedSizeChunker(Chunker):
    """Split text into fixed-size token windows with optional overlap."""

    def __init__(
        self,
        chunk_size: int,
        *,
        chunk_overlap: int = 0,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        validate_sizes(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer or CharacterTokenizer()

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        tokens = list(self._tokenizer.encode(text))
        step = self._chunk_size - self._chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return chunks
