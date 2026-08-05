import re

from retrieval import Chunker
from retrieval.chunking.recursive import RecursiveChunker
from retrieval.chunking.tokenizers import TextTokenizer

_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


class MarkdownChunker(Chunker):
    """Keep Markdown sections intact, recursively splitting oversized sections."""

    def __init__(
        self,
        chunk_size: int,
        *,
        chunk_overlap: int = 0,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self._section_chunker = RecursiveChunker(
            chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
        )

    def chunk(self, text: str) -> list[str]:
        return [
            chunk
            for section in _split_sections(text)
            for chunk in self._section_chunker.chunk(section)
        ]


def _split_sections(text: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    fence_marker: str | None = None

    for line in text.splitlines(keepends=True):
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                fence_marker = marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                fence_marker = None

        if fence_marker is None and _HEADING.match(line) and current:
            sections.append("".join(current))
            current = []
        current.append(line)

    if current:
        sections.append("".join(current))
    return sections
