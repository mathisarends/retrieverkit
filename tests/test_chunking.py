from collections.abc import Sequence

import pytest

from retrieval import Chunker
from retrieval.chunking import (
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveChunker,
    TextTokenizer,
)


def test_fixed_size_chunks_with_overlap() -> None:
    chunker = FixedSizeChunker(5, chunk_overlap=2)

    assert chunker.chunk("abcdefghij") == ["abcde", "defgh", "ghij"]


def test_fixed_size_uses_injected_tokenizer_for_size_and_overlap() -> None:
    class DoubleCharacterTokenizer(TextTokenizer):
        def encode(self, text: str) -> Sequence[int]:
            return [
                token
                for character in text
                for token in (ord(character), ord(character))
            ]

        def decode(self, tokens: Sequence[int]) -> str:
            return "".join(chr(tokens[index]) for index in range(0, len(tokens), 2))

    chunker = FixedSizeChunker(
        4,
        chunk_overlap=2,
        tokenizer=DoubleCharacterTokenizer(),
    )

    assert chunker.chunk("abcd") == ["ab", "bc", "cd"]


def test_fixed_size_chunker_implements_public_port() -> None:
    assert isinstance(FixedSizeChunker(10), Chunker)


def test_empty_text_has_no_chunks() -> None:
    assert FixedSizeChunker(10).chunk("") == []
    assert RecursiveChunker(10).chunk("") == []
    assert MarkdownChunker(10).chunk("") == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be at least 1"),
        (10, -1, "chunk_overlap must not be negative"),
        (10, 10, "chunk_overlap must be smaller than chunk_size"),
    ],
)
def test_invalid_sizes_are_rejected(
    chunk_size: int, chunk_overlap: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FixedSizeChunker(chunk_size, chunk_overlap=chunk_overlap)


def test_recursive_chunker_prefers_paragraph_boundaries() -> None:
    chunker = RecursiveChunker(20)

    assert chunker.chunk("First paragraph.\n\nSecond paragraph.") == [
        "First paragraph.\n\n",
        "Second paragraph.",
    ]


def test_recursive_chunker_uses_token_overlap() -> None:
    chunker = RecursiveChunker(5, chunk_overlap=2, separators=())

    assert chunker.chunk("abcdefghij") == ["abcde", "defgh", "ghij"]


def test_recursive_chunker_rejects_empty_separator() -> None:
    with pytest.raises(ValueError, match="separators must not contain empty strings"):
        RecursiveChunker(10, separators=("",))


def test_markdown_chunker_starts_a_new_chunk_at_each_heading() -> None:
    text = "Intro\n\n# Memory\nFacts\n\n## Tone\nWarm and concise."

    assert MarkdownChunker(100).chunk(text) == [
        "Intro\n\n",
        "# Memory\nFacts\n\n",
        "## Tone\nWarm and concise.",
    ]


def test_markdown_headings_inside_code_fences_do_not_start_sections() -> None:
    text = "# Examples\n```markdown\n# Not a heading\n```\n# Real heading\nText"

    assert MarkdownChunker(100).chunk(text) == [
        "# Examples\n```markdown\n# Not a heading\n```\n",
        "# Real heading\nText",
    ]
