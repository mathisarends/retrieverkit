from retrieval.chunking import (
    CharacterTokenizer,
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveChunker,
)

TEXT = """# Memory
Cara remembers durable facts about the user and ongoing projects.

## Personality
Cara is warm, direct, and concise. It avoids unnecessary repetition.

## Tonality
- Prefer plain language.
- Explain technical terms when they first appear.
- Keep answers focused on the user's goal.
"""


def _print_chunks(name: str, chunks: list[str]) -> None:
    print(f"\n{name}")
    for number, chunk in enumerate(chunks, start=1):
        print(f"--- chunk {number} ---")
        print(repr(chunk))


def main() -> None:
    fixed = FixedSizeChunker(chunk_size=80, chunk_overlap=10)
    recursive = RecursiveChunker(chunk_size=80, chunk_overlap=10)
    markdown = MarkdownChunker(chunk_size=120, chunk_overlap=15)

    _print_chunks("Fixed-size", fixed.chunk(TEXT))
    _print_chunks("Recursive", recursive.chunk(TEXT))
    _print_chunks("Markdown-aware", markdown.chunk(TEXT))

    # CharacterTokenizer is the default. Pass any tokenizer with compatible
    # encode(str) and decode(token_ids) methods for model-token-aware limits.
    token_aware_api = RecursiveChunker(
        chunk_size=80,
        chunk_overlap=10,
        tokenizer=CharacterTokenizer(),
    )
    _print_chunks("Explicit tokenizer API", token_aware_api.chunk(TEXT))


if __name__ == "__main__":
    main()
