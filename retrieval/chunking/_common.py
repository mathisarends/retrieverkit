from retrieval.chunking.tokenizers import TextTokenizer


def validate_sizes(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def token_count(text: str, tokenizer: TextTokenizer) -> int:
    return len(tokenizer.encode(text))


def largest_prefix(text: str, size: int, tokenizer: TextTokenizer) -> int:
    """Return a non-empty prefix whose encoded size does not exceed size."""
    low = 1
    high = len(text)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if token_count(text[:middle], tokenizer) <= size:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    if best == 0:
        raise ValueError(
            "tokenizer cannot encode any non-empty prefix within chunk_size"
        )
    return best


def overlap_start(text: str, overlap: int, tokenizer: TextTokenizer) -> int:
    if overlap == 0:
        return len(text)

    low = 0
    high = len(text) - 1
    best = len(text)
    while low <= high:
        middle = (low + high) // 2
        if token_count(text[middle:], tokenizer) <= overlap:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    return best
