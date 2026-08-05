from abc import ABC, abstractmethod
from collections.abc import Sequence


class TextTokenizer(ABC):
    """Minimal tokenizer interface used to measure and slice chunks."""

    @abstractmethod
    def encode(self, text: str) -> Sequence[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(TextTokenizer):
    """Treat each Unicode code point as one token."""

    def encode(self, text: str) -> Sequence[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(token) for token in tokens)
