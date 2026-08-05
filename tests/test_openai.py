import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from openai import AsyncOpenAI

from retrieval.openai import OpenAIEmbeddingProvider


def test_provider_builds_client_with_configured_api_key() -> None:
    with patch("retrieval.openai.provider.AsyncOpenAI") as client_type:
        OpenAIEmbeddingProvider(
            "text-embedding-3-small",
            api_key="test-api-key",
        )

    client_type.assert_called_once_with(api_key="test-api-key")


def test_provider_delegates_environment_authentication_to_client() -> None:
    with patch("retrieval.openai.provider.AsyncOpenAI") as client_type:
        OpenAIEmbeddingProvider("text-embedding-3-small")

    client_type.assert_called_once_with()


def test_embed_returns_vectors_in_input_order() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            ]
        )
    )
    client = cast(
        AsyncOpenAI, SimpleNamespace(embeddings=SimpleNamespace(create=create))
    )
    provider = OpenAIEmbeddingProvider(
        "text-embedding-3-small",
        client=client,
    )

    result = asyncio.run(provider.embed(["first", "second"]))

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    create.assert_awaited_once_with(
        input=["first", "second"],
        model="text-embedding-3-small",
        encoding_format="float",
    )


def test_embed_passes_configured_dimensions() -> None:
    create = AsyncMock(return_value=SimpleNamespace(data=[]))
    client = cast(
        AsyncOpenAI, SimpleNamespace(embeddings=SimpleNamespace(create=create))
    )
    provider = OpenAIEmbeddingProvider(
        "custom-embedding-model",
        client=client,
        dimensions=256,
    )

    asyncio.run(provider.embed(["text"]))

    create.assert_awaited_once_with(
        input=["text"],
        model="custom-embedding-model",
        dimensions=256,
        encoding_format="float",
    )
