from collections.abc import Sequence

from openai import AsyncOpenAI

from retrieval import Embedding, EmbeddingProvider
from retrieval.openai.models import OpenAIEmbeddingModel


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: OpenAIEmbeddingModel | str,
        *,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model = str(model)
        if client is not None:
            self._client = client
        elif api_key is not None:
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            self._client = AsyncOpenAI()
        self._dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        inputs = list(texts)
        if self._dimensions is None:
            response = await self._client.embeddings.create(
                input=inputs,
                model=self._model,
                encoding_format="float",
            )
        else:
            response = await self._client.embeddings.create(
                input=inputs,
                model=self._model,
                dimensions=self._dimensions,
                encoding_format="float",
            )

        return [
            list(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]
