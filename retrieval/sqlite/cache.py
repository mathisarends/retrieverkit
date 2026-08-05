import asyncio
import hashlib
from array import array
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import aiosqlite

from retrieval import Embedding, EmbeddingProvider
from retrieval.sqlite._common import open_database

_CACHE = "embedding_cache"
_LOOKUP_BATCH = 500


class SQLiteEmbeddingCache(EmbeddingProvider):
    """Embedding provider that remembers what it has already embedded.

    Re-indexing a corpus normally re-embeds every unchanged chunk. Wrapping
    the real provider in this cache reduces that to the texts that actually
    changed, at the cost of one SQLite lookup per text.
    """

    def __init__(
        self,
        database: str | Path,
        embedding_provider: EmbeddingProvider,
        *,
        namespace: str,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must not be empty")

        self._embedding_provider = embedding_provider
        self._database = database
        self._namespace = namespace
        self._connection: aiosqlite.Connection | None = None
        self._operation_lock = asyncio.Lock()
        self._closed = False

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        requested = list(texts)
        if not requested:
            return []

        cached = await self._load(set(requested))
        missing = [text for text in dict.fromkeys(requested) if text not in cached]
        if missing:
            embeddings = await self._embedding_provider.embed(missing)
            if len(embeddings) != len(missing):
                raise ValueError(
                    "Embedding provider returned an unexpected number of embeddings"
                )

            fresh = dict(zip(missing, embeddings, strict=True))
            await self._store(fresh)
            cached.update(fresh)

        return [cached[text] for text in requested]

    async def delete(self, texts: Sequence[str]) -> None:
        """Forget cached embeddings for texts in this cache namespace."""
        digests = [(_digest(text), self._namespace) for text in dict.fromkeys(texts)]
        if not digests:
            return

        async with self._operation_lock:
            connection = await self._ensure_connection()
            try:
                await connection.executemany(
                    f"DELETE FROM {_CACHE} WHERE text_digest = ? AND namespace = ?",
                    digests,
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def close(self) -> None:
        async with self._operation_lock:
            self._closed = True
            if self._connection is not None:
                await self._connection.close()
                self._connection = None

    async def __aenter__(self) -> "SQLiteEmbeddingCache":
        async with self._operation_lock:
            await self._ensure_connection()
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._closed:
            raise RuntimeError("SQLiteEmbeddingCache is closed")
        if self._connection is not None:
            return self._connection

        connection = await open_database(self._database)
        try:
            await self._create_schema(connection)
        except BaseException:
            await connection.close()
            raise
        self._connection = connection
        return connection

    async def _create_schema(self, connection: aiosqlite.Connection) -> None:
        try:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_CACHE} (
                    namespace TEXT NOT NULL,
                    text_digest BLOB NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (namespace, text_digest)
                )
                """
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    async def _load(self, texts: set[str]) -> dict[str, Embedding]:
        digests = {_digest(text): text for text in texts}
        keys = list(digests)
        cached: dict[str, Embedding] = {}

        async with self._operation_lock:
            connection = await self._ensure_connection()
            for start in range(0, len(keys), _LOOKUP_BATCH):
                batch = keys[start : start + _LOOKUP_BATCH]
                placeholders = ", ".join("?" * len(batch))
                async with connection.execute(
                    f"SELECT text_digest, embedding FROM {_CACHE} "
                    "WHERE namespace = ? "
                    f"AND text_digest IN ({placeholders})",
                    [self._namespace, *batch],
                ) as cursor:
                    rows = await cursor.fetchall()
                cached.update(
                    {
                        digests[row["text_digest"]]: _deserialize(row["embedding"])
                        for row in rows
                    }
                )

        return cached

    async def _store(self, embeddings: dict[str, Embedding]) -> None:
        async with self._operation_lock:
            connection = await self._ensure_connection()
            try:
                await connection.executemany(
                    f"""
                    INSERT INTO {_CACHE} (namespace, text_digest, embedding)
                    VALUES (?, ?, ?)
                    ON CONFLICT(namespace, text_digest) DO UPDATE SET
                        embedding = excluded.embedding
                    """,
                    [
                        (self._namespace, _digest(text), _serialize(embedding))
                        for text, embedding in embeddings.items()
                    ],
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _serialize(embedding: Embedding) -> bytes:
    return array("f", embedding).tobytes()


def _deserialize(value: bytes) -> Embedding:
    return array("f", value).tolist()
