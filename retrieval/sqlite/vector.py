import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol

import aiosqlite

from retrieval import Document, Embedding, EmbeddingProvider, RetrievalResult, TextIndex
from retrieval.sqlite._common import (
    DOCUMENT_COLUMNS,
    aliased_document_columns,
    document_from_row,
    document_rows,
    document_table_statement,
    open_database,
    upsert_document_statement,
)

_DOCUMENTS = "documents"
_EMBEDDINGS = "document_embeddings"


class _SQLiteVec(Protocol):
    def loadable_path(self) -> str: ...

    def serialize_float32(self, vector: Sequence[float], /) -> bytes: ...


class SQLiteVectorIndex(TextIndex):
    """Persistent cosine-similarity index backed by SQLite and sqlite-vec."""

    def __init__(
        self,
        database: str | Path,
        embedding_provider: EmbeddingProvider,
        *,
        dimensions: int,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")

        self._sqlite_vec = _import_sqlite_vec()
        self._database = database
        self._embedding_provider = embedding_provider
        self._dimensions = dimensions
        self._connection: aiosqlite.Connection | None = None
        self._operation_lock = asyncio.Lock()
        self._closed = False

    async def index(self, documents: Sequence[Document]) -> None:
        items = list(documents)
        if not items:
            return

        rows = document_rows(items)

        embeddings = await self._embedding_provider.embed(
            [document.text for document in items]
        )
        if len(embeddings) != len(items):
            raise ValueError(
                "Embedding provider returned an unexpected number of embeddings"
            )
        self._validate_embeddings(embeddings)

        async with self._operation_lock:
            connection = await self._ensure_connection()
            try:
                await connection.execute("BEGIN")
                await connection.executemany(
                    upsert_document_statement(_DOCUMENTS), rows
                )
                for document, embedding in zip(items, embeddings, strict=True):
                    await connection.execute(
                        f"DELETE FROM {_EMBEDDINGS} WHERE document_id = ?",
                        (document.id,),
                    )
                    await connection.execute(
                        f"INSERT INTO {_EMBEDDINGS} "
                        "(document_id, embedding, parent_key) VALUES (?, ?, ?)",
                        (
                            document.id,
                            self._sqlite_vec.serialize_float32(embedding),
                            _parent_key(document.parent_id),
                        ),
                    )
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        return await self.search(query, limit=limit)

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        parent_id: str | None = None,
    ) -> list[RetrievalResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        async with self._operation_lock:
            connection = await self._ensure_connection()
            if not await self._has_embeddings(connection, parent_id):
                return []

        embeddings = await self._embedding_provider.embed([query])
        if len(embeddings) != 1:
            raise ValueError(
                "Embedding provider returned an unexpected number of embeddings"
            )
        self._validate_embeddings(embeddings)

        parameters: list[object] = [
            self._sqlite_vec.serialize_float32(embeddings[0]),
            limit,
        ]
        parent_filter = ""
        if parent_id is not None:
            parent_filter = "AND parent_key = ?"
            parameters.append(_parent_key(parent_id))

        async with self._operation_lock:
            connection = await self._ensure_connection()
            async with connection.execute(
                f"""
                WITH matches AS (
                    SELECT document_id, distance
                    FROM {_EMBEDDINGS}
                    WHERE embedding MATCH ?
                      AND k = ?
                      {parent_filter}
                )
                SELECT {aliased_document_columns("d")}, matches.distance
                FROM matches
                JOIN {_DOCUMENTS} AS d ON d.id = matches.document_id
                ORDER BY matches.distance
                """,
                parameters,
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            RetrievalResult(
                document=document_from_row(row), score=1.0 - row["distance"]
            )
            for row in rows
        ]

    async def get(self, document_id: str) -> Document | None:
        async with self._operation_lock:
            connection = await self._ensure_connection()
            async with connection.execute(
                f"SELECT {DOCUMENT_COLUMNS} FROM {_DOCUMENTS} WHERE id = ?",
                (document_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return None if row is None else document_from_row(row)

    async def list_documents(
        self, *, parent_id: str | None = None, limit: int = 100
    ) -> list[Document]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        async with self._operation_lock:
            connection = await self._ensure_connection()
            if parent_id is None:
                statement = f"""
                    SELECT {DOCUMENT_COLUMNS} FROM {_DOCUMENTS}
                    WHERE parent_id IS NULL
                    ORDER BY created_at DESC, id
                    LIMIT ?
                """
                parameters = (limit,)
            else:
                statement = f"""
                    SELECT {DOCUMENT_COLUMNS} FROM {_DOCUMENTS}
                    WHERE parent_id = ?
                    ORDER BY created_at, id
                    LIMIT ?
                """
                parameters = (parent_id, limit)

            async with connection.execute(statement, parameters) as cursor:
                rows = await cursor.fetchall()
        return [document_from_row(row) for row in rows]

    async def delete(self, document_ids: Sequence[str]) -> None:
        ids = [(document_id,) for document_id in dict.fromkeys(document_ids)]
        if not ids:
            return

        async with self._operation_lock:
            connection = await self._ensure_connection()
            try:
                await connection.execute("BEGIN")
                await connection.executemany(
                    f"DELETE FROM {_EMBEDDINGS} WHERE document_id = ?", ids
                )
                await connection.executemany(
                    f"DELETE FROM {_DOCUMENTS} WHERE id = ?", ids
                )
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    async def close(self) -> None:
        async with self._operation_lock:
            self._closed = True
            if self._connection is not None:
                await self._connection.close()
                self._connection = None

    async def __aenter__(self) -> "SQLiteVectorIndex":
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
            raise RuntimeError("SQLiteVectorIndex is closed")
        if self._connection is not None:
            return self._connection

        connection = await open_database(self._database)
        try:
            await self._load_extension(connection)
            await self._create_schema(connection)
        except BaseException:
            await connection.close()
            raise
        self._connection = connection
        return connection

    async def _load_extension(self, connection: aiosqlite.Connection) -> None:
        await connection.enable_load_extension(True)
        try:
            await connection.load_extension(self._sqlite_vec.loadable_path())
        finally:
            await connection.enable_load_extension(False)

    async def _create_schema(self, connection: aiosqlite.Connection) -> None:
        try:
            await connection.executescript(document_table_statement(_DOCUMENTS))
            await connection.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {_EMBEDDINGS} USING vec0(
                    document_id TEXT PRIMARY KEY,
                    embedding float[{self._dimensions}] distance_metric=cosine,
                    parent_key TEXT
                )
                """
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
        await self._validate_schema_dimensions(connection)

    async def _validate_schema_dimensions(
        self, connection: aiosqlite.Connection
    ) -> None:
        async with connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (_EMBEDDINGS,)
        ) as cursor:
            row = await cursor.fetchone()
        expected = f"float[{self._dimensions}]"
        if row is None or expected not in row["sql"].casefold():
            raise ValueError(
                f"The existing {_EMBEDDINGS} table uses different embedding dimensions"
            )

    def _validate_embeddings(self, embeddings: Sequence[Embedding]) -> None:
        if any(len(embedding) != self._dimensions for embedding in embeddings):
            raise ValueError(
                f"Embedding vectors must have {self._dimensions} dimensions"
            )

    async def _has_embeddings(
        self, connection: aiosqlite.Connection, parent_id: str | None
    ) -> bool:
        if parent_id is None:
            parameters: tuple[str, ...] = ()
            statement = f"SELECT EXISTS(SELECT 1 FROM {_EMBEDDINGS})"
        else:
            parameters = (_parent_key(parent_id),)
            statement = (
                f"SELECT EXISTS(SELECT 1 FROM {_EMBEDDINGS} WHERE parent_key = ?)"
            )
        async with connection.execute(statement, parameters) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return bool(row[0])


def _parent_key(parent_id: str | None) -> str:
    return "root" if parent_id is None else f"id:{parent_id}"


def _import_sqlite_vec() -> _SQLiteVec:
    try:
        import sqlite_vec
    except ModuleNotFoundError as error:
        if error.name != "sqlite_vec":
            raise
        raise ImportError(
            "SQLiteVectorIndex requires the optional SQLite dependencies. "
            "Install them with `pip install 'retrieval[sqlite]'`."
        ) from error
    return sqlite_vec
