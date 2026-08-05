import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import aiosqlite

from retrieval import Document, RetrievalResult, TextIndex
from retrieval.sqlite._common import (
    DOCUMENT_COLUMNS,
    aliased_document_columns,
    document_from_row,
    document_rows,
    document_table_statement,
    open_database,
    upsert_document_statement,
)

_DOCUMENTS = "lexical_documents"
_SEARCH = "lexical_documents_fts"
_QUERY_TERMS = re.compile(r"\w+", re.UNICODE)
_MINIMUM_PREFIX_LENGTH = 6


class SQLiteFts5Index(TextIndex):
    """Persistent BM25 index backed by SQLite's FTS5 extension.

    Terms are matched by FTS5 rather than by a LexicalTokenizer: the
    tokenizer runs inside SQLite, so it is configured with an FTS5
    tokenizer specification instead of a Python object.
    """

    def __init__(
        self, database: str | Path, *, tokenizer: str = "unicode61 remove_diacritics 2"
    ) -> None:
        self._database = database
        self._tokenizer = tokenizer
        self._connection: aiosqlite.Connection | None = None
        self._operation_lock = asyncio.Lock()
        self._closed = False

    async def index(self, documents: Sequence[Document]) -> None:
        items = list(documents)
        if not items:
            return

        rows = document_rows(items)
        async with self._operation_lock:
            connection = await self._ensure_connection()
            try:
                await connection.executemany(
                    upsert_document_statement(_DOCUMENTS), rows
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

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

        match_expression = _match_expression(query)
        if match_expression is None:
            return []

        parameters: list[object] = [match_expression]
        parent_filter = ""
        if parent_id is not None:
            parent_filter = "AND d.parent_id = ?"
            parameters.append(parent_id)
        parameters.append(limit)

        async with self._operation_lock:
            connection = await self._ensure_connection()
            async with connection.execute(
                f"""
                SELECT {aliased_document_columns("d")}, bm25({_SEARCH}) AS relevance
                FROM {_SEARCH}
                JOIN {_DOCUMENTS} AS d ON d.rowid = {_SEARCH}.rowid
                WHERE {_SEARCH} MATCH ?
                  {parent_filter}
                ORDER BY relevance
                LIMIT ?
                """,
                parameters,
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            RetrievalResult(document=document_from_row(row), score=-row["relevance"])
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

    async def delete(self, document_ids: Sequence[str]) -> None:
        ids = [(document_id,) for document_id in dict.fromkeys(document_ids)]
        if not ids:
            return

        async with self._operation_lock:
            connection = await self._ensure_connection()
            try:
                await connection.executemany(
                    f"DELETE FROM {_DOCUMENTS} WHERE id = ?", ids
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

    async def __aenter__(self) -> "SQLiteFts5Index":
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
            raise RuntimeError("SQLiteFts5Index is closed")
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
            await connection.executescript(document_table_statement(_DOCUMENTS))
            await connection.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {_SEARCH} USING fts5(
                    text,
                    content={_DOCUMENTS},
                    content_rowid=rowid,
                    tokenize='{self._tokenizer}'
                )
                """
            )
            await connection.executescript(_synchronization_triggers())
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise


def _match_expression(query: str) -> str | None:
    """Turn a user query into an FTS5 OR expression over its bare terms.

    Query text is never passed to MATCH directly: it would be parsed as FTS5
    syntax, so a stray quote or a word like NOT would change the query or make
    it fail. Matching any term rather than all of them preserves recall. Longer
    terms use FTS5 prefix matching so queries such as ``Fahrrad`` also find
    German compounds such as ``Fahrradlicht``.
    """
    terms = _QUERY_TERMS.findall(query)
    if not terms:
        return None
    return " OR ".join(_match_term(term) for term in terms)


def _match_term(term: str) -> str:
    suffix = "*" if len(term) >= _MINIMUM_PREFIX_LENGTH else ""
    return f'"{term}"{suffix}'


def _synchronization_triggers() -> str:
    """Keep the FTS5 shadow tables in step with the content table.

    An external-content FTS5 table stores no copy of the text, so every
    write to the content table has to be mirrored into the index. Doing it
    with triggers means the upsert in index() and the plain DELETE in
    delete() need no FTS5-specific code.
    """
    return f"""
        CREATE TRIGGER IF NOT EXISTS {_DOCUMENTS}_after_insert
        AFTER INSERT ON {_DOCUMENTS} BEGIN
            INSERT INTO {_SEARCH}(rowid, text) VALUES (new.rowid, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS {_DOCUMENTS}_after_delete
        AFTER DELETE ON {_DOCUMENTS} BEGIN
            INSERT INTO {_SEARCH}({_SEARCH}, rowid, text)
            VALUES ('delete', old.rowid, old.text);
        END;

        CREATE TRIGGER IF NOT EXISTS {_DOCUMENTS}_after_update
        AFTER UPDATE ON {_DOCUMENTS} BEGIN
            INSERT INTO {_SEARCH}({_SEARCH}, rowid, text)
            VALUES ('delete', old.rowid, old.text);
            INSERT INTO {_SEARCH}(rowid, text) VALUES (new.rowid, new.text);
        END;
    """
