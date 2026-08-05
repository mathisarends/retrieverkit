import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from retrieval import Document

DOCUMENT_COLUMNS = "id, parent_id, text, metadata, created_at"

type DocumentRow = tuple[str, str | None, str, str, str]


async def open_database(database: str | Path) -> aiosqlite.Connection:
    connection = await aiosqlite.connect(database)
    connection.row_factory = sqlite3.Row
    await connection.execute("PRAGMA journal_mode = WAL")
    return connection


def document_table_statement(table: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            parent_id TEXT,
            text TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS {table}_parent_id ON {table}(parent_id);
    """


def upsert_document_statement(table: str) -> str:
    return f"""
        INSERT INTO {table} ({DOCUMENT_COLUMNS})
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            parent_id = excluded.parent_id,
            text = excluded.text,
            metadata = excluded.metadata
    """


def document_rows(documents: Sequence[Document]) -> list[DocumentRow]:
    """Serialize documents, timestamping any that carry no creation time.

    Documents indexed in one batch are spaced a microsecond apart so that
    ordering by creation time reproduces the order they were passed in.
    """
    indexed_at = datetime.now(UTC)
    return [
        (
            document.id,
            document.parent_id,
            document.text,
            serialize_metadata(document.metadata),
            serialize_datetime(
                document.created_at or indexed_at + timedelta(microseconds=position)
            ),
        )
        for position, document in enumerate(documents)
    ]


def document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        parent_id=row["parent_id"],
        text=row["text"],
        metadata=deserialize_metadata(row["metadata"]),
        created_at=deserialize_datetime(row["created_at"]),
    )


def aliased_document_columns(alias: str) -> str:
    return ", ".join(f"{alias}.{column}" for column in DOCUMENT_COLUMNS.split(", "))


def serialize_metadata(metadata: Mapping[str, object]) -> str:
    try:
        return json.dumps(dict(metadata), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "metadata must contain only JSON-serializable values"
        ) from error


def deserialize_metadata(value: str | None) -> Mapping[str, object]:
    if value is None:
        return {}
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise ValueError("Stored document metadata must be a JSON object")
    return metadata


def serialize_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("created_at must include timezone information")
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def deserialize_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)
