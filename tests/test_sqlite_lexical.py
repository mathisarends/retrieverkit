import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from retrieval import Document
from retrieval.sqlite import SQLiteFts5Index


@contextmanager
def _closing(index: SQLiteFts5Index) -> Iterator[SQLiteFts5Index]:
    try:
        yield index
    finally:
        asyncio.run(index.close())


def test_retrieve_ranks_documents_by_term_overlap(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "lexical.db")) as index:
        asyncio.run(
            index.index(
                [
                    Document(id="python", text="Python is a programming language."),
                    Document(id="berlin", text="Berlin is the capital of Germany."),
                ]
            )
        )

        results = asyncio.run(index.retrieve("programming language"))

    assert [result.document.id for result in results] == ["python"]
    assert results[0].score > 0.0


def test_index_survives_reopening_the_database(tmp_path: Path) -> None:
    database = tmp_path / "persistent.db"
    created_at = datetime(2026, 8, 2, 9, 30, tzinfo=UTC)
    document = Document(
        id="note",
        parent_id="daily",
        text="We talked about SQLite full text search.",
        metadata={"kind": "note"},
        created_at=created_at,
    )
    with _closing(SQLiteFts5Index(database)) as index:
        asyncio.run(index.index([document]))

    with _closing(SQLiteFts5Index(database)) as reopened:
        results = asyncio.run(reopened.retrieve("sqlite"))

    assert results[0].document == document


def test_reindexing_replaces_the_searchable_text(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "replace.db")) as index:
        asyncio.run(index.index([Document(id="note", text="the original wording")]))
        asyncio.run(index.index([Document(id="note", text="the replacement wording")]))

        assert asyncio.run(index.retrieve("original")) == []
        assert [
            result.document.id for result in asyncio.run(index.retrieve("replacement"))
        ] == ["note"]


def test_delete_removes_documents_from_the_search_index(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "delete.db")) as index:
        asyncio.run(index.index([Document(id="note", text="a deletable note")]))

        asyncio.run(index.delete(["note", "never-indexed"]))

        assert asyncio.run(index.retrieve("deletable")) == []
        assert asyncio.run(index.get("note")) is None


def test_search_can_be_limited_to_a_parent(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "filter.db")) as index:
        asyncio.run(
            index.index(
                [
                    Document(id="first", text="shared topic", parent_id="monday"),
                    Document(id="second", text="shared topic", parent_id="tuesday"),
                ]
            )
        )

        results = asyncio.run(index.search("shared", parent_id="tuesday"))

    assert [result.document.id for result in results] == ["second"]


def test_retrieve_matches_any_query_term(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "any.db")) as index:
        asyncio.run(index.index([Document(id="note", text="only about wakewords")]))

        results = asyncio.run(index.retrieve("wakewords and embeddings"))

    assert [result.document.id for result in results] == ["note"]


def test_retrieve_matches_long_query_terms_as_prefixes(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "prefix.db")) as index:
        asyncio.run(
            index.index(
                [
                    Document(id="tour", text="Eine Fahrradrunde am Kanal."),
                    Document(id="light", text="Das Fahrradlicht sitzt locker."),
                    Document(id="garden", text="Ein Besuch im botanischen Garten."),
                ]
            )
        )

        results = asyncio.run(index.retrieve("Fahrrad"))

    assert {result.document.id for result in results} == {"tour", "light"}


def test_retrieve_treats_fts5_operators_as_plain_words(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "operators.db")) as index:
        asyncio.run(index.index([Document(id="note", text="a note about retrieval")]))

        assert [
            result.document.id
            for result in asyncio.run(index.retrieve('retrieval OR "'))
        ] == ["note"]
        assert asyncio.run(index.retrieve("NOT")) == []


def test_retrieve_without_usable_query_terms_returns_nothing(tmp_path: Path) -> None:
    with _closing(SQLiteFts5Index(tmp_path / "empty.db")) as index:
        asyncio.run(index.index([Document(id="note", text="a note")]))

        assert asyncio.run(index.retrieve("!?-")) == []


def test_retrieve_rejects_non_positive_limit(tmp_path: Path) -> None:
    with (
        _closing(SQLiteFts5Index(tmp_path / "limit.db")) as index,
        pytest.raises(ValueError, match="limit must be at least 1"),
    ):
        asyncio.run(index.retrieve("query", limit=0))
