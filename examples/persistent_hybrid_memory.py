"""A note archive that survives restarts and stays in step with its files.

Run it twice. The second run re-reads the same notes and spends nothing on
embeddings, because the cache already holds every chunk. Then edit, shorten or
delete a file in NOTES_DIRECTORY and run it again.
"""

import asyncio
from datetime import UTC, date, datetime, time
from pathlib import Path

from dotenv import load_dotenv

from retrieval import Document, TextIndex
from retrieval.chunking import MarkdownChunker
from retrieval.fusion import ReciprocalRankFusion
from retrieval.openai import OpenAIEmbeddingProvider
from retrieval.rerank import MaximalMarginalRelevance
from retrieval.sqlite import SQLiteEmbeddingCache, SQLiteFts5Index, SQLiteVectorIndex

load_dotenv(override=True)

DATABASE = Path("notes.db")
NOTES_DIRECTORY = Path("notes")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

_UNLIMITED = 1_000_000

SAMPLE_NOTES = {
    "2026-07-28.md": """
        # Retrieval

        Decided the lexical index has to persist too, otherwise every restart
        rebuilds it from the source files.

        # Wake word

        wakewordkit misfires on the radio. Raising the threshold helped.
        """,
    "2026-07-30.md": """
        # Retrieval

        Reciprocal rank fusion is in. Weighting it 70/30 towards the vector
        index reads better on paraphrased questions.

        # Errands

        Bike service, and the OPENAI_API_KEY rotation is due.
        """,
    "2026-08-01.md": """
        # Retrieval

        Added an embedding cache. Re-indexing unchanged notes is free now.

        # Wake word

        Still misfires occasionally. Worth collecting the false positives.
        """,
}


async def main() -> None:
    _write_sample_notes()

    embedding_provider = SQLiteEmbeddingCache(
        DATABASE,
        OpenAIEmbeddingProvider(EMBEDDING_MODEL),
        namespace=EMBEDDING_MODEL,
    )
    vector_index = SQLiteVectorIndex(
        DATABASE, embedding_provider, dimensions=EMBEDDING_DIMENSIONS
    )
    lexical_index = SQLiteFts5Index(DATABASE)

    await synchronize(NOTES_DIRECTORY, [vector_index, lexical_index])

    # The vector index carries paraphrases, the lexical index carries the exact
    # identifiers an embedding model has never seen. The weights say which one
    # this corpus trusts more; MMR stops one loud note taking every place.
    search = MaximalMarginalRelevance(
        ReciprocalRankFusion(
            [vector_index, lexical_index], weights=[0.7, 0.3], candidate_multiplier=4
        ),
        embedding_provider,
        relevance=0.7,
    )

    for query in ("what did we decide about ranking?", "OPENAI_API_KEY"):
        print(f"\n{query}")
        for result in await search.retrieve(query, limit=3):
            print(
                f"  {result.score:+.4f}  {result.document.parent_id}  "
                f"{_summarize(result.document.text)}"
            )

    # The newest notes are wanted whole and in order, which is a database
    # query rather than a retrieval problem. Search is for everything older,
    # where relevance matters more than recency.
    print("\nthe two newest notes, loaded without searching")
    for note in await vector_index.list_documents(limit=2):
        print(f"  {note.id}  {_summarize(note.text)}")

    await vector_index.close()
    await lexical_index.close()
    await embedding_provider.close()


async def synchronize(directory: Path, indexes: list[TextIndex]) -> None:
    """Bring every index in line with what is on disk right now.

    This is the whole file-watcher story: chunk what is there, hand it to
    index(), and delete the ids nothing on disk accounts for any more. Both
    calls are idempotent, so it makes no difference whether this runs at
    startup, on a debounced watcher event, or on a timer.
    """
    chunker = MarkdownChunker(chunk_size=400, chunk_overlap=40)
    documents = [
        document
        for path in sorted(directory.glob("*.md"))
        for document in _chunk_note(path, chunker)
    ]
    stale = await _stale_ids(indexes[0], {document.id for document in documents})

    for index in indexes:
        await index.index(documents)
        await index.delete(stale)

    print(f"indexed {len(documents)} documents, removed {len(stale)}")


def _chunk_note(path: Path, chunker: MarkdownChunker) -> list[Document]:
    """Represent one file as a parent document plus one child per chunk."""
    text = path.read_text(encoding="utf-8")
    day = path.stem
    created_at = datetime.combine(date.fromisoformat(day), time.min, tzinfo=UTC)
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)

    note = Document(
        id=day,
        text=text,
        metadata={"path": str(path), "modified_at": modified_at.isoformat()},
        created_at=created_at,
    )
    chunks = [
        Document(
            id=f"{day}:{number}",
            parent_id=day,
            text=chunk,
            metadata={"path": str(path), "day": day},
            created_at=created_at,
        )
        for number, chunk in enumerate(chunker.chunk(text), start=1)
    ]
    return [note, *chunks]


async def _stale_ids(index: SQLiteVectorIndex, current: set[str]) -> list[str]:
    """Ids the database still holds that nothing on disk accounts for.

    Covers a deleted file and an edited one that now produces fewer chunks —
    the second is easy to forget, and leaves the tail of the old version
    answering queries forever.
    """
    stale: list[str] = []
    for note in await index.list_documents(limit=_UNLIMITED):
        chunks = await index.list_documents(parent_id=note.id, limit=_UNLIMITED)
        stale.extend(chunk.id for chunk in chunks if chunk.id not in current)
        if note.id not in current:
            stale.append(note.id)
    return stale


def _write_sample_notes() -> None:
    """Seed the directory once, then leave it alone so edits and deletions stick."""
    if NOTES_DIRECTORY.exists():
        return

    NOTES_DIRECTORY.mkdir()
    for name, body in SAMPLE_NOTES.items():
        (NOTES_DIRECTORY / name).write_text(_undent(body), encoding="utf-8")


def _undent(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines()) + "\n"


def _summarize(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= 70 else f"{collapsed[:67]}..."


if __name__ == "__main__":
    asyncio.run(main())
