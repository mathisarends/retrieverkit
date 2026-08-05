# retrieval

Provider-agnostic indexing and retrieval for Python.

`retrieval` gives you the building blocks of a RAG pipeline — chunking, dense
retrieval, lexical retrieval and rank fusion — behind small abstract base
classes you can implement yourself. The core package has **no dependencies**;
embedding providers are optional extras.

```python
hybrid = ReciprocalRankFusion([vector_index, lexical_index])
results = await hybrid.retrieve("How do I talk to Cara?", limit=5)
```

---

## Table of contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Concepts](#concepts)
  - [Ports](#ports)
  - [Types](#types)
- [Chunking](#chunking)
  - [Strategies](#strategies)
  - [Token-aware sizing](#token-aware-sizing)
- [Dense retrieval](#dense-retrieval)
- [Persistent SQLite retrieval](#persistent-sqlite-retrieval)
  - [Persistent lexical retrieval](#persistent-lexical-retrieval)
  - [Caching embeddings](#caching-embeddings)
- [Lexical retrieval](#lexical-retrieval)
  - [BM25 parameters](#bm25-parameters)
  - [Tokenization, stemming and stop words](#tokenization-stemming-and-stop-words)
- [Hybrid retrieval](#hybrid-retrieval)
  - [Weighting retrievers](#weighting-retrievers)
  - [Over-fetching](#over-fetching)
- [Reranking](#reranking)
- [Embedding providers](#embedding-providers)
  - [OpenAI](#openai)
  - [Writing your own](#writing-your-own)
- [Extending the package](#extending-the-package)
- [Scope and limitations](#scope-and-limitations)
- [Examples](#examples)
- [Development](#development)

---

## Installation

```bash
pip install retrieval
```

The core package pulls in nothing. Install an extra if you want a bundled
embedding provider:

```bash
pip install "retrieval[openai]"
```

Requires Python 3.13.

---

## Quickstart

```python
import asyncio

from retrieval import Document
from retrieval.chunking import MarkdownChunker
from retrieval.lexical import InMemoryBM25Index


async def main() -> None:
    source = Document(id="cara", text=open("docs/cara.md").read())

    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
    chunks = [
        Document(id=f"{source.id}:{number}", text=text)
        for number, text in enumerate(chunker.chunk(source.text), start=1)
    ]

    index = InMemoryBM25Index()
    await index.index(chunks)

    for result in await index.retrieve("wake word detection", limit=3):
        print(f"{result.score:.3f}  {result.document.text[:60]}")


asyncio.run(main())
```

`InMemoryBM25Index` needs no API key and no model — it is the fastest way to
get a working retrieval loop before you decide on an embedding provider.

---

## Concepts

The package is built around four abstract base classes. Everything else is an
implementation of one of them, and every implementation is interchangeable.

### Ports

```python
from retrieval import Chunker, EmbeddingProvider, LexicalTokenizer, Retriever, TextIndex
```

| Port | Method | Purpose |
| --- | --- | --- |
| `Chunker` | `chunk(text) -> list[str]` | Split a document into indexable pieces. |
| `EmbeddingProvider` | `async embed(texts) -> list[Embedding]` | Turn text into dense vectors. |
| `LexicalTokenizer` | `tokenize(text) -> list[str]` | Split text into the terms a lexical index matches on. |
| `Retriever` | `async retrieve(query, *, limit) -> list[RetrievalResult]` | Anything that answers a query with ranked documents. |
| `TextIndex` | `Retriever` + `async index(documents)`, `async delete(document_ids)` | A retriever you can also write to. |

`TextIndex` extends `Retriever`, so an index and a fusion layer are the same
kind of thing to a caller. That is what makes [hybrid
retrieval](#hybrid-retrieval) composable.

`index()` upserts and `delete()` ignores unknown ids, which is what an index
kept in sync with changing source files needs: re-index what changed, delete
the ids that disappeared, and neither call has to know what the index already
holds.

### Types

```python
from retrieval import Document, Embedding, RetrievalResult
```

```python
Embedding = list[float]


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    parent_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    document: Document
    score: float
```

`Document.id` is the identity used for deduplication: re-indexing an existing
id replaces the document in every index, and fusion merges results by id.
`parent_id` can link a chunk or message back to its source document, while
`metadata` carries application-specific JSON data. Persistent indexes populate
`created_at`; in-memory indexes leave it untouched.

Scores are **not comparable across retrievers.** Cosine similarity lands
around `0.9`, BM25 around `12`, reciprocal rank fusion around `0.03`. Use them
to rank within one retriever, never to threshold across several.

---

## Chunking

```python
from retrieval.chunking import FixedSizeChunker, MarkdownChunker, RecursiveChunker
```

### Strategies

| Chunker | Behaviour | Use when |
| --- | --- | --- |
| `FixedSizeChunker` | Hard token windows with optional overlap. | Content has no structure worth preserving. |
| `RecursiveChunker` | Splits on the first separator that fits — paragraph, line, sentence, word — then falls back to a hard boundary. | General prose. |
| `MarkdownChunker` | Keeps Markdown sections intact, recursively splitting oversized ones. Heading-like lines inside fenced code blocks do not start a new section. | Docs, READMEs, wikis. |

```python
chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(text)
```

`RecursiveChunker` accepts custom separators, applied in order:

```python
RecursiveChunker(500, chunk_overlap=50, separators=("\n\n", "\n", "; ", " "))
```

### Token-aware sizing

`chunk_size` is measured in *tokens*, and a token is whatever the injected
tokenizer says it is. The default `CharacterTokenizer` counts Unicode code
points, which keeps the core dependency-free.

To size chunks by real model tokens, pass any object implementing
`TextTokenizer`:

```python
from collections.abc import Sequence

import tiktoken

from retrieval.chunking import RecursiveChunker, TextTokenizer


class TiktokenTokenizer(TextTokenizer):
    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> Sequence[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))


chunker = RecursiveChunker(500, chunk_overlap=50, tokenizer=TiktokenTokenizer())
```

---

## Dense retrieval

```python
from retrieval.vector import InMemoryVectorIndex
```

`InMemoryVectorIndex` embeds documents on `index()` and ranks by cosine
similarity. Vectors are normalized on write, so retrieval is a dot product.

```python
index = InMemoryVectorIndex(embedding_provider)
await index.index(documents)

results = await index.retrieve("how does the assistant listen?", limit=5)
```

Dense retrieval matches *meaning*: it finds the paraphrase that shares no
words with the query. It fails on terms the embedding model never saw —
product names, identifiers, error codes, version numbers. That is exactly what
[lexical retrieval](#lexical-retrieval) is good at.

All documents in an index must share one embedding dimensionality;
`index()` raises `ValueError` otherwise.

---

## Persistent SQLite retrieval

Install the optional SQLite backend together with your embedding provider:

```bash
pip install "retrieval[openai,sqlite]"
```

`SQLiteVectorIndex` implements the same `TextIndex` port as the in-memory
index, but persists documents and cosine-searchable vectors in one SQLite
file. It uses `sqlite-vec`; the embedding model stays outside the database.

```python
from retrieval import Document
from retrieval.openai import OpenAIEmbeddingProvider
from retrieval.sqlite import SQLiteVectorIndex

provider = OpenAIEmbeddingProvider("text-embedding-3-small")

async with SQLiteVectorIndex("cara.db", provider, dimensions=1536) as index:
    await index.index(
        [
            Document(
                id="conversation-1",
                text="Planning the SQLite memory experiment",
                metadata={"kind": "conversation"},
            ),
            Document(
                id="message-1",
                parent_id="conversation-1",
                text="How should Cara remember earlier conversations?",
                metadata={"kind": "message", "role": "user"},
            ),
        ]
    )

    previous_messages = await index.list_documents(parent_id="conversation-1")
    matches = await index.search(
        "What did we decide about memory?",
        parent_id="conversation-1",
        limit=5,
    )
```

The normal table is deliberately plain SQLite and is the source of truth:

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    text TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Vectors live in the separate `document_embeddings` virtual table. This avoids
coupling conversation data to one embedding model and leaves the `documents`
table easy to inspect or migrate. Re-indexing an id updates its contents and
vector while preserving its original `created_at` timestamp.

Besides `index()` and `retrieve()`, the persistent index provides:

- `get(document_id)` to load one document;
- `list_documents()` to list root documents newest first;
- `list_documents(parent_id=...)` to reconstruct ordered children;
- `search(..., parent_id=...)` to search within one conversation or source;
- `delete(document_ids)` to remove documents and their vectors atomically.

The embedding dimensions are part of the vector-table schema. Open an existing
database with the same `dimensions` value and embedding model configuration.

Database operations run asynchronously via `aiosqlite`; connection setup and
teardown are asynchronous as well, so use `async with` or `await index.close()`.
The connection runs in WAL mode, so a process reading the index does not block
the process writing to it — which is what an index kept current by a file
watcher needs.

### Persistent lexical retrieval

`SQLiteFts5Index` is the lexical counterpart, built on SQLite's FTS5
extension. It needs no embedding provider; the `sqlite` extra installs its
asynchronous SQLite driver:

```python
from retrieval.sqlite import SQLiteFts5Index

async with SQLiteFts5Index("cara.db") as index:
    await index.index(documents)
    matches = await index.search("wakewordkit setup", parent_id="daily-notes")
```

Text lives in a `lexical_documents` table and is mirrored into an
external-content FTS5 table by triggers, so `index()` and `delete()` stay
ordinary SQL writes and the search index can never drift from the content.
Both persistent indexes can share one database file — their tables do not
overlap.

Two differences from `InMemoryBM25Index` are worth knowing:

- **Tokenization happens inside SQLite,** so it is configured with an FTS5
  tokenizer string rather than a `LexicalTokenizer`. The default is
  `unicode61 remove_diacritics 2`. Pass `tokenizer="porter unicode61"` for
  English stemming, or a custom one you registered on the connection.
- **FTS5 owns the BM25 implementation,** so `k1` and `b` are not tunable.

Queries are never handed to FTS5 verbatim. The words in a query are extracted
and recombined into an `OR` expression, which keeps FTS5 operator syntax in a
user's query from changing or breaking it, and matches the any-term ranking
`InMemoryBM25Index` does.

### Caching embeddings

Re-indexing a corpus re-embeds every chunk, including the ones that did not
change. `SQLiteEmbeddingCache` is an `EmbeddingProvider` that wraps another
one and remembers its results, so a re-index costs API calls only for text
that is actually new:

```python
from retrieval.openai import OpenAIEmbeddingProvider
from retrieval.sqlite import SQLiteEmbeddingCache, SQLiteVectorIndex

provider = SQLiteEmbeddingCache(
    "cara.db",
    OpenAIEmbeddingProvider("text-embedding-3-small"),
    namespace="text-embedding-3-small",
)
index = SQLiteVectorIndex("cara.db", provider, dimensions=1536)
```

Both objects own asynchronous database connections. Close them with
`await index.close()` and `await provider.close()`, or manage each with
`async with`.

Because it is just an `EmbeddingProvider`, anything that takes one benefits —
the vector index, an [MMR reranker](#reranking), your own code.

`namespace` is required and keys the cache alongside a hash of the text.
Change it whenever the vectors would change: a different model, different
`dimensions`, a different provider. Sharing a namespace between two models
returns one model's vectors for the other's queries, and nothing detects it.

Vectors are stored as 32-bit floats, matching what embedding APIs return and
what `sqlite-vec` stores. A cached vector can therefore differ from a freshly
computed one in the last digits.

---

## Lexical retrieval

```python
from retrieval.lexical import InMemoryBM25Index, WordTokenizer
```

`InMemoryBM25Index` implements Okapi BM25 with Lucene's inverse document
frequency. It matches exact terms, needs no model, no API key and no network.

```python
index = InMemoryBM25Index()
await index.index(documents)

results = await index.retrieve("wakewordkit", limit=5)
```

Documents that share no term with the query are omitted rather than returned
with a zero score — this keeps irrelevant documents out of the rankings that
[fusion](#hybrid-retrieval) consumes.

Document frequencies are maintained incrementally, so re-indexing a document
under an existing id correctly retires the terms of the previous version.

### BM25 parameters

```python
InMemoryBM25Index(k1=1.5, b=0.75)
```

| Parameter | Default | Effect |
| --- | --- | --- |
| `k1` | `1.5` | Term-frequency saturation. Lower means repeated terms stop helping sooner. `0` reduces scoring to presence/absence. |
| `b` | `0.75` | Length normalization, from `0` (ignore document length) to `1` (fully normalize). Lower it when your chunks vary wildly in length for legitimate reasons. |

The defaults are the standard ones and are a reasonable starting point for
most corpora. Tune them against a labelled query set, not by intuition.

### Tokenization, stemming and stop words

BM25 matches terms literally, so the tokenizer decides what counts as a match.
The default `WordTokenizer` splits on Unicode word boundaries and case-folds:

```python
WordTokenizer().tokenize("Grüße, Welt! 42")
# ['grüsse', 'welt', '42']
```

It deliberately does **no stemming and removes no stop words.** Both are
language-specific, and shipping a German or English word list in a
general-purpose package would silently degrade every other language. Instead,
they are a one-method extension point — implement `LexicalTokenizer` and pass
it in (this example needs `pip install snowballstemmer`):

```python
import re

import snowballstemmer

from retrieval import LexicalTokenizer

STOP_WORDS = {"der", "die", "das", "und", "ist", "ein", "eine"}


class GermanTokenizer(LexicalTokenizer):
    """Case-folded word tokens, stop-word filtered and Snowball-stemmed."""

    _WORDS = re.compile(r"\w+", re.UNICODE)

    def __init__(self) -> None:
        self._stemmer = snowballstemmer.stemmer("german")

    def tokenize(self, text: str) -> list[str]:
        words = self._WORDS.findall(text.casefold())
        return [self._stemmer.stemWord(word) for word in words if word not in STOP_WORDS]


index = InMemoryBM25Index(GermanTokenizer())
```

With stemming, a query for `Spracherkennung` matches a document containing
`Spracherkennungen`; without it, the two are unrelated terms.

Use the **same tokenizer for indexing and querying.** The index applies its
tokenizer to both, so this holds automatically — but if you swap the tokenizer
on a populated index, re-index everything.

---

## Hybrid retrieval

```python
from retrieval.fusion import ReciprocalRankFusion
```

Dense and lexical retrieval fail in different places, which makes them worth
combining. Their scores are not comparable, so `ReciprocalRankFusion` combines
their *rankings* instead: each retriever contributes
`weight / (rank_constant + rank)` per document, and the sums are re-sorted.

```python
hybrid = ReciprocalRankFusion([vector_index, lexical_index])
results = await hybrid.retrieve("wakewordkit setup", limit=5)
```

`ReciprocalRankFusion` is itself a `Retriever`, so it accepts any mix of
indexes, remote services and other fusions — and can be nested.

Retrievers are queried concurrently via `asyncio.gather`.

| Parameter | Default | Effect |
| --- | --- | --- |
| `weights` | equal | One positive weight per retriever, scaling everything that retriever contributes. |
| `rank_constant` | `60` | Dampens the influence of top ranks. The standard value from the original RRF paper. Lower values make rank 1 dominate. |
| `candidates_per_retriever` | `60` | Lower bound on how many results to request from each retriever. Depth matters: a document ranked 40th by both retrievers can only outrank a one-sided winner if both lists reach that far. |
| `candidate_multiplier` | `1` | Over-fetch relative to the requested limit. Each retriever is asked for `max(limit * candidate_multiplier, candidates_per_retriever)`. |

A document found by one retriever still surfaces — fusion rewards agreement
without requiring it.

### Weighting retrievers

Weights let you say that one retriever is generally more trustworthy for your
corpus without letting it decide alone:

```python
hybrid = ReciprocalRankFusion([vector_index, lexical_index], weights=[0.7, 0.3])
```

Weighting happens on the rank contributions, not on retriever scores, so it
stays immune to the scale problem that motivates rank fusion in the first
place. What the numbers control is how much one retriever's opinion is worth
relative to another's — only their ratio matters, so `[0.7, 0.3]` and `[7, 3]`
behave identically.

Two retrievers that agree still outrank a single weighted favourite, which is
usually what you want: with `[0.7, 0.3]`, a document ranked first by the
lexical index alone scores `0.0049`, while one ranked second by both scores
`0.0161`.

### Over-fetching

`candidate_multiplier` scales the over-fetch with the request rather than
pinning it to a constant. It matters when a stage after the fusion narrows the
results further — a [reranker](#reranking) that sees only five candidates
cannot do much:

```python
hybrid = ReciprocalRankFusion(
    [vector_index, lexical_index],
    candidates_per_retriever=1,
    candidate_multiplier=4,
)
```

Keep `candidates_per_retriever` as the floor for small limits and let the
multiplier take over for large ones; the effective depth is the larger of the
two.

---

## Reranking

```python
from retrieval.rerank import MaximalMarginalRelevance
```

Ranking by relevance alone fills the top places with near duplicates: the same
section from three revisions of a file, or three overlapping chunks of one
paragraph. Every one of them is a correct answer, and together they say one
thing where the caller asked for five.

`MaximalMarginalRelevance` wraps any `Retriever`, over-fetches, and picks
results one at a time. Each pick is scored on how well it answers the query
*minus* how much it repeats what is already picked:

```python
diverse = MaximalMarginalRelevance(hybrid, embedding_provider, relevance=0.7)
results = await diverse.retrieve("what did we decide about memory?", limit=5)
```

| Parameter | Default | Effect |
| --- | --- | --- |
| `relevance` | `0.5` | Weight of relevance against diversity. `1.0` ranks purely by similarity to the query; `0.0` purely by novelty. |
| `candidates` | `30` | Lower bound on how many results to re-rank. |
| `candidate_multiplier` | `4` | Over-fetch relative to the requested limit, as in [fusion](#over-fetching). |

Since it is a `Retriever`, it composes in either direction: put it over a
fusion to diversify a hybrid ranking, or under one to feed a fusion pre-thinned
candidates.

Two things are worth knowing before you reach for it:

- **It embeds the candidates.** Relevance and redundancy are measured between
  vectors, and the upstream retriever's scores cannot supply that — an RRF
  score says nothing about what a document is about. One extra embedding call
  per query is the cost, which a [cache](#caching-embeddings) largely absorbs
  for text that keeps resurfacing.
- **The returned scores are MMR scores,** not the upstream retriever's. They
  explain the order you got and, like every score in this package, mean
  nothing outside it. Diversity penalties can push them negative.

Install the extra, then:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from dotenv import load_dotenv

from retrieval.openai import OpenAIEmbeddingProvider

load_dotenv()

provider = OpenAIEmbeddingProvider("text-embedding-3-small")
```

With no key argument, the official OpenAI client reads `OPENAI_API_KEY` from
the environment. Pass the key directly if you obtain it from a secret manager
or another source. The provider itself does not parse `.env` files; the
examples load them explicitly with `python-dotenv` before constructing it.

```python
provider = OpenAIEmbeddingProvider(
    "text-embedding-3-small",
    api_key=api_key,
)
```

Known model ids are described by the `OpenAIEmbeddingModel` literal type. The
argument also accepts any model id as a plain string, so new models work before
the type alias catches up. Shorten vectors with `dimensions=` where the model
supports it:

```python
OpenAIEmbeddingProvider(
    "text-embedding-3-large",
    dimensions=512,
)
```

Pass `client=` to reuse a configured `AsyncOpenAI` instance — for Azure,
a proxy, or custom retry and timeout settings.

### Writing your own

Any embedding model fits behind one method. The provider must return one
vector per input, in input order:

```python
from collections.abc import Sequence

from retrieval import Embedding, EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return self._model.encode(list(texts)).tolist()
```

---

## Extending the package

Every part of the pipeline is replaceable through the port it implements:

- **A vector database** — implement `TextIndex` over Qdrant, pgvector or
  Elasticsearch. It drops straight into `ReciprocalRankFusion` next to the
  in-memory indexes.
- **A reranker** — implement `Retriever`, wrap another retriever, over-fetch
  and re-sort. Callers cannot tell the difference.
- **A remote search API** — implement `Retriever` and it becomes fusible with
  your local indexes.
- **A chunker** — implement `Chunker` for code-aware, semantic or
  layout-aware splitting.

```python
from retrieval import RetrievalResult, Retriever


class CrossEncoderReranker(Retriever):
    def __init__(self, retriever: Retriever, *, candidates: int = 50) -> None:
        self._retriever = retriever
        self._candidates = candidates

    async def retrieve(self, query: str, *, limit: int = 5) -> list[RetrievalResult]:
        results = await self._retriever.retrieve(query, limit=self._candidates)
        return _rerank(query, results)[:limit]
```

---

## Scope and limitations

Be deliberate about what this package is:

- **The default bundled indexes are in-memory.** Use `SQLiteVectorIndex` when
  state must survive a restart.
- **The in-memory indexes score linearly over every document.** `InMemoryBM25Index` has no
  inverted-index posting lists, and `InMemoryVectorIndex` has no ANN structure.
  Both are fine into the low tens of thousands of chunks and are the wrong tool
  above that — move to a real store behind the same `TextIndex` port.
- **The indexes are not concurrency-safe.** Serialize writes yourself if you
  index from multiple tasks. The SQLite indexes run in WAL mode, so separate
  processes can read while one writes, but a single connection must not be
  shared across threads.
- **No query expansion** is included, and reranking goes only as far as
  [MMR](#reranking) — a cross-encoder is yours to add behind `Retriever`.
- **Filtering stops at `parent_id`.** The SQLite indexes can scope a search to
  one parent; anything else you store in `metadata` is yours to filter on
  after retrieval.
- **Nothing here watches your files.** Deciding what changed and calling
  `index()` and `delete()` is the application's job; the package only makes
  those two calls cheap to repeat.

The ports are the stable part; the in-memory implementations are reference
implementations that keep the package honest and dependency-free.

---

## Examples

Runnable scripts live in [`examples/`](examples):

| Script | Shows |
| --- | --- |
| `chunking_strategies.py` | All three chunkers on the same input, plus the tokenizer API. |
| `index_and_retrieve.py` | Chunk, embed and query with the OpenAI provider. |
| `hybrid_retrieval.py` | Vector and BM25 indexes merged with reciprocal rank fusion. |
| `sqlite_conversations.py` | Persist and search conversation messages with SQLite. |
| `persistent_hybrid_memory.py` | A note archive kept in step with its files: both SQLite indexes, the embedding cache, weighted fusion and MMR. |

The OpenAI examples need `OPENAI_API_KEY`.

---

## Development

```bash
uv sync --all-extras
uv run pytest packages/retrieval/tests
uv run ruff check retrieval
uv run ruff format retrieval
```
