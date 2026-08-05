# Framework comparison

Kurze Notiz zu populären Projekten, die ähnliche Retrieval- und RAG-Use-Cases
abdecken. Vor größeren API-Entscheidungen lohnt sich ein Blick auf deren
Ansätze und Trade-offs.

- [Haystack](https://docs.haystack.deepset.ai/docs/retrievers): Store-Protocols,
  Filter, Duplicate Policies und Hybrid-Pipelines; wesentlich größeres
  Komponenten-Framework.
- [txtai](https://neuml.github.io/txtai/embeddings/query/): lokale Dense-,
  BM25- und Hybrid-Suche mit SQLite und SQL-Filtern; eher fertige Search-Engine.
- [LlamaIndex][llama-retriever]:
  Documents/Nodes, Chunk-Provenienz, Ingestion und viele Backends; sehr breite
  RAG-Abstraktion.
- [LangChain](https://docs.langchain.com/oss/python/langchain/retrieval): großes
  Adapter-Ökosystem; Retrieval ist nur ein Teil des Frameworks.
- [LanceDB](https://docs.lancedb.com/search/hybrid-search): eingebettete
  Vector-, Full-Text- und Hybrid-Suche; Backend statt allgemeiner Abstraktion.
- [Qdrant](https://qdrant.tech/documentation/search/hybrid-queries/):
  Collections, Filter, Dense/Sparse Vectors und RRF; primär Vector-Datenbank.
- [Weaviate](https://docs.weaviate.io/weaviate/search/hybrid): BM25F, Vektoren,
  Hybrid Fusion und Filter; vollständiger Search-Server.

## Was wir uns gezielt ansehen sollten

- **Haystack:** öffentliche Store-Verträge, Filtermodell und Adapter-Tests.
- **txtai:** kompakte lokale Hybrid-Suche und persistente Indexkonfiguration.
- **LlamaIndex:** stabile Chunk-Identität, Quellenbezug und Node-Beziehungen.
- **LangChain:** Adapter-Kompatibilität, nicht zwingend dessen Core-API kopieren.
- **LanceDB:** möglicher eingebetteter Backend-Adapter neben SQLite.
- **Qdrant:** Request-Modell, Filter-AST, Collections und mehrere Vektorräume.
- **Weaviate:** Score-Semantik, Hybrid-Gewichtung und Pre-Filtering.

## Mögliche Positionierung

`retrieval` kann bewusst kleiner bleiben: ein typisierter, async-fähiger und
provider-unabhängiger Retrieval-Core für Chunking, Embeddings, lokale Indizes,
Fusion und Reranking. Prompting, Agents, Loader und vollständige
RAG-Orchestrierung müssen nicht Teil des Kerns werden.

Die nächsten Designentscheidungen sollten besonders mit Haystack, txtai und
LlamaIndex gegengeprüft werden. LanceDB und Qdrant sind vor allem als spätere
Backends interessant.

[llama-retriever]: https://docs.llamaindex.ai/en/stable/module_guides/querying/
