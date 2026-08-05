import asyncio

from dotenv import load_dotenv

from retrieval import Document
from retrieval.fusion import ReciprocalRankFusion
from retrieval.lexical import InMemoryBM25Index
from retrieval.openai import OpenAIEmbeddingProvider
from retrieval.vector import InMemoryVectorIndex

load_dotenv(override=True)

DOCUMENTS = [
    Document(
        id="voice", text="Cara listens for a wake word and answers over the speaker."
    ),
    Document(
        id="terminal", text="Cara can also run as a text session in the terminal."
    ),
    Document(
        id="wakewordkit", text="The wakewordkit package detects the activation phrase."
    ),
]


async def main() -> None:
    embedding_provider = OpenAIEmbeddingProvider("text-embedding-3-small")
    vector_index = InMemoryVectorIndex(embedding_provider)
    lexical_index = InMemoryBM25Index()
    await asyncio.gather(vector_index.index(DOCUMENTS), lexical_index.index(DOCUMENTS))

    hybrid = ReciprocalRankFusion([vector_index, lexical_index])

    # "wakewordkit" is an unknown token for the embedding model, so the lexical
    # index carries this query while the vector index carries paraphrases.
    for query in ("How do I talk to Cara?", "wakewordkit"):
        print(f"\n{query}")
        for result in await hybrid.retrieve(query, limit=2):
            print(f"{result.score:.4f}: {result.document.text}")


if __name__ == "__main__":
    asyncio.run(main())
