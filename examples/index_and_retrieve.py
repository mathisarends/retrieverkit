import asyncio

from dotenv import load_dotenv

from retrieval import Document
from retrieval.chunking import MarkdownChunker
from retrieval.openai import OpenAIEmbeddingProvider
from retrieval.vector import InMemoryVectorIndex

load_dotenv(override=True)


async def main() -> None:
    embedding_provider = OpenAIEmbeddingProvider("text-embedding-3-small")
    index = InMemoryVectorIndex(embedding_provider)

    source = Document(
        id="cara",
        text="""# Bedienung
Cara ist ein sprachgesteuerter Assistent mit Terminal- und Voice-Sessions.

# Persönlichkeit
Cara antwortet freundlich, direkt und prägnant.
""",
    )
    chunker = MarkdownChunker(chunk_size=100, chunk_overlap=15)
    chunks = [
        Document(id=f"{source.id}:{number}", text=text)
        for number, text in enumerate(chunker.chunk(source.text), start=1)
    ]
    await index.index(chunks)

    results = await index.retrieve("Wie kann ich mit Cara sprechen?", limit=1)
    for result in results:
        print(f"{result.score:.3f}: {result.document.text}")


if __name__ == "__main__":
    asyncio.run(main())
