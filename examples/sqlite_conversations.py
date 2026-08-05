import asyncio
from uuid import uuid4

from dotenv import load_dotenv

from retrieval import Document
from retrieval.openai import OpenAIEmbeddingProvider
from retrieval.sqlite import SQLiteVectorIndex

load_dotenv(override=True)


async def main() -> None:
    conversation_id = str(uuid4())
    documents = [
        Document(
            id=conversation_id,
            text="SQLite persistence experiment",
            metadata={"kind": "conversation"},
        ),
        Document(
            id=str(uuid4()),
            parent_id=conversation_id,
            text="Can Cara remember an earlier conversation?",
            metadata={"kind": "message", "role": "user"},
        ),
        Document(
            id=str(uuid4()),
            parent_id=conversation_id,
            text="Yes. Store each turn as a child document in SQLite.",
            metadata={"kind": "message", "role": "assistant"},
        ),
    ]

    provider = OpenAIEmbeddingProvider("text-embedding-3-small")
    async with SQLiteVectorIndex(
        "conversations.db", provider, dimensions=1536
    ) as index:
        await index.index(documents)

        print("Conversation:")
        for message in await index.list_documents(parent_id=conversation_id):
            print(f"[{message.metadata['role']}] {message.text}")

        print("\nMost relevant memory:")
        for result in await index.retrieve(
            "How is conversation history stored?", limit=1
        ):
            print(f"{result.score:.3f}: {result.document.text}")


if __name__ == "__main__":
    asyncio.run(main())
