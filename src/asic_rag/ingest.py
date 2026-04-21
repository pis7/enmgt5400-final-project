"""Embed chunks and insert into ChromaDB. Idempotent: deletes and recreates the collection."""

import argparse
import time

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import chromadb

from asic_rag.config import (
    VECTORSTORE_DIR, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE, CORPUS_PATH,
)


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts."""
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def ingest(corpus_path=CORPUS_PATH, chunker: str = "fixed") -> None:
    """Full ingest pipeline: chunk -> embed -> store."""
    if chunker == "structured":
        from asic_rag.chunker_structured import load_and_chunk_corpus
    else:
        from asic_rag.chunker import load_and_chunk_corpus

    print(f"Loading and chunking corpus from {corpus_path} (chunker={chunker})...")
    chunks = load_and_chunk_corpus(corpus_path)
    print(f"  {len(chunks)} chunks created")

    chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Delete existing collection if present (makes re-runs idempotent)
    existing = chroma_client.list_collections()
    if "asic_debug" in existing:
        chroma_client.delete_collection("asic_debug")

    collection = chroma_client.create_collection(
        name="asic_debug",
        metadata={"hnsw:space": "cosine"},
    )

    openai_client = OpenAI()

    total = len(chunks)
    batch_size = EMBEDDING_BATCH_SIZE
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = chunks[batch_start:batch_end]

        texts = [c.text for c in batch]
        embeddings = embed_batch(openai_client, texts)

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in batch],
        )

        print(f"  Ingested {batch_end}/{total} chunks", end="\r")
        time.sleep(1)  # stay under TPM rate limit

    print(f"\nDone. Collection '{collection.name}' has {collection.count()} chunks.")


def main():
    parser = argparse.ArgumentParser(description="Ingest corpus into vector store")
    parser.add_argument("--chunker", choices=["fixed", "structured"], default="fixed",
                        help="Chunking strategy (default: fixed)")
    args = parser.parse_args()
    ingest(chunker=args.chunker)


if __name__ == "__main__":
    main()
