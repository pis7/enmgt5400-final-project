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


class IngestExecutor:
    def __init__(self):
        self.chunker: str = "fixed"
        self.corpus_path = CORPUS_PATH
        self.chunks: list = []
        self.openai_client: OpenAI | None = None
        self.collection = None

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(description="Ingest corpus into vector store")
        parser.add_argument("--chunker", choices=["fixed", "structured"], default="fixed",
                            help="Chunking strategy (default: fixed)")
        args = parser.parse_args()
        self.chunker = args.chunker

    def load_and_chunk(self) -> None:
        """Load corpus and split into chunks using the selected strategy."""
        if self.chunker == "structured":
            from asic_rag.chunker_structured import load_and_chunk_corpus
        else:
            from asic_rag.chunker import load_and_chunk_corpus

        print(f"Loading and chunking corpus from {self.corpus_path} (chunker={self.chunker})...")
        self.chunks = load_and_chunk_corpus(self.corpus_path)
        print(f"  {len(self.chunks)} chunks created")

    def init_vectorstore(self) -> None:
        """Create (or recreate) the ChromaDB collection."""
        chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

        existing = chroma_client.list_collections()
        if "asic_debug" in existing:
            chroma_client.delete_collection("asic_debug")

        self.collection = chroma_client.create_collection(
            name="asic_debug",
            metadata={"hnsw:space": "cosine"},
        )

    def embed_and_store(self) -> None:
        """Embed all chunks and store them in the collection."""
        self.openai_client = OpenAI()
        total = len(self.chunks)
        batch_size = EMBEDDING_BATCH_SIZE

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = self.chunks[batch_start:batch_end]

            texts = [c.text for c in batch]
            response = self.openai_client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            embeddings = [item.embedding for item in response.data]

            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in batch],
            )

            print(f"  Ingested {batch_end}/{total} chunks", end="\r")
            time.sleep(1)  # stay under TPM rate limit

        print(f"\nDone. Collection '{self.collection.name}' has {self.collection.count()} chunks.")

def main():
    executor = IngestExecutor()
    executor.parse_args()
    executor.load_and_chunk()
    executor.init_vectorstore()
    executor.embed_and_store()


if __name__ == "__main__":
    main()
