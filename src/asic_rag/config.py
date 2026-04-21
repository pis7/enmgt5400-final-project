"""Shared constants for the RAG pipeline."""

from pathlib import Path

# Paths
CORPUS_PATH = Path("corpus.jsonl")
VECTORSTORE_DIR = Path("vectorstore")

# Chunking
CHUNK_SIZE = 1500       # characters
CHUNK_OVERLAP = 200     # characters

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
EMBEDDING_BATCH_SIZE = 32

# Retrieval
TOP_K = 10

# Generation
LLM_MODEL = "claude-sonnet-4-6"
MAX_CONTEXT_CHUNKS = 8
