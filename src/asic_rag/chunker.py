"""Fixed-size character chunking with overlap and metadata propagation."""

import json
from dataclasses import dataclass
from pathlib import Path

from asic_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, CORPUS_PATH


@dataclass
class Chunk:
    chunk_id: str       # "{doc_id}::chunk_{i}"
    text: str
    doc_id: str
    metadata: dict      # parent document metadata, copied verbatim


def chunk_document(
    doc: dict,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a single document into fixed-size character chunks with overlap."""
    content = doc["content"]
    doc_id = doc["id"]
    metadata = doc["metadata"]

    if len(content) <= chunk_size:
        return [Chunk(chunk_id=f"{doc_id}::chunk_0", text=content, doc_id=doc_id, metadata=metadata)]

    chunks = []
    start = 0
    i = 0
    step = chunk_size - overlap
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::chunk_{i}",
            text=content[start:end],
            doc_id=doc_id,
            metadata=metadata,
        ))
        start += step
        i += 1
    return chunks


def load_and_chunk_corpus(corpus_path: Path = CORPUS_PATH) -> list[Chunk]:
    """Load corpus.jsonl and return all chunks."""
    chunks = []
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            chunks.extend(chunk_document(doc))
    return chunks
