# ASIC Design RAG System

RAG system for ASIC design debugging, built for ENMGT 5400 final project.

## Related project

The corpus comes from the Blimp processor project. See
`/work/global/pis7/blimp/CLAUDE.md` for RTL structure, build instructions,
and ASIC flow details.

Build directory used for corpus:
`/work/global/pis7/blimp/hw/top/asic/build-batch-1-20260415-194241-45/`

## Project layout

```
src/asic_rag/
  config.py               # Shared constants
  corpus.py               # Assembles corpus.jsonl from mflowgen build dir
  chunker.py              # Fixed-size chunking (baseline)
  chunker_structured.py   # Structure-aware chunking (experiment)
  ingest.py               # Embed chunks + store in ChromaDB
  query.py                # RAG query CLI (retrieve + generate)
  eval.py                 # Benchmark evaluation
benchmark.jsonl           # 30 eval questions with ground truth
```

## How to run

Uses `uv` for package management. Python 3.12 (uv-managed, not system).
API keys in `.env` (OPENAI_API_KEY, ANTHROPIC_API_KEY).

All commands run from a build directory; outputs go into cwd.

```bash
uv sync
mkdir build && cd build
uv run corpus /path/to/build-dir           # assemble corpus
uv run ingest --chunker structured         # embed + store
cp ../benchmark.jsonl .
uv run query "your question"               # query
uv run eval -o results.json                # evaluate
```

## CLI entry points

- `corpus` — assemble corpus from mflowgen build directory
- `ingest` — chunk, embed, and store in ChromaDB (`--chunker fixed|structured`)
- `query` — ask questions (`--filter-variant`, `--rerank`, `--retrieve-only`, `-i`)
- `eval` — run benchmark (`--filter-variant`, `--rerank`, `--retrieval-only`)

## Key design decisions

- Embedding: OpenAI text-embedding-3-small, 512 dimensions, cosine similarity
- Vector store: ChromaDB (persistent, file-based)
- LLM: Claude Sonnet for answer generation
- No LangChain -- direct SDK usage only
- Two chunking strategies for A/B comparison (fixed vs structure-aware)
- Evaluation separates retrieval quality from generation quality
