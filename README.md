# ASIC Design RAG System

Retrieval-augmented generation system for answering ASIC design debugging
questions. Built for ENMGT 5400 (Applications of AI for Engineering Managers)
final project at Cornell.

## Overview

The system ingests design artifacts from an mflowgen ASIC build (RTL, synthesis
reports, timing analysis, power reports, DRC logs) and answers natural language
questions about the design by retrieving relevant context and generating
grounded answers with source citations.

Corpus source: Zeppelin/Blimp superscalar RISC-V processor tapeout targeting
FreePDK 45nm, with 6 design variants across 11 flow stages.

## Stack

- **Embedding**: OpenAI `text-embedding-3-small` (512 dimensions)
- **Vector store**: ChromaDB (persistent, cosine similarity)
- **LLM**: Claude Sonnet via Anthropic SDK
- **No frameworks** (no LangChain) -- just the SDKs directly

## Setup

```bash
uv sync
# Create .env in project root with OPENAI_API_KEY and ANTHROPIC_API_KEY
```

## Usage

All commands are run from within a build directory. Outputs (corpus, vector
store, eval results) go into the current working directory.

```bash
mkdir build && cd build

# 1. Assemble corpus from mflowgen build directory
uv run corpus /path/to/mflowgen/build-dir

# 2. Ingest (chunk + embed + store)
uv run ingest                         # fixed-size chunking (baseline)
uv run ingest --chunker structured    # structure-aware chunking

# 3. Query
uv run query "What is the setup slack for BlimpV11_2fe_2be?"
uv run query -i                       # interactive mode
uv run query --retrieve-only "DRC violations"      # debug retrieval
uv run query --filter-variant "..."                 # auto-filter by variant
uv run query --rerank "..."                         # re-rank with Claude
uv run query --filter-variant --rerank "..."        # both

# 4. Evaluate against benchmark (copy benchmark.jsonl into build dir first)
cp ../benchmark.jsonl .

# To regenerate all results (requires re-ingesting for chunker changes):
uv run ingest --chunker fixed
uv run eval -o eval_results_full.json                                    # fixed baseline

uv run ingest --chunker structured
uv run eval -o eval_results_structured.json                              # structured
uv run eval --filter-variant -o eval_results_filter.json                 # + variant filter
uv run eval --filter-variant --rerank -o eval_results_filter_rerank.json # + rerank
uv run eval --retrieval-only -o eval_results_retrieval.json              # retrieval only
```

Note: `--filter-variant` and `--rerank` are query-time flags that work on
whatever chunking strategy is currently in the vector store. To switch chunking
strategies, re-run `ingest` with the desired `--chunker` flag.

## Project structure

```
src/asic_rag/
  config.py               # Shared constants (models, chunk params, paths)
  corpus.py               # Corpus assembly from mflowgen build directory
  chunker.py              # Fixed-size character chunking (baseline)
  chunker_structured.py   # Structure-aware chunking (experiment)
  ingest.py               # Embedding + ChromaDB storage
  query.py                # Retrieval + generation CLI
  eval.py                 # Benchmark evaluation framework
benchmark.jsonl           # 30 evaluation questions with ground-truth answers
proposal.tex              # Project proposal
```

Build directory (generated):
```
build/
  corpus.jsonl            # Assembled corpus
  vectorstore/            # ChromaDB persistent storage
  eval_results.json       # Evaluation output
```

## Evaluation results

Five configurations compared on 30 benchmark questions:

| Strategy | Retrieval Recall | Correctness (/3) | Completeness (/3) |
|---|---|---|---|
| Fixed-size (baseline) | 0.38 | 1.40 | 1.80 |
| Structured v1 | 0.35 | 1.77 | 2.17 |
| Structured v2 (split headers) | 0.48 | 2.13 | 2.37 |
| Struct v2 + variant filter | 0.57 | 2.03 | 2.23 |
| Struct v2 + filter + rerank | 0.57 | 2.13 | 2.37 |
