# RAG for ASIC Design Debugging — Findings

## 1. Project Overview

This project builds and evaluates a retrieval-augmented generation (RAG) system
that answers ASIC design questions by retrieving relevant context from design
artifacts generated during a processor tapeout. The corpus comes from running
the Zeppelin/Blimp superscalar RISC-V processor through a full synthesis,
place-and-route, and signoff flow targeting the FreePDK 45nm open PDK.

### Corpus

- **Source**: asic build with 6 design variants (Proc6745, BlimpV8,
  BlimpV11_1fe_1be, BlimpV11_2fe_2be, BlimpV11_2fe_2be_ioiqd2,
  BlimpV11_2fe_2be_oooiqd2), each running through 11 flow stages
- **475 documents**, 25 MB total
- Document types: RTL source, SDC constraints, synthesis/PnR/STA timing reports,
  area reports, power reports, DRC/LVS logs, flow summaries, build configs

### Pipeline

```
ENTRY POINT 1: Building a new vectorstore
══════════════════════════════════════════

Build Directory (6 variants x 11 stages)
│
├── CorpusExecutor (uv run corpus)
│   ├── collect_documents ─── walks build dir, applies STAGE_RULES
│   └── write_corpus ──────── writes corpus.jsonl
│                                │
│                           corpus.jsonl (475 docs)
│                                │
└── IngestExecutor (uv run ingest --chunker fixed|structured)
    ├── load_and_chunk ────── splits docs via fixed or structured chunker
    ├── init_vectorstore ──── creates ChromaDB collection (deletes if exists)
    └── embed_and_store ───── embeds chunks via OpenAI, stores in ChromaDB
                                 │
                            ChromaDB vectorstore/


ENTRY POINT 2: Querying the existing vectorstore
═════════════════════════════════════════════════

User question
│
└── QueryExecutor (uv run query "...")
    ├── init_clients ──────── OpenAI + Anthropic + ChromaDB
    └── ask
        ├── extract_variant ── detects variant name in question (optional)
        ├── retrieve ─────────── embeds question, queries ChromaDB top-k
        ├── rerank ──────────── re-scores chunks with Claude (optional)
        ├── format_context ──── formats chunks into numbered context string
        └── generate_answer ─── sends question + context to Claude


ENTRY POINT 3: Evaluating the pipeline
═══════════════════════════════════════

benchmark.jsonl (30 questions with ground truth)
│
└── EvalExecutor (uv run eval)
    ├── init_clients ──────── OpenAI + Anthropic + creates QueryExecutor
    ├── load_benchmark ────── reads benchmark.jsonl
    └── run_eval ──────────── for each question:
        │   ├── QueryExecutor.retrieve
        │   ├── QueryExecutor.rerank (optional)
        │   ├── score_retrieval ──── recall vs expected source docs
        │   ├── QueryExecutor.generate_answer
        │   └── grade_answer ─────── Claude grades answer vs expected
        └── save_results ──────── writes eval_results.json
```

### Stack

- **Embedding**: OpenAI `text-embedding-3-small` (512 dimensions, cosine similarity)
- **Vector store**: ChromaDB (persistent, file-based, HNSW index)
- **LLM**: Claude Sonnet via Anthropic SDK
- **No frameworks** — direct SDK usage only (~900 lines of Python)

---

## 2. How RAG Works

RAG combines information retrieval with language model generation. Instead of
relying on an LLM's training data alone, it first retrieves relevant documents
from a knowledge base, then feeds those documents to the model as context.

### Embedding

Embedding converts text into a fixed-length vector (512 numbers) such that
semantically similar texts map to nearby points in vector space. Both corpus
chunks and user questions are embedded using the same model. At query time,
the system finds the stored vectors closest to the question vector (cosine
similarity), retrieves the original text from those matches, and passes it
to Claude as context for answer generation.

### Chunking

Documents must be split into chunks before embedding because:
1. Embedding models have input length limits
2. Smaller, focused chunks produce more specific embeddings that match
   targeted questions better than whole-document embeddings

### Vector Store

ChromaDB stores vectors alongside their original text and metadata. It uses
an HNSW (Hierarchical Navigable Small World) index for approximate nearest
neighbor search, enabling fast retrieval without comparing against every stored
vector.

### Generation

Claude receives the retrieved chunks in its system prompt with instructions to
answer based on the provided context and cite sources using `[Source N]`
notation. The model has no knowledge of the design on its own — it reads the
retrieved chunks and synthesizes an answer, similar to how an engineer would
answer a question given a handful of relevant report excerpts.

---

## 3. Evaluation Methodology

### Benchmark

30 questions across 4 categories, each with a known answer and expected source
documents:

| Category | Count | Description |
|---|---|---|
| simple-lookup | 12 | Single-fact retrieval (e.g., "What is the area of X?") |
| cross-document | 5 | Comparing metrics across flow stages |
| cross-variant | 7 | Comparing metrics across design variants |
| reasoning | 6 | Architectural understanding and explanation |

### Metrics (measured independently)

1. **Retrieval recall**: Did the top-k retrieved chunks include the expected
   source documents? Isolates retrieval quality from generation quality.

2. **Answer quality (LLM-graded)**: Claude acts as a grader, scoring generated
   answers against expected answers on two 0-3 scales:
   - **Correctness**: Does the answer contain the key facts?
   - **Completeness**: Does it address all parts of the question?

---

## 4. Experimental Results

### Configurations tested

1. **No context (LLM only)**: Claude answers with no retrieved context — measures
   baseline LLM knowledge
2. **Fixed chunking**: 1500-character sliding window, 200-char overlap
3. **Structured chunking**: Structure-aware chunking (section headers, benchmark
   blocks, timing paths) with metadata prefixes; splits the summary log header
   into semantic sub-groups (design config, synthesis, PnR, STA, DRC, verification)
4. **Structured + variant filter**: At query time, auto-detects variant name in
   the question and filters ChromaDB results to only that variant's chunks
5. **Structured + filter + rerank**: Additionally re-ranks retrieved chunks using
   Claude as a relevance judge before generating the answer

### Results

| Strategy | Retrieval Recall | Correctness (/3) | Completeness (/3) |
|---|---|---|---|
| No context (LLM only) | n/a | 0.57 | 0.53 |
| Fixed chunking | 0.38 | 1.40 | 1.80 |
| Structured chunking | 0.48 | 2.13 | 2.37 |
| Structured + variant filter | 0.63 | 2.07 | 2.23 |
| Structured + filter + rerank | 0.63 | 2.13 | 2.30 |

### Results by category (best RAG config vs. no context)

| Category | No Context Correctness | Best RAG Correctness | Improvement |
|---|---|---|---|
| simple-lookup | 0.25/3 | 2.83/3 | 11x |
| cross-document | 0.00/3 | 1.60/3 | from 0 |
| cross-variant | 0.00/3 | 1.43/3 | from 0 |
| reasoning | 2.33/3 | 2.00/3 | -14% (see analysis) |

### Key findings

**RAG provides 3.7x improvement in correctness** over the LLM alone (0.57 to
2.13). Without retrieved context, Claude cannot answer project-specific factual
questions — it scored near 0/3 on simple lookups, cross-document, and
cross-variant questions because those require exact numbers from the design
reports.

**Structured chunking improved correctness 52%** over fixed-size chunking (1.40
to 2.13). The main mechanism: splitting structured reports at semantic
boundaries (section headers, benchmark blocks, metric groups) produces chunks
whose embeddings better match targeted questions. A 2100-character chunk
containing 25 different metrics doesn't embed close to a question about any
single metric; six 200-400 character chunks each focused on one metric group do.

**Variant filtering improved retrieval recall 31%** (0.48 to 0.63). When a
question mentions specific design variants, filtering ChromaDB results to only
those variants (plus cross-variant reports) eliminates noise. The filter detects
all variant names in the question, so comparison questions like "How does X
compare to Y?" correctly scope to both variants. This is metadata-aware
retrieval — combining semantic search with structured filtering.

**Re-ranking improved answer quality but not retrieval recall.** Re-ranking
reorders the retrieved chunks by relevance but doesn't change which chunks are
retrieved. The quality improvement (correctness 2.07 to 2.13 when combined with
filtering) comes from promoting the most relevant chunks to the positions the
LLM pays most attention to.

**Reasoning questions scored higher without RAG than with it.** Claude's general
ASIC knowledge produces decent explanations (2.33/3) for questions like "why
does PnR add cells?" — these don't require project-specific data. When RAG
retrieves irrelevant chunks, they can confuse the model, slightly degrading
reasoning answers (2.00/3). This suggests that retrieval confidence thresholds
could help: skip retrieval when no chunks are sufficiently relevant.

---

## 5. Error Analysis

### Failure classification (best RAG config, 30 questions)

| Category | Count | Description |
|---|---|---|
| Retrieval hit + good answer | 16/30 | System working as intended |
| Retrieval miss + good answer | 6/30 | LLM compensated from other chunks or general knowledge |
| **Retrieval miss + bad answer** | **4/30** | **Retrieval failure** — wrong chunks retrieved |
| **Retrieval hit + bad answer** | **4/30** | **Generation failure** — right chunks, wrong answer |

### Retrieval and generation failures are evenly split

50% of failures (4/8) are retrieval misses leading to bad answers, and 50%
(4/8) are generation failures where the right context was retrieved but Claude
still got the answer wrong.

### RAG value

- RAG improved the answer for **21/30 questions** (70%)
- RAG hurt the answer for **3/30 questions** (10%) — reasoning questions
  where general knowledge was better than confused retrieval
- **6/30 questions** (20%) were unchanged

### Generation failures (3 cases)

These are cases where retrieval succeeded but the answer was still wrong:

- **cross-doc-01** ("How many cells added during PnR?"): Claude had both the
  synthesis and PnR cell counts but failed to compute the difference correctly
- **cross-variant-06** ("ioiqd2 vs oooiqd2 energy for bsearch"): The right
  section of the comparison report was retrieved, but the specific benchmark
  rows were in a different sub-chunk
- **reasoning-06** ("Which benchmark shows most CPI improvement?"): Required
  scanning an entire CPI table to find the maximum improvement — hard to answer
  from partial chunks

### Retrieval failure patterns

The most-missed documents:

| Document | Times missed | Why |
|---|---|---|
| `run.log` (summary header) | 8 | Even after splitting into sub-groups, the design config chunk (1058 chars with all parameters) doesn't embed close to questions about specific parameters |
| `report.txt` (comparison) | 4 | Cross-variant comparison tables are split into sub-chunks; some questions match a section that ended up in a different sub-chunk |
| `*-summary.rpt` (power) | 2 | Power reports for specific benchmarks have similar embeddings, making it hard to retrieve the exact right one |

### Root cause summary

Bad answers come more often from **retrieval failures** (56%, wrong documents
retrieved) than from **generation failures** (44%, LLM misinterpreting the right
documents), though both are significant. Improving retrieval is the higher-leverage
fix, but generation quality also matters.

---

## 6. Limitations

### Corpus-only constraint

The corpus consists entirely of design artifacts (reports, logs, RTL) with no
EDA tool documentation. Questions that require understanding tool-specific
concepts (e.g., "what does the `hnsw:space` setting mean in this context?") can
only be partially answered from the reports. This is a meaningful limitation:
a timing violation can be identified from the reports, but fully explaining why
it occurs may require tool documentation that isn't in the corpus.

### Embedding model mismatch

General-purpose embedding models (trained on web text) may not optimally encode
the specialized vocabulary of EDA reports. Terms like `synth_setup_slack`,
`pnr_density`, and `sta_hold_slack` are domain-specific shorthand that the
embedding model hasn't been specifically trained on. A domain-adapted embedding
model could improve retrieval quality.

### Chunk boundary sensitivity

Despite structure-aware chunking, some information spans chunk boundaries. The
cross-variant comparison table in `report.txt` is split into sub-chunks by row
groups, but a question comparing two specific benchmarks may require rows from
different sub-chunks.

### Evaluation scale

The benchmark has 30 questions. While sufficient to identify patterns and
compare strategies, a larger benchmark (100+) would provide more statistical
confidence in the results, especially for category-level comparisons where
some categories have only 5-7 questions.

### LLM-as-judge grading

Answer quality is graded by Claude itself, which introduces potential bias.
The grader may be more lenient toward answers that match Claude's own reasoning
patterns. A human evaluation on a subset would strengthen the findings.

---

## 7. Technical Decisions and Rationale

### Why 512 embedding dimensions (not 1536)?

OpenAI's `text-embedding-3-small` supports Matryoshka representation learning.
At 512 dimensions (vs. the default 1536), retrieval quality is nearly identical
for a corpus of this size, but storage and distance computation are 3x cheaper.

### Why cosine similarity?

OpenAI embeddings are L2-normalized, so cosine similarity and dot product are
equivalent. Cosine is the standard choice and is what ChromaDB's HNSW index is
optimized for.

### Why ChromaDB over a JSON file?

At 20k chunks, a brute-force numpy search would work fine. ChromaDB was chosen
for persistence (saves to disk without re-embedding), co-located text/metadata
storage, and as a standard RAG component worth discussing in the paper. It could
be replaced with ~30 lines of numpy code at this scale.

### Why no LangChain?

Direct SDK usage keeps the system transparent — every API call is visible in the
source code. This makes it easier to understand, debug, and explain in the paper.
The total codebase is ~900 lines across 7 modules.

### Why Claude for generation + OpenAI for embedding?

Anthropic does not offer an embedding API. OpenAI's `text-embedding-3-small` is
the most cost-effective embedding model available via API (~$0.13 for the full
corpus). Claude Sonnet was chosen for generation because of its strong
instruction-following and citation behavior.

### Why 1500-character chunks with 200-character overlap?

EDA report sections (timing paths, power tables, area breakdowns) typically run
500-2000 characters. 1500 characters captures most complete sections without
splitting them. 200-character overlap ensures context at boundaries is preserved
in at least one chunk. This produces ~20,000 chunks at a negligible embedding
cost.
