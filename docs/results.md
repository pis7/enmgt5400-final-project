# Evaluation Results

## Results by RAG Configuration (30 benchmark questions)

| Strategy | Retrieval Recall | Correctness (/3) | Completeness (/3) |
|---|---|---|---|
| No context (LLM only) | n/a | 0.57 | 0.53 |
| Fixed chunking | 0.38 | 1.40 | 1.80 |
| Structured chunking | 0.48 | 2.13 | 2.37 |
| Structured + variant filter | 0.63 | 2.07 | 2.23 |
| **Structured + filter + rerank** | **0.63** | **2.13** | **2.30** |

## Results by Question Category (best config vs. no context)

| Category | No Context | Best RAG | Improvement |
|---|---|---|---|
| Simple-lookup (12) | 0.25/3 | 2.83/3 | 11x |
| Cross-document (5) | 0.00/3 | 1.60/3 | from 0 |
| Cross-variant (7) | 0.00/3 | 1.43/3 | from 0 |
| Reasoning (6) | 2.33/3 | 2.00/3 | -14% |

## Error Analysis (best config, 30 questions)

| Category | Count | Description |
|---|---|---|
| Retrieval hit + good answer | 16/30 | System working as intended |
| Retrieval miss + good answer | 6/30 | LLM compensated |
| Retrieval miss + bad answer | 4/30 | Retrieval failure |
| Retrieval hit + bad answer | 4/30 | Generation failure |

## Key Takeaways

- **3.7x improvement** in answer correctness over the LLM alone
- **11x improvement** on simple factual lookups (0.25 to 2.83)
- **70%** of benchmark questions improved by RAG
- Retrieval and generation failures split **50/50** at the best config
- Entire system is **~900 lines of Python**, no LangChain
