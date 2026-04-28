# In-Class Demo Script (5-10 minutes)

Pre-requisites: `cd build` directory with pre-generated `vectorstore/`,
`corpus.jsonl`, and `benchmark.jsonl` already in place. Terminal font size
large enough for the back of the room.

---

## 1. Introduction and Problem Statement (~3 min)

### 1a. Who I am

> "I'm Parker, finishing my MEng in ECE at Cornell. I'm currently working on
> a tapeout of a superscalar RISC-V processor called Zeppelin. After
> graduation I'm starting full-time at Apple in physical design."

### 1b. What I built and why

> "For this project I built a retrieval-augmented generation system — a RAG
> system — that can answer natural language questions about my ASIC design
> by searching through the hundreds of reports that the design flow produces."

**Talking points — the problem:**
- When you're designing a chip, you write RTL, then run it through a long pipeline of EDA tools: synthesis, place-and-route, static timing analysis, power analysis, design rule checking
- Each tool produces dense, structured reports — timing paths, area breakdowns, power tables, violation logs
- Over the course of a tapeout, you accumulate hundreds of these files across multiple design configurations
- When something goes wrong — a timing violation after back-annotation, unexpected power on a benchmark, a DRC error — the answer almost always exists somewhere in these reports
- The hard part isn't that the information doesn't exist. The hard part is *finding* it
- I've spent months on exactly this kind of debugging: simulation failures, timing violations on clock-gating cells, disagreements between tools, DRC errors
- This project asks: can we use RAG to make that search fast and natural-language-accessible?

**Talking points — why RAG:**
- LLMs like Claude are good at reading and summarizing technical documents, but they don't know anything about *my* design
- Fine-tuning would bake knowledge into model weights — expensive, static, and overkill for this
- RAG keeps the knowledge external: retrieve the relevant report snippets at query time, feed them to the LLM as context
- This means the system stays up-to-date automatically — re-run the corpus when reports change, no retraining
- The closest prior work is NVIDIA's ChipNeMo (2023), which built RAG assistants for internal chip design tasks. My project is much smaller in scope, but applies the same idea to a real student tapeout and focuses on honest evaluation

**Talking points — why this is feasible as a class project:**
- I already have a corpus of real design files that I know well
- Because I generated these reports and debugged these designs myself, I can tell whether the system gives good answers
- That ground truth is the hard part of evaluating any LLM tool — having it from my own experience is what makes this project workable in 30-35 hours

### 1c. Show the chip and the design space

Open the chip plot of Zeppelin in Innovus (FreePDK 45nm) full-screen.

> "This is one configuration of the processor — about 67,000 standard cells
> in a 45nm process. I ran 6 different configurations through the full flow."

**Talking points:**
- 6 design variants: a simple 5-stage in-order pipeline (Proc6745), a baseline superscalar (BlimpV8), and four BlimpV11 variants varying lane width (1-wide vs 2-wide), issue queue depth, and in-order vs out-of-order issue
- Each configuration goes through the full 11-stage ASIC flow
- The target PDK is FreePDK 45nm — open source, so everything in this project can be shared
- This design space exploration is itself a useful exercise — the reports let us compare area, timing, power, and performance across all 6 variants

### 1d. Show what the ASIC flow produces

> "Let me show you what these tools actually output."

**Step 1 — Flow summary** (start here to orient the audience):

```bash
code /work/global/pis7/blimp/hw/top/asic/build-batch-1-20260415-194241-45/BlimpV11_2fe_2be/11-summarize-results/run.log
```

> "This is the summary for one design variant — all 11 stages, pass/fail
> status, key metrics, and per-benchmark performance."

**Talking points:**
- Point out the key metrics: synth area (99,873 um^2), PnR density (36.16%), STA setup slack (2.8132 ns)
- Point out the per-benchmark section — 28 benchmarks, each with cycles, power, energy, CPI
- This summary is relatively concise, but it's just one of 6 variants
- And this is the *summary* — the detailed reports behind each metric are much bigger

**Step 2 — Timing report** (show the density):

```bash
code /work/global/pis7/blimp/hw/top/asic/build-batch-1-20260415-194241-45/BlimpV11_2fe_2be/06-synopsys-pt-sta/timing-setup.rpt
```

> "Static timing analysis — every critical path, gate-by-gate delay
> breakdowns. This single file is 96 KB of dense tables."

**Talking points:**
- Scroll through briefly to show the density — gate names, fanouts, incremental delays
- Each "Startpoint" to "slack" block is one timing path
- When debugging a timing violation, you need to find the right path in here — and there may be hundreds of paths
- Now multiply this by 6 design variants

**Step 3 — Power report** (small but multiplied):

```bash
code /work/global/pis7/blimp/hw/top/asic/build-batch-1-20260415-194241-45/BlimpV11_2fe_2be/08-synopsys-pt-pwr/BlimpV11_sim-vvadd-summary.rpt
```

> "Power analysis for one benchmark on one design. There's a separate report
> like this for each of 28 benchmarks times 6 variants — 168 power reports."

**Talking points:**
- Point out the power groups: clock (7.6%), register (14.2%), combinational (78.2%)
- This one file is short and readable — but the problem is scale, not complexity
- "Which benchmark has the highest power on which variant?" requires opening dozens of these

**Step 4 — State the problem clearly:**

> "475 files, 25 MB of text across 6 designs. The information I need to
> debug a problem is in here — but finding it manually takes minutes each
> time, and I do it dozens of times a day. That's what this project solves."

---

## 2. Show the RAG Pipeline Concept (~1 min)

Briefly explain the pipeline (reference the diagram from the report if
presenting slides alongside):

> "The idea is simple: chunk these reports, embed them into vectors, store
> them in a database, and then retrieve only the relevant pieces when I ask
> a question. The corpus and vector store are pre-built -- the only thing
> running live is the query."

**Talking points:**
- Embedding converts text into a 512-number vector where similar meaning = nearby points
- ChromaDB stores ~20,000 chunk vectors and finds the closest matches to a question
- The top matching chunks are fed to Claude as context, and it generates a grounded answer
- No fine-tuning needed — the model works out of the box with the right context
- Entire system is ~900 lines of Python, no LangChain

---

## 3. Live Queries (~4-5 min)

### 3a. Simple lookup -- no RAG features

Ask a straightforward factual question:

```bash
uv run query "What is the setup timing slack for BlimpV11_2fe_2be after STA?"
```

> Point out: the answer cites specific sources, gives the exact number
> (2.8132 ns), and explains what it means -- all from a 25 MB corpus
> searched in seconds.

**Talking points:**
- Without RAG, Claude scores 0.25/3 on these lookups — it literally cannot answer
- With RAG, simple lookups score 2.83/3
- The `[Source N]` citations let you trace every fact back to the original report
- This took about 2 seconds — manually grepping would take much longer

### 3b. Retrieve-only mode

Show what the LLM actually sees before generating:

```bash
uv run query --retrieve-only "How many standard cells are in BlimpV11_2fe_2be after synthesis?"
```

> "These are the raw chunks the system retrieved. Each one has a source path,
> distance score, and the actual text from the report. The LLM reads these
> and synthesizes an answer."

**Talking points:**
- This is the "R" in RAG — what actually gets retrieved
- Each chunk has a distance score (lower = more similar to the question)
- The metadata prefix (`[Design: ... | Stage: ... | Type: ...]`) helps the embedding model understand context
- Structure-aware chunking is key — we split reports at section boundaries, not arbitrary character positions

### 3c. Cross-variant comparison with filtering

```bash
uv run query --filter-variant "How does vvadd energy compare between BlimpV11_2fe_2be_oooiqd2 and BlimpV11_2fe_2be?"
```

> "This question spans two designs. The variant filter narrows the search
> space to only chunks from the mentioned design, which improves retrieval
> quality."

**Talking points:**
- Variant filtering uses ChromaDB's `where` clause to scope results by metadata
- Without filtering, you get chunks from all 6 variants competing for the top-k slots
- This improved retrieval recall by 11% (0.57 to 0.63)
- It's essentially metadata-aware retrieval — combining semantic search with structured filtering
- The system auto-detects which variant is mentioned in the question

### 3d. Reasoning question with reranking

```bash
uv run query --filter-variant --rerank "Why does BlimpV11_2fe_2be have tighter timing slack than simpler variants like Proc6745?"
```

> "Reranking asks the LLM to score each retrieved chunk for relevance before
> generating the final answer. It doesn't change what's retrieved, but it
> reorders the context so the most useful chunks come first."

**Talking points:**
- Interesting finding: reasoning questions actually score higher *without* RAG (2.33 vs 2.17)
- Claude's general ASIC knowledge is enough to explain *why* things happen
- RAG is most valuable for *factual* questions where the answer is a specific number or metric
- When retrieval pulls in irrelevant context, it can actually confuse the model
- This suggests future work: retrieval confidence thresholds to skip retrieval when no chunks are relevant

---

## 4. Results Summary (~1 min)

Pull up [results.md](results.md) (or a rendered version) to show the tables.

Key numbers to highlight:

- **3.7x improvement** in answer correctness over LLM alone
- **11x improvement** on simple factual lookups (0.25 to 2.83)
- **70%** of benchmark questions improved by RAG
- Retrieval and generation failures split evenly — **50/50** at the best config
- Entire system is **~900 lines of Python**, no LangChain

> "The takeaway: even a simple RAG pipeline with the right chunking strategy
> can make a 25 MB corpus of ASIC reports searchable in natural language."

**Talking points for the conclusion:**
- The main contribution: an honest evaluation of RAG on real ASIC design artifacts
- Structure-aware chunking matters — 52% correctness improvement over naive fixed-size
- Variant filtering matters — 31% retrieval recall improvement by scoping to mentioned designs
- At the best config, retrieval and generation failures are evenly split — both are worth improving
- The system is practical — I've been using it while debugging the actual tapeout
- Future work: domain-adapted embeddings, retrieval confidence thresholds, adding tool documentation to the corpus

---

## Timing Checklist

| Section | Target | Running |
|---|---|---|
| Intro + problem (who, chip, reports) | 3 min | 3 min |
| Pipeline concept | 1 min | 4 min |
| Live queries (3-4 shown) | 4 min | 8 min |
| Results summary | 1 min | 9 min |
| Audience Q&A / buffer | 1-2 min | 10-11 min |

## Backup Plan

If the OpenAI/Anthropic APIs are down or slow during the demo, have
screenshots of each query's output saved as PNGs in `docs/` to show instead.

## Anticipated Questions

- **"Why not fine-tune?"** — Fine-tuning bakes knowledge into model weights, which is expensive and static. RAG keeps knowledge external and updatable — re-run the corpus when reports change, no retraining needed.
- **"Why not just use grep?"** — Grep finds exact strings. RAG finds semantically similar content. "setup timing slack" matches "synth_setup_slack = 0.0021 ns" even though they share no words.
- **"How much did this cost?"** — Embedding the full corpus: ~$0.13. Each query: ~$0.01 (embedding) + ~$0.03 (Claude generation). Total project API spend: under $10.
- **"Would this work on a larger design?"** — ChromaDB's HNSW index scales well. The bottleneck would be embedding cost (linear in corpus size) and retrieval quality (more noise in a larger corpus).
- **"Why not LangChain?"** — Wanted to understand every API call rather than hiding behind abstractions. Also makes the system easier to explain in the paper.