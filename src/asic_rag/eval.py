"""Evaluate the RAG pipeline against the benchmark.

Measures two things separately:
1. Retrieval quality — did the right source documents appear in the top-k?
2. Answer quality — does the generated answer contain the expected information?

Answer quality is judged by an LLM (Claude) acting as a grader.
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from anthropic import Anthropic
import chromadb

from asic_rag.config import (
    VECTORSTORE_DIR, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    TOP_K, LLM_MODEL, MAX_CONTEXT_CHUNKS,
)
from asic_rag.query import retrieve, rerank, extract_variant, format_context, generate_answer

BENCHMARK_PATH = Path("benchmark.jsonl")


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def score_retrieval(results: dict, expected_docs: list[str], top_k: int) -> dict:
    """Check if any expected source docs appear in the retrieved chunk IDs."""
    retrieved_ids = results["ids"][0][:top_k]

    # A retrieved chunk ID looks like "BlimpV11_2fe_2be/11-summarize-results/run.log::chunk_3"
    # Match if the chunk ID starts with any expected doc prefix
    hits = set()
    for expected in expected_docs:
        for rid in retrieved_ids:
            if rid.startswith(expected):
                hits.add(expected)
                break

    return {
        "recall": len(hits) / len(expected_docs) if expected_docs else 0.0,
        "hits": list(hits),
        "misses": [d for d in expected_docs if d not in hits],
        "num_retrieved": len(retrieved_ids),
    }


def grade_answer(question: str, expected: str, actual: str, client: Anthropic) -> dict:
    """Use Claude as a grader to judge answer quality."""
    grading_prompt = (
        "You are grading a RAG system's answer against an expected answer.\n\n"
        "Score the answer on two dimensions:\n"
        "1. **correctness** (0-3): Does the answer contain the key facts from the expected answer?\n"
        "   0 = completely wrong or irrelevant\n"
        "   1 = partially correct but missing key facts\n"
        "   2 = mostly correct with minor inaccuracies\n"
        "   3 = fully correct\n"
        "2. **completeness** (0-3): Does the answer address all parts of the question?\n"
        "   0 = doesn't address the question\n"
        "   1 = addresses part of the question\n"
        "   2 = addresses most of the question\n"
        "   3 = fully addresses the question\n\n"
        "Respond with ONLY a JSON object: {\"correctness\": N, \"completeness\": N, \"explanation\": \"...\"}\n"
        "No other text."
    )

    user_msg = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"Actual answer: {actual}"
    )

    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=512,
        system=grading_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    try:
        return json.loads(message.content[0].text)
    except (json.JSONDecodeError, IndexError):
        return {"correctness": -1, "completeness": -1, "explanation": "Failed to parse grader response"}


def run_eval(
    benchmark: list[dict],
    collection,
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    top_k: int = TOP_K,
    skip_generation: bool = False,
    use_variant_filter: bool = False,
    use_rerank: bool = False,
) -> list[dict]:
    """Run evaluation on all benchmark questions."""
    results = []

    for i, item in enumerate(benchmark):
        print(f"[{i+1}/{len(benchmark)}] {item['id']}: {item['question'][:60]}...")

        variant = extract_variant(item["question"]) if use_variant_filter else None
        retrieval_results = retrieve(
            item["question"], collection, openai_client,
            top_k=top_k, variant_filter=variant,
        )

        if use_rerank:
            retrieval_results = rerank(
                item["question"], retrieval_results, anthropic_client,
            )

        retrieval_score = score_retrieval(retrieval_results, item["source_docs"], top_k)

        entry = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "retrieval": retrieval_score,
        }

        if not skip_generation:
            context = format_context(retrieval_results)
            answer = generate_answer(item["question"], context, anthropic_client)
            grade = grade_answer(item["question"], item["expected_answer"], answer, anthropic_client)
            entry["answer"] = answer
            entry["grade"] = grade
            time.sleep(0.5)  # rate limit courtesy

        results.append(entry)

    return results


def print_summary(results: list[dict], skip_generation: bool = False) -> None:
    """Print evaluation summary by category."""
    categories = sorted(set(r["category"] for r in results))

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Retrieval summary
    print("\n--- Retrieval (source doc recall) ---")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        avg_recall = sum(r["retrieval"]["recall"] for r in cat_results) / len(cat_results)
        perfect = sum(1 for r in cat_results if r["retrieval"]["recall"] == 1.0)
        print(f"  {cat:20s}  avg_recall={avg_recall:.2f}  perfect={perfect}/{len(cat_results)}")

    all_recall = sum(r["retrieval"]["recall"] for r in results) / len(results)
    all_perfect = sum(1 for r in results if r["retrieval"]["recall"] == 1.0)
    print(f"  {'OVERALL':20s}  avg_recall={all_recall:.2f}  perfect={all_perfect}/{len(results)}")

    if not skip_generation:
        # Answer quality summary
        print("\n--- Answer Quality (LLM graded) ---")
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat and "grade" in r]
            if not cat_results:
                continue
            avg_correct = sum(r["grade"]["correctness"] for r in cat_results) / len(cat_results)
            avg_complete = sum(r["grade"]["completeness"] for r in cat_results) / len(cat_results)
            print(f"  {cat:20s}  correctness={avg_correct:.2f}/3  completeness={avg_complete:.2f}/3")

        graded = [r for r in results if "grade" in r]
        avg_correct = sum(r["grade"]["correctness"] for r in graded) / len(graded)
        avg_complete = sum(r["grade"]["completeness"] for r in graded) / len(graded)
        print(f"  {'OVERALL':20s}  correctness={avg_correct:.2f}/3  completeness={avg_complete:.2f}/3")

    # Show retrieval misses
    misses = [(r["id"], r["retrieval"]["misses"]) for r in results if r["retrieval"]["misses"]]
    if misses:
        print("\n--- Retrieval Misses ---")
        for qid, missed in misses:
            print(f"  {qid}: missed {missed}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K)
    parser.add_argument("--retrieval-only", action="store_true", help="Only evaluate retrieval, skip generation")
    parser.add_argument("-o", "--output", type=Path, default=Path("eval_results.json"), help="Output file")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--filter-variant", action="store_true", help="Auto-filter by design variant in question")
    parser.add_argument("--rerank", action="store_true", help="Re-rank chunks with Claude before generating")
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    print(f"Loaded {len(benchmark)} benchmark questions")

    flags = []
    if args.filter_variant:
        flags.append("variant-filter")
    if args.rerank:
        flags.append("rerank")
    if flags:
        print(f"  Enabled: {', '.join(flags)}")

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = chroma_client.get_collection("asic_debug")
    anthropic_client = Anthropic()

    results = run_eval(
        benchmark, collection, openai_client, anthropic_client,
        top_k=args.top_k, skip_generation=args.retrieval_only,
        use_variant_filter=args.filter_variant, use_rerank=args.rerank,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

    print_summary(results, skip_generation=args.retrieval_only)


if __name__ == "__main__":
    main()
