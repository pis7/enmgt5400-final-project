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
from asic_rag.query import QueryExecutor

BENCHMARK_PATH = Path("benchmark.jsonl")


class EvalExecutor:
    def __init__(self):
        self.top_k: int = TOP_K
        self.skip_generation: bool = False
        self.use_variant_filter: bool = False
        self.use_rerank: bool = False
        self.no_context: bool = False
        self.output_path: Path = Path("eval_results.json")
        self.benchmark_path: Path = BENCHMARK_PATH
        self.benchmark: list[dict] = []
        self.results: list[dict] = []
        self.openai_client: OpenAI | None = None
        self.anthropic_client: Anthropic | None = None
        self.query_executor: QueryExecutor | None = None

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
        parser.add_argument("-k", "--top-k", type=int, default=TOP_K)
        parser.add_argument("--retrieval-only", action="store_true", help="Only evaluate retrieval, skip generation")
        parser.add_argument("-o", "--output", type=Path, default=Path("eval_results.json"), help="Output file")
        parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
        parser.add_argument("--filter-variant", action="store_true", help="Auto-filter by design variant in question")
        parser.add_argument("--rerank", action="store_true", help="Re-rank chunks with Claude before generating")
        parser.add_argument("--no-context", action="store_true", help="No retrieval — ask Claude directly (baseline)")
        args = parser.parse_args()

        self.top_k = args.top_k
        self.skip_generation = args.retrieval_only
        self.output_path = args.output
        self.benchmark_path = args.benchmark
        self.use_variant_filter = args.filter_variant
        self.use_rerank = args.rerank
        self.no_context = args.no_context

    def init_clients(self) -> None:
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()

        self.query_executor = QueryExecutor()
        self.query_executor.openai_client = self.openai_client
        self.query_executor.anthropic_client = self.anthropic_client
        self.query_executor.top_k = self.top_k

        if not self.no_context:
            chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
            self.query_executor.collection = chroma_client.get_collection("asic_debug")

    def load_benchmark(self) -> None:
        with open(self.benchmark_path) as f:
            self.benchmark = [json.loads(line) for line in f]
        print(f"Loaded {len(self.benchmark)} benchmark questions")

        flags = []
        if self.no_context:
            flags.append("no-context")
        if self.use_variant_filter:
            flags.append("variant-filter")
        if self.use_rerank:
            flags.append("rerank")
        if flags:
            print(f"  Enabled: {', '.join(flags)}")

    def score_retrieval(self, results: dict, expected_docs: list[str]) -> dict:
        """Check if any expected source docs appear in the retrieved chunk IDs."""
        retrieved_ids = results["ids"][0][:self.top_k]

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

    def grade_answer(self, question: str, expected: str, actual: str) -> dict:
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

        message = self.anthropic_client.messages.create(
            model=LLM_MODEL,
            max_tokens=512,
            system=grading_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        try:
            return json.loads(message.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return {"correctness": -1, "completeness": -1, "explanation": "Failed to parse grader response"}

    def run_eval(self) -> None:
        """Run evaluation on all benchmark questions."""
        qe = self.query_executor

        for i, item in enumerate(self.benchmark):
            print(f"[{i+1}/{len(self.benchmark)}] {item['id']}: {item['question'][:60]}...")

            entry = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
            }

            if self.no_context:
                entry["retrieval"] = {"recall": 0.0, "hits": [], "misses": item["source_docs"], "num_retrieved": 0}
                answer = qe.generate_answer(item["question"], "(No context provided — answer from your own knowledge.)")
                grade = self.grade_answer(item["question"], item["expected_answer"], answer)
                entry["answer"] = answer
                entry["grade"] = grade
                time.sleep(0.5)
                self.results.append(entry)
                continue

            variant = qe.extract_variant(item["question"]) if self.use_variant_filter else None
            retrieval_results = qe.retrieve(item["question"], variant_filter=variant)

            if self.use_rerank:
                retrieval_results = qe.rerank(item["question"], retrieval_results)

            retrieval_score = self.score_retrieval(retrieval_results, item["source_docs"])
            entry["retrieval"] = retrieval_score

            if not self.skip_generation:
                context = qe.format_context(retrieval_results)
                answer = qe.generate_answer(item["question"], context)
                grade = self.grade_answer(item["question"], item["expected_answer"], answer)
                entry["answer"] = answer
                entry["grade"] = grade
                time.sleep(0.5)

            self.results.append(entry)

    def save_results(self) -> None:
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {self.output_path}")

    def print_summary(self) -> None:
        """Print evaluation summary by category."""
        categories = sorted(set(r["category"] for r in self.results))

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Retrieval summary
        print("\n--- Retrieval (source doc recall) ---")
        for cat in categories:
            cat_results = [r for r in self.results if r["category"] == cat]
            avg_recall = sum(r["retrieval"]["recall"] for r in cat_results) / len(cat_results)
            perfect = sum(1 for r in cat_results if r["retrieval"]["recall"] == 1.0)
            print(f"  {cat:20s}  avg_recall={avg_recall:.2f}  perfect={perfect}/{len(cat_results)}")

        all_recall = sum(r["retrieval"]["recall"] for r in self.results) / len(self.results)
        all_perfect = sum(1 for r in self.results if r["retrieval"]["recall"] == 1.0)
        print(f"  {'OVERALL':20s}  avg_recall={all_recall:.2f}  perfect={all_perfect}/{len(self.results)}")

        if not self.skip_generation:
            print("\n--- Answer Quality (LLM graded) ---")
            for cat in categories:
                cat_results = [r for r in self.results if r["category"] == cat and "grade" in r]
                if not cat_results:
                    continue
                avg_correct = sum(r["grade"]["correctness"] for r in cat_results) / len(cat_results)
                avg_complete = sum(r["grade"]["completeness"] for r in cat_results) / len(cat_results)
                print(f"  {cat:20s}  correctness={avg_correct:.2f}/3  completeness={avg_complete:.2f}/3")

            graded = [r for r in self.results if "grade" in r]
            avg_correct = sum(r["grade"]["correctness"] for r in graded) / len(graded)
            avg_complete = sum(r["grade"]["completeness"] for r in graded) / len(graded)
            print(f"  {'OVERALL':20s}  correctness={avg_correct:.2f}/3  completeness={avg_complete:.2f}/3")

        misses = [(r["id"], r["retrieval"]["misses"]) for r in self.results if r["retrieval"]["misses"]]
        if misses:
            print("\n--- Retrieval Misses ---")
            for qid, missed in misses:
                print(f"  {qid}: missed {missed}")

        print()

def main():
    executor = EvalExecutor()
    executor.parse_args()
    executor.init_clients()
    executor.load_benchmark()
    executor.run_eval()
    executor.save_results()
    executor.print_summary()


if __name__ == "__main__":
    main()
