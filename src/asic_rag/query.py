"""CLI query interface: question -> retrieve -> generate answer with citations."""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import re
from openai import OpenAI
from anthropic import Anthropic
import chromadb

from asic_rag.config import (
    VECTORSTORE_DIR, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    TOP_K, LLM_MODEL, MAX_CONTEXT_CHUNKS,
)

# Known design variants for metadata filtering
KNOWN_VARIANTS = [
    "BlimpV11_2fe_2be_oooiqd2",
    "BlimpV11_2fe_2be_ioiqd2",
    "BlimpV11_2fe_2be",
    "BlimpV11_1fe_1be",
    "BlimpV8",
    "Proc6745",
]


class QueryExecutor:
    def __init__(self):
        self.top_k: int = TOP_K
        self.interactive: bool = False
        self.retrieve_only: bool = False
        self.filter_variant: bool = False
        self.use_rerank: bool = False
        self.question: str | None = None
        self.openai_client: OpenAI | None = None
        self.anthropic_client: Anthropic | None = None
        self.collection = None

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(description="Query the ASIC design RAG system")
        parser.add_argument("question", nargs="?", help="Question to ask")
        parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
        parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
        parser.add_argument("--retrieve-only", action="store_true", help="Show retrieved chunks without generating an answer")
        parser.add_argument("--filter-variant", action="store_true", help="Auto-filter by design variant mentioned in question")
        parser.add_argument("--rerank", action="store_true", help="Re-rank retrieved chunks with Claude before generating")
        args = parser.parse_args()

        self.question = args.question
        self.top_k = args.top_k
        self.interactive = args.interactive
        self.retrieve_only = args.retrieve_only
        self.filter_variant = args.filter_variant
        self.use_rerank = args.rerank

    def init_clients(self) -> None:
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()
        chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = chroma_client.get_collection("asic_debug")

    def extract_variants(self, question: str) -> list[str]:
        """Extract all design variant names mentioned in the question."""
        found = []
        q_lower = question.lower()
        for variant in KNOWN_VARIANTS:
            if variant.lower() in q_lower:
                found.append(variant)
        return found

    def retrieve(self, question: str, top_k: int | None = None, variant_filter: list[str] | None = None) -> dict:
        """Embed the question and retrieve top-k chunks from ChromaDB."""
        if top_k is None:
            top_k = self.top_k

        response = self.openai_client.embeddings.create(
            input=[question],
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        query_embedding = response.data[0].embedding

        where = None
        if variant_filter:
            # Match any mentioned variant OR "all" (cross-variant reports)
            where = {"$or": [{"design_variant": v} for v in variant_filter]
                     + [{"design_variant": "all"}]}

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

    def rerank(self, question: str, results: dict, top_n: int = MAX_CONTEXT_CHUNKS) -> dict:
        """Re-rank retrieved chunks using Claude to score relevance."""
        n = min(len(results["ids"][0]), top_n * 2)
        if n == 0:
            return results

        chunk_list = []
        for i in range(n):
            meta = results["metadatas"][0][i]
            text = results["documents"][0][i][:500]
            source = f"{meta.get('doc_id', '?')} ({meta.get('report_type')}, {meta.get('design_variant')})"
            chunk_list.append(f"Chunk {i}: [{source}]\n{text}")

        rerank_prompt = (
            "You are a relevance judge for an ASIC design RAG system.\n\n"
            "Given the question and candidate chunks below, score each chunk's relevance "
            "to answering the question on a scale of 0-10.\n\n"
            "Respond with ONLY a JSON array of objects: [{\"chunk\": 0, \"score\": N}, ...]\n"
            "Include ALL chunks. No other text."
        )

        user_msg = (
            f"Question: {question}\n\n"
            f"Candidates:\n\n" + "\n\n".join(chunk_list)
        )

        message = self.anthropic_client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=rerank_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        try:
            scores = {item["chunk"]: item["score"] for item in json.loads(message.content[0].text)}
        except Exception:
            return results

        indices = sorted(range(n), key=lambda i: scores.get(i, 0), reverse=True)

        reranked = {
            "ids": [[results["ids"][0][i] for i in indices]],
            "documents": [[results["documents"][0][i] for i in indices]],
            "metadatas": [[results["metadatas"][0][i] for i in indices]],
            "distances": [[results["distances"][0][i] for i in indices]],
        }
        return reranked

    def format_context(self, results: dict, max_chunks: int = MAX_CONTEXT_CHUNKS) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        parts = []
        n = min(max_chunks, len(results["ids"][0]))
        for i in range(n):
            meta = results["metadatas"][0][i]
            text = results["documents"][0][i]
            distance = results["distances"][0][i]
            source = (
                f"{meta.get('doc_id', 'unknown')} "
                f"({meta.get('design_variant')}, {meta.get('flow_stage')}, {meta.get('tool')})"
            )
            parts.append(f"[Source {i+1}: {source} | distance={distance:.3f}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def generate_answer(self, question: str, context: str) -> str:
        """Send question + retrieved context to Claude and return the answer."""
        system_prompt = (
            "You are an ASIC design debugging assistant. You help engineers understand "
            "synthesis reports, timing analysis, power reports, DRC results, and other "
            "EDA tool outputs from a pyhflow-based ASIC flow.\n\n"
            "Answer the user's question based on the retrieved context below. "
            "Cite your sources using [Source N] notation. "
            "If the context does not contain enough information, say so clearly.\n\n"
            f"Retrieved context:\n\n{context}"
        )

        message = self.anthropic_client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return message.content[0].text

    def ask(self, question: str) -> None:
        """Run the full retrieve-and-answer pipeline for a single question."""
        variants = self.extract_variants(question) if self.filter_variant else []
        results = self.retrieve(question, variant_filter=variants or None)

        if self.use_rerank:
            results = self.rerank(question, results)

        context = self.format_context(results)

        if self.retrieve_only:
            if variants:
                print(f"[Filtered to variants: {', '.join(variants)}]\n")
            print(context)
            return

        print("Generating answer...\n")
        answer = self.generate_answer(question, context)
        print(answer)
        print("\n--- Sources ---")
        for i in range(min(MAX_CONTEXT_CHUNKS, len(results["ids"][0]))):
            meta = results["metadatas"][0][i]
            print(f"  [{i+1}] {meta.get('doc_id', '?')} ({meta.get('report_type')}, {meta.get('design_variant')})")

def main():
    executor = QueryExecutor()
    executor.parse_args()
    executor.init_clients()

    if executor.interactive:
        print("ASIC Debug RAG — interactive mode (type 'quit' to exit)")
        while True:
            try:
                q = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                executor.ask(q)
    elif executor.question:
        executor.ask(executor.question)
    else:
        argparse.ArgumentParser(description="Query the ASIC design RAG system").print_help()


if __name__ == "__main__":
    main()
