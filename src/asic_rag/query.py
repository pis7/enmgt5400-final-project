"""CLI query interface: question -> retrieve -> generate answer with citations."""

from dotenv import load_dotenv
load_dotenv()

import argparse
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


def extract_variant(question: str) -> str | None:
    """Extract a design variant name from the question, if mentioned."""
    for variant in KNOWN_VARIANTS:
        if variant.lower() in question.lower():
            return variant
    return None


def retrieve(
    question: str,
    collection,
    openai_client: OpenAI,
    top_k: int = TOP_K,
    variant_filter: str | None = None,
) -> dict:
    """Embed the question and retrieve top-k chunks from ChromaDB."""
    response = openai_client.embeddings.create(
        input=[question],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    query_embedding = response.data[0].embedding

    where = None
    if variant_filter:
        # Match the specific variant OR "all" (for cross-variant reports)
        where = {"$or": [
            {"design_variant": variant_filter},
            {"design_variant": "all"},
        ]}

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )


def rerank(
    question: str,
    results: dict,
    anthropic_client: Anthropic,
    top_n: int = MAX_CONTEXT_CHUNKS,
) -> dict:
    """Re-rank retrieved chunks using Claude to score relevance."""
    n = min(len(results["ids"][0]), top_n * 2)  # re-rank from a wider pool
    if n == 0:
        return results

    # Build a numbered list of chunks for Claude to score
    chunk_list = []
    for i in range(n):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i][:500]  # truncate for re-ranking prompt
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

    message = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=rerank_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    try:
        scores = {item["chunk"]: item["score"] for item in __import__("json").loads(message.content[0].text)}
    except Exception:
        return results  # fall back to original order on parse failure

    # Re-sort by score descending
    indices = sorted(range(n), key=lambda i: scores.get(i, 0), reverse=True)

    reranked = {
        "ids": [[results["ids"][0][i] for i in indices]],
        "documents": [[results["documents"][0][i] for i in indices]],
        "metadatas": [[results["metadatas"][0][i] for i in indices]],
        "distances": [[results["distances"][0][i] for i in indices]],
    }
    return reranked


def format_context(results: dict, max_chunks: int = MAX_CONTEXT_CHUNKS) -> str:
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


def generate_answer(question: str, context: str, anthropic_client: Anthropic) -> str:
    """Send question + retrieved context to Claude and return the answer."""
    system_prompt = (
        "You are an ASIC design debugging assistant. You help engineers understand "
        "synthesis reports, timing analysis, power reports, DRC results, and other "
        "EDA tool outputs from an mflowgen-based ASIC flow.\n\n"
        "Answer the user's question based on the retrieved context below. "
        "Cite your sources using [Source N] notation. "
        "If the context does not contain enough information, say so clearly.\n\n"
        f"Retrieved context:\n\n{context}"
    )

    message = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


def main():
    parser = argparse.ArgumentParser(description="Query the ASIC design RAG system")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--retrieve-only", action="store_true", help="Show retrieved chunks without generating an answer")
    parser.add_argument("--filter-variant", action="store_true", help="Auto-filter by design variant mentioned in question")
    parser.add_argument("--rerank", action="store_true", help="Re-rank retrieved chunks with Claude before generating")
    args = parser.parse_args()

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = chroma_client.get_collection("asic_debug")
    anthropic_client = Anthropic()

    def ask(question: str):
        variant = extract_variant(question) if args.filter_variant else None
        results = retrieve(question, collection, openai_client, top_k=args.top_k, variant_filter=variant)

        if args.rerank:
            results = rerank(question, results, anthropic_client)

        context = format_context(results)

        if args.retrieve_only:
            if variant:
                print(f"[Filtered to variant: {variant}]\n")
            print(context)
            return

        print("Generating answer...\n")
        answer = generate_answer(question, context, anthropic_client)
        print(answer)
        print("\n--- Sources ---")
        for i in range(min(MAX_CONTEXT_CHUNKS, len(results["ids"][0]))):
            meta = results["metadatas"][0][i]
            print(f"  [{i+1}] {meta.get('doc_id', '?')} ({meta.get('report_type')}, {meta.get('design_variant')})")

    if args.interactive:
        print("ASIC Debug RAG — interactive mode (type 'quit' to exit)")
        while True:
            try:
                q = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                ask(q)
    elif args.question:
        ask(args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
