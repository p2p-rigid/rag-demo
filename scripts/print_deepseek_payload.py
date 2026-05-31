from __future__ import annotations

import argparse
import json

from rag_pipeline.config import load_settings
from rag_pipeline.embeddings import SentenceTransformerEmbedder
from rag_pipeline.generate import format_context
from rag_pipeline.retrieve import retrieve
from rag_pipeline.store import RagStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the exact DeepSeek chat/completions JSON payload."
    )
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--generation-k", type=int, default=4)
    args = parser.parse_args()

    settings = load_settings()
    top_k = args.top_k or settings.top_k
    store = RagStore(settings.database_path)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    chunks = retrieve(args.question, store, embedder, top_k)
    generation_chunks = chunks[: args.generation_k]
    numbered_context = format_context(generation_chunks)

    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an insurance document assistant. Use only the provided "
                    "Zurich document excerpts. Do not invent policy terms. If the "
                    "excerpts do not contain enough information, say what is missing. "
                    "Do not give legal, financial, or insurance advice. Cite source "
                    "labels from the provided context. Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{args.question}\n\n"
                    f"Context:\n{numbered_context}\n\n"
                    "Return exactly these sections:\n"
                    "Answer:\n"
                    "Relevant conditions / limitations:\n"
                    "Sources:\n\n"
                    "Use short bullets where helpful. Do not include more than "
                    "8 bullets in total."
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
