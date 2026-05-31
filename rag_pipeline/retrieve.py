from __future__ import annotations

from dataclasses import replace
import re

from .embeddings import SentenceTransformerEmbedder, cosine_scores
from .store import RagStore, StoredChunk


STOP_TERMS = {
    "the",
    "and",
    "for",
    "zurich",
    "insurance",
    "policy",
    "apply",
    "does",
    "what",
    "who",
    "how",
    "under",
}


def query_terms(question: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if len(term) > 2 and term not in STOP_TERMS
    }


def metadata_boost(question: str, chunk: StoredChunk) -> float:
    terms = query_terms(question)
    if not terms:
        return 0.0
    metadata = " ".join(
        [
            chunk.filename,
            chunk.title,
            chunk.product,
            chunk.document_type,
        ]
    ).lower()
    hits = sum(1 for term in terms if term in metadata)
    return 0.45 * (hits / len(terms))


def content_boost(question: str, chunk: StoredChunk) -> float:
    terms = query_terms(question)
    if not terms:
        return 0.0
    text = chunk.text.lower()
    hits = sum(1 for term in terms if term in text)
    return 0.20 * (hits / len(terms))


def retrieve(
    question: str,
    store: RagStore,
    embedder: SentenceTransformerEmbedder,
    top_k: int = 6,
) -> list[StoredChunk]:
    chunks, matrix = store.fetch_embeddings()
    if not chunks:
        raise RuntimeError("No indexed chunks found. Run scripts.build_index first.")
    query = embedder.encode([question])[0]
    scores = cosine_scores(query, matrix)
    ranked = sorted(
        (
            (
                chunk,
                float(score)
                + metadata_boost(question, chunk)
                + content_boost(question, chunk),
            )
            for chunk, score in zip(chunks, scores, strict=True)
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [replace(chunk, score=float(score)) for chunk, score in ranked[:top_k]]
