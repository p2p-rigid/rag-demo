from __future__ import annotations

from .config import Settings
from .embeddings import SentenceTransformerEmbedder
from .generate import DeepSeekClient
from .retrieve import retrieve
from .store import RagStore, StoredChunk


def answer_question(
    question: str,
    settings: Settings,
    top_k: int | None = None,
    chunks: list[StoredChunk] | None = None,
) -> tuple[str, list[StoredChunk]]:
    if chunks is None:
        store = RagStore(settings.database_path)
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        chunks = retrieve(question, store, embedder, top_k or settings.top_k)
    generation_chunks = chunks[: min(4, len(chunks))]
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
    )
    return client.chat(question, generation_chunks), generation_chunks
