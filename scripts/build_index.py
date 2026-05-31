from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from rag_pipeline.chunking import chunk_text
from rag_pipeline.config import load_settings
from rag_pipeline.embeddings import SentenceTransformerEmbedder
from rag_pipeline.ingest import load_manifest, read_document_text
from rag_pipeline.store import RagStore


console = Console()


def build_chunks():
    settings = load_settings()
    records = load_manifest(settings.manifest_path, settings.text_dir)
    chunks = []
    for record in records:
        chunks.extend(chunk_text(record, read_document_text(record)))
    return settings, records, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Zurich insurance RAG index.")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only; do not write DB.")
    parser.add_argument(
        "--init-db-only", action="store_true", help="Create SQLite schema only."
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not clear existing indexed rows before rebuilding.",
    )
    args = parser.parse_args()

    settings = load_settings()
    store = RagStore(settings.database_path)
    store.init_schema()

    if args.init_db_only:
        console.print(f"Initialized database: {settings.database_path}")
        return

    settings, records, chunks = build_chunks()
    total_words = sum(chunk.word_count for chunk in chunks)
    avg_words = int(total_words / len(chunks)) if chunks else 0

    table = Table(title="Dry Run" if args.dry_run else "Index Build")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Documents loaded", str(len(records)))
    table.add_row("Chunks created", str(len(chunks)))
    table.add_row("Average chunk words", str(avg_words))
    table.add_row("Empty chunks", str(sum(1 for chunk in chunks if not chunk.text.strip())))
    console.print(table)

    if args.dry_run:
        for chunk in chunks[:3]:
            console.print(
                f"\n[bold]{chunk.document.title}[/bold] chunk {chunk.chunk_index} "
                f"({chunk.word_count} words)"
            )
            console.print(chunk.text[:600].strip())
        return

    if not args.no_reset:
        store.reset_index()

    chunk_ids = store.insert_chunks(chunks)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    texts = [chunk.text for chunk in chunks]
    vectors = []
    batch_size = 32
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[start : start + batch_size]
        vectors.append(embedder.encode(batch, batch_size=batch_size))

    import numpy as np

    matrix = np.vstack(vectors).astype("float32")
    store.insert_embeddings(chunk_ids, settings.embedding_model, matrix)
    console.print(store.stats())


if __name__ == "__main__":
    main()
