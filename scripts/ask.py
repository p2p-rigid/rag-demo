from __future__ import annotations

import argparse
import re
import time

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_pipeline.answer import answer_question
from rag_pipeline.config import load_settings
from rag_pipeline.embeddings import SentenceTransformerEmbedder
from rag_pipeline.generate import DeepSeekClient
from rag_pipeline.retrieve import retrieve
from rag_pipeline.store import RagStore, StoredChunk


console = Console()


def clean_preview(text: str, max_chars: int = 260) -> str:
    text = re.sub(r"\.{5,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def print_sources(chunks: list[StoredChunk]) -> None:
    table = Table(title="Sources", box=box.SIMPLE, expand=False)
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Document", overflow="fold")
    table.add_column("Product", overflow="fold")
    table.add_column("Chunk", justify="right")
    for index, chunk in enumerate(chunks, start=1):
        table.add_row(
            str(index),
            f"{chunk.score:.3f}" if chunk.score is not None else "",
            chunk.title,
            chunk.product,
            str(chunk.chunk_index),
        )
    console.print(table)


def print_retrieval(chunks: list[StoredChunk], show_chunks: bool) -> None:
    print_sources(chunks)
    if not show_chunks:
        console.print(
            "\nUse [bold]--show-chunks[/bold] to print retrieved text snippets."
        )
        return

    console.print("\n[bold]Retrieved snippets[/bold]")
    for index, chunk in enumerate(chunks, start=1):
        console.print(
            f"[bold]{index}. {chunk.title}[/bold] "
            f"(chunk {chunk.chunk_index}, {chunk.filename})"
        )
        console.print(clean_preview(chunk.text))
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Zurich insurance RAG pipeline.")
    parser.add_argument("question")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="When using --retrieve-only, print compact retrieved text snippets.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--timing", action="store_true", help="Print timing diagnostics.")
    parser.add_argument("--stream", action="store_true", help="Stream the DeepSeek answer.")
    args = parser.parse_args()

    settings = load_settings()
    top_k = args.top_k or settings.top_k
    start = time.perf_counter()
    store = RagStore(settings.database_path)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    chunks = retrieve(args.question, store, embedder, top_k)
    retrieved_at = time.perf_counter()

    console.print(Panel(args.question, title="Question"))
    if args.retrieve_only:
        print_retrieval(chunks, args.show_chunks)
        if args.timing:
            console.print(f"Timing: retrieval={retrieved_at - start:.3f}s")
        return

    if args.stream:
        generation_chunks = chunks[: min(4, len(chunks))]
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        client = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
        console.print("[bold]Answer[/bold]\n")
        first_token_at = None
        for token in client.stream_chat(args.question, generation_chunks):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            console.print(token, end="")
        console.print("\n")
        answered_at = time.perf_counter()
        print_sources(generation_chunks)
        if args.timing:
            first = (first_token_at - retrieved_at) if first_token_at else 0.0
            console.print(
                f"Timing: retrieval={retrieved_at - start:.3f}s "
                f"first_token={first:.3f}s "
                f"generation_total={answered_at - retrieved_at:.3f}s "
                f"total={answered_at - start:.3f}s"
            )
        return

    answer, chunks = answer_question(args.question, settings, top_k, chunks=chunks)
    answered_at = time.perf_counter()
    console.print(Panel(answer, title="Answer"))
    print_sources(chunks)
    if args.timing:
        console.print(
            f"Timing: retrieval={retrieved_at - start:.3f}s "
            f"generation={answered_at - retrieved_at:.3f}s "
            f"total={answered_at - start:.3f}s"
        )


if __name__ == "__main__":
    main()
