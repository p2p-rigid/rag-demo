from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag_pipeline.config import load_settings
from rag_pipeline.embeddings import SentenceTransformerEmbedder
from rag_pipeline.retrieve import retrieve
from rag_pipeline.store import RagStore


console = Console()


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    settings = load_settings()
    questions = load_questions(Path("eval/questions.jsonl"))
    store = RagStore(settings.database_path)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)

    hit3 = 0
    hit6 = 0
    table = Table(title="Retrieval Evaluation")
    table.add_column("Question")
    table.add_column("Expected")
    table.add_column("Hit@3")
    table.add_column("Hit@6")
    table.add_column("Top Source")

    for item in questions:
        chunks = retrieve(item["question"], store, embedder, top_k=6)
        filenames = [chunk.filename for chunk in chunks]
        expected = item["expected_source"]
        ok3 = expected in filenames[:3]
        ok6 = expected in filenames[:6]
        hit3 += int(ok3)
        hit6 += int(ok6)
        table.add_row(
            item["question"],
            expected,
            "yes" if ok3 else "no",
            "yes" if ok6 else "no",
            filenames[0] if filenames else "",
        )

    console.print(table)
    total = len(questions)
    console.print(f"Questions: {total}")
    console.print(f"Hit@3: {hit3 / total:.0%}")
    console.print(f"Hit@6: {hit6 / total:.0%}")


if __name__ == "__main__":
    main()
