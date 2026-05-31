from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from rag_pipeline.config import load_settings
from rag_pipeline.ingest import load_manifest
from rag_pipeline.store import RagStore


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the RAG corpus or SQLite DB.")
    parser.add_argument("--check-manifest", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    if args.check_manifest:
        records = load_manifest(settings.manifest_path, settings.text_dir)
        console.print(f"Manifest OK: {len(records)} documents")
        return

    store = RagStore(settings.database_path)
    store.init_schema()
    stats = store.stats()
    table = Table(title=str(settings.database_path))
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    main()
