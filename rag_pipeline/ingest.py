from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "filename",
    "title",
    "product",
    "document_type",
    "jurisdiction",
    "source_url",
    "retrieved_date",
}


@dataclass(frozen=True)
class DocumentRecord:
    filename: str
    title: str
    product: str
    document_type: str
    jurisdiction: str
    source_url: str
    retrieved_date: str
    notes: str
    text_path: Path


def load_manifest(manifest_path: Path, text_dir: Path) -> list[DocumentRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        records: list[DocumentRecord] = []
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"Manifest row {row_number} has no filename")
            text_path = text_dir / f"{Path(filename).stem}.txt"
            if not text_path.exists():
                raise FileNotFoundError(
                    f"Text file missing for manifest row {row_number}: {text_path}"
                )
            records.append(
                DocumentRecord(
                    filename=filename,
                    title=(row.get("title") or "").strip(),
                    product=(row.get("product") or "").strip(),
                    document_type=(row.get("document_type") or "").strip(),
                    jurisdiction=(row.get("jurisdiction") or "").strip(),
                    source_url=(row.get("source_url") or "").strip(),
                    retrieved_date=(row.get("retrieved_date") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                    text_path=text_path,
                )
            )
    return records


def read_document_text(record: DocumentRecord) -> str:
    return record.text_path.read_text(encoding="utf-8", errors="replace")
