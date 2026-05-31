from __future__ import annotations

from dataclasses import dataclass
import re

from .ingest import DocumentRecord


@dataclass(frozen=True)
class TextChunk:
    document: DocumentRecord
    chunk_index: int
    text: str
    word_count: int


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    normalized = normalize_text(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    return paragraphs


def chunk_text(
    document: DocumentRecord,
    text: str,
    target_words: int = 850,
    overlap_words: int = 120,
) -> list[TextChunk]:
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")

    paragraphs = split_paragraphs(text)
    chunks: list[TextChunk] = []
    current: list[str] = []
    current_words = 0

    def emit() -> None:
        nonlocal current, current_words
        chunk_text_value = "\n\n".join(current).strip()
        if not chunk_text_value:
            current = []
            current_words = 0
            return
        words = chunk_text_value.split()
        chunks.append(
            TextChunk(
                document=document,
                chunk_index=len(chunks),
                text=chunk_text_value,
                word_count=len(words),
            )
        )
        overlap = words[-overlap_words:] if overlap_words else []
        current = [" ".join(overlap)] if overlap else []
        current_words = len(overlap)

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        if len(words) > target_words:
            if current_words:
                emit()
            step = target_words - overlap_words
            for start in range(0, len(words), step):
                part = " ".join(words[start : start + target_words]).strip()
                if not part:
                    continue
                chunks.append(
                    TextChunk(
                        document=document,
                        chunk_index=len(chunks),
                        text=part,
                        word_count=len(part.split()),
                    )
                )
            current = []
            current_words = 0
            continue

        if current_words and current_words + len(words) > target_words:
            emit()

        current.append(paragraph)
        current_words += len(words)

    if current_words:
        emit()

    return [chunk for chunk in chunks if chunk.text.strip()]
