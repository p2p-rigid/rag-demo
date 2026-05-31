from pathlib import Path

from rag_pipeline.chunking import chunk_text
from rag_pipeline.ingest import load_manifest, read_document_text


def test_chunking_creates_non_empty_chunks_with_metadata():
    record = load_manifest(
        Path("data/zurich_insurance/manifest.csv"),
        Path("data/zurich_insurance/text"),
    )[0]
    chunks = chunk_text(record, read_document_text(record), target_words=500)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.document.filename == record.filename for chunk in chunks)
    assert chunks[0].chunk_index == 0
