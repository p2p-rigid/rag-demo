from pathlib import Path

from rag_pipeline.ingest import load_manifest


def test_manifest_loads_all_documents():
    records = load_manifest(
        Path("data/zurich_insurance/manifest.csv"),
        Path("data/zurich_insurance/text"),
    )
    assert len(records) == 21
    assert all(record.text_path.exists() for record in records)
    assert all(record.title for record in records)
