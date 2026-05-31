from pathlib import Path

import numpy as np

from rag_pipeline.chunking import TextChunk
from rag_pipeline.ingest import DocumentRecord
from rag_pipeline.retrieve import retrieve
from rag_pipeline.store import RagStore


class FakeEmbedder:
    def encode(self, texts, batch_size=32):
        del texts, batch_size
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_retrieve_returns_scored_results(tmp_path):
    record = DocumentRecord(
        filename="motor.pdf",
        title="Zurich Motor Insurance PDS",
        product="Motor Insurance",
        document_type="PDS and policy wording",
        jurisdiction="Australia",
        source_url="https://example.test/motor.pdf",
        retrieved_date="2026-05-29",
        notes="",
        text_path=Path("unused.txt"),
    )
    store = RagStore(tmp_path / "rag.sqlite")
    store.init_schema()
    chunk_ids = store.insert_chunks(
        [
            TextChunk(record, 0, "motor exclusions and policy conditions", 5),
            TextChunk(record, 1, "unrelated wording", 2),
        ]
    )
    store.insert_embeddings(
        chunk_ids,
        "fake",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    results = retrieve("motor exclusions", store, FakeEmbedder(), top_k=1)

    assert len(results) == 1
    assert results[0].filename == "motor.pdf"
    assert results[0].score is not None
