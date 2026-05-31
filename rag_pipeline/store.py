from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np

from .chunking import TextChunk
from .embeddings import blob_to_vector, vector_to_blob
from .ingest import DocumentRecord


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    text: str
    word_count: int
    filename: str
    title: str
    product: str
    document_type: str
    jurisdiction: str
    source_url: str
    score: float | None = None


class RagStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  id INTEGER PRIMARY KEY,
                  filename TEXT NOT NULL UNIQUE,
                  title TEXT NOT NULL,
                  product TEXT NOT NULL,
                  document_type TEXT NOT NULL,
                  jurisdiction TEXT NOT NULL,
                  source_url TEXT NOT NULL,
                  retrieved_date TEXT NOT NULL,
                  notes TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                  id INTEGER PRIMARY KEY,
                  document_id INTEGER NOT NULL,
                  chunk_index INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  word_count INTEGER NOT NULL,
                  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                  UNIQUE(document_id, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                  chunk_id INTEGER PRIMARY KEY,
                  model TEXT NOT NULL,
                  dimensions INTEGER NOT NULL,
                  vector BLOB NOT NULL,
                  FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );

                """
            )

    def reset_index(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                DELETE FROM embeddings;
                DELETE FROM chunks;
                DELETE FROM documents;
                """
            )

    def upsert_document(self, conn: sqlite3.Connection, record: DocumentRecord) -> int:
        conn.execute(
            """
            INSERT INTO documents (
              filename, title, product, document_type, jurisdiction,
              source_url, retrieved_date, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
              title=excluded.title,
              product=excluded.product,
              document_type=excluded.document_type,
              jurisdiction=excluded.jurisdiction,
              source_url=excluded.source_url,
              retrieved_date=excluded.retrieved_date,
              notes=excluded.notes
            """,
            (
                record.filename,
                record.title,
                record.product,
                record.document_type,
                record.jurisdiction,
                record.source_url,
                record.retrieved_date,
                record.notes,
            ),
        )
        row = conn.execute(
            "SELECT id FROM documents WHERE filename = ?", (record.filename,)
        ).fetchone()
        return int(row["id"])

    def insert_chunks(self, chunks: list[TextChunk]) -> list[int]:
        with self.connect() as conn:
            chunk_ids: list[int] = []
            for chunk in chunks:
                document_id = self.upsert_document(conn, chunk.document)
                conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, text, word_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(document_id, chunk_index) DO UPDATE SET
                      text=excluded.text,
                      word_count=excluded.word_count
                    """,
                    (document_id, chunk.chunk_index, chunk.text, chunk.word_count),
                )
                row = conn.execute(
                    """
                    SELECT id FROM chunks
                    WHERE document_id = ? AND chunk_index = ?
                    """,
                    (document_id, chunk.chunk_index),
                ).fetchone()
                chunk_id = int(row["id"])
                chunk_ids.append(chunk_id)
            return chunk_ids

    def insert_embeddings(
        self, chunk_ids: list[int], model_name: str, vectors: np.ndarray
    ) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must have the same length")
        with self.connect() as conn:
            for chunk_id, vector in zip(chunk_ids, vectors):
                conn.execute(
                    """
                    INSERT INTO embeddings (chunk_id, model, dimensions, vector)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                      model=excluded.model,
                      dimensions=excluded.dimensions,
                      vector=excluded.vector
                    """,
                    (chunk_id, model_name, int(len(vector)), vector_to_blob(vector)),
                )

    def stats(self) -> dict[str, int | str | None]:
        with self.connect() as conn:
            docs = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
            chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
            embeddings = conn.execute("SELECT COUNT(*) AS count FROM embeddings").fetchone()[
                "count"
            ]
            row = conn.execute(
                "SELECT model, dimensions FROM embeddings LIMIT 1"
            ).fetchone()
            return {
                "documents": int(docs),
                "chunks": int(chunks),
                "embeddings": int(embeddings),
                "embedding_model": row["model"] if row else None,
                "embedding_dimensions": int(row["dimensions"]) if row else None,
            }

    def fetch_embeddings(self) -> tuple[list[StoredChunk], np.ndarray]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  c.id AS chunk_id,
                  c.document_id,
                  c.chunk_index,
                  c.text,
                  c.word_count,
                  d.filename,
                  d.title,
                  d.product,
                  d.document_type,
                  d.jurisdiction,
                  d.source_url,
                  e.vector
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN embeddings e ON e.chunk_id = c.id
                ORDER BY c.id
                """
            ).fetchall()
        chunks = [
            StoredChunk(
                chunk_id=int(row["chunk_id"]),
                document_id=int(row["document_id"]),
                chunk_index=int(row["chunk_index"]),
                text=row["text"],
                word_count=int(row["word_count"]),
                filename=row["filename"],
                title=row["title"],
                product=row["product"],
                document_type=row["document_type"],
                jurisdiction=row["jurisdiction"],
                source_url=row["source_url"],
            )
            for row in rows
        ]
        vectors = [blob_to_vector(row["vector"]) for row in rows]
        matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0))
        return chunks, matrix
