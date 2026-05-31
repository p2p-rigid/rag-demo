from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
    "zurich",
}


@dataclass
class LocalHashingEmbedder:
    model_name: str

    def __post_init__(self) -> None:
        match = re.search(r"(\d+)$", self.model_name)
        self.n_features = int(match.group(1)) if match else 4096

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        vectors = np.zeros((len(texts), self.n_features), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in feature_tokens(text):
                vectors[row, stable_hash(token) % self.n_features] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


def feature_tokens(text: str) -> list[str]:
    tokens = [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


SentenceTransformerEmbedder = LocalHashingEmbedder


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(query_norm * matrix_norms, 1e-12)
    return (matrix @ query) / denom
