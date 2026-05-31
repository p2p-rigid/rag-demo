from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    manifest_path: Path
    text_dir: Path
    database_path: Path
    embedding_model: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    top_k: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        manifest_path=Path(os.getenv("MANIFEST_PATH", "data/zurich_insurance/manifest.csv")),
        text_dir=Path(os.getenv("TEXT_DIR", "data/zurich_insurance/text")),
        database_path=Path(os.getenv("DATABASE_PATH", "data/rag.sqlite")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "hashing-vectorizer-4096"
        ),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        top_k=int(os.getenv("TOP_K", "6")),
    )
