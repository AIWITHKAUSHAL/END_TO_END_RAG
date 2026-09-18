"""Load application settings from environment variables and the local .env."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration shared by the UI and CLI workflows."""

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    generation_model: str = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    data_dir: str = os.getenv("DATA_DIR", "data/sample_docs")
    index_dir: str = os.getenv("INDEX_DIR", "indexes")
    default_chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    default_chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))


settings = Settings()
