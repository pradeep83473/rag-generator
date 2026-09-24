from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables come from environment variables (or a .env file).

    Nothing about a specific document set lives in code, which is what lets the
    same deployment serve any number of different knowledge bases.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_base_url: str | None = None  # optional: OpenAI-compatible gateway
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # Some reasoning models only accept the default temperature; set to empty to omit.
    llm_temperature: float | None = 0.0
    embedding_batch_size: int = 64
    openai_timeout_seconds: float = 60.0

    # --- Storage ---
    data_dir: Path = Path("data")

    # --- Ingestion ---
    chunk_size: int = 1000  # characters (~250 tokens)
    chunk_overlap: int = 150
    max_upload_mb: int = 25

    # --- Retrieval / generation ---
    retrieval_top_k: int = 6  # chunks sent to the LLM
    retrieval_candidates: int = 25  # per-retriever candidate pool before fusion
    history_turns: int = 6  # prior messages used for follow-up questions


@lru_cache
def get_settings() -> Settings:
    return Settings()
