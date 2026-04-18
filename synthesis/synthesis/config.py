from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables."""

    # Anthropic
    anthropic_api_key: str = ""
    default_model: str = "claude-haiku-4-5-20251001"
    premium_model: str = "claude-opus-4-6-20250414"

    # Postgres
    database_url: str = "postgresql+asyncpg://localhost:5432/personas"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "personas"

    # Embedding
    embedding_model: str = "voyage-3"
    voyage_api_key: str = ""

    # Synthesis
    max_retries: int = 2
    synthesis_timeout_seconds: int = 60

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
