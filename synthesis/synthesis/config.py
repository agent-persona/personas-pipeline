from __future__ import annotations

from pydantic_settings import BaseSettings

from .provider_registry import validate_provider_model


class Settings(BaseSettings):
    """Application settings, loaded from environment variables."""

    # Provider routing
    model_provider: str = "anthropic"
    judge_provider: str = ""
    persona_schema_version: str = "v1"
    persona_birth_year: int = 1988
    persona_eval_year: int = 2026

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    default_model: str = "claude-haiku-4-5-20251001"
    premium_model: str = "claude-opus-4-6-20250414"
    judge_model: str = ""
    twin_model: str = ""

    # Gemini
    gemini_api_key_max: str = ""
    gemini_api_key_turkey: str = ""
    gemini_api_key_selector: str = "max"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Kimi
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.ai/v1"

    # Z.AI
    z_ai_glm_api_key: str = ""
    z_ai_anthropic_base_url: str = "https://api.z.ai/api/anthropic"

    # MiniMax
    minimax_api_key: str = ""
    minimax_anthropic_base_url: str = "https://api.minimax.io/anthropic"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def resolved_judge_provider(self) -> str:
        return self.judge_provider or self.model_provider

    @property
    def resolved_judge_model(self) -> str:
        return self.judge_model or self.default_model

    @property
    def resolved_twin_model(self) -> str:
        return self.twin_model or self.default_model

    def validate_runtime_settings(self) -> None:
        validate_provider_model(self.model_provider, self.default_model, label="Synthesis")
        validate_provider_model(self.resolved_judge_provider, self.resolved_judge_model, label="Judge")
        validate_provider_model(self.model_provider, self.resolved_twin_model, label="Twin")
        if self.persona_schema_version not in {"v1", "v2"}:
            raise ValueError(
                "PERSONA_SCHEMA_VERSION must be 'v1' or 'v2'",
            )


settings = Settings()
