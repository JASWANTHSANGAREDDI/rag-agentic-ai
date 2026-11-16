"""
config.py — Centralised, validated application settings.

PRODUCTION RULES (Google SRE Book, Chapter 5):
  1. Fail fast: if a required secret is missing, crash at startup — not at
     the first request at 3 AM when the on-call is asleep.
  2. Single source of truth: every configurable value lives here.
  3. Never read os.getenv() outside this file.

We use Pydantic's BaseSettings so we get:
  - Type coercion (str "0" → int 0, str "true" → bool True)
  - Required-field validation (missing key → clear error on startup)
  - .env file loading out of the box
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Required keys raise a ValidationError at import time if absent.
    Optional keys have sensible production defaults.
    """

    model_config = SettingsConfigDict(
        # Look for .env relative to this file's parent (project root)
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,       # OPENROUTER_API_KEY == openrouter_api_key
        extra="ignore",             # ignore unknown env vars — don't crash on typos
    )

    # ------------------------------------------------------------------
    # REQUIRED secrets — loaded from .env/env, verified non-empty
    # ------------------------------------------------------------------
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    pinecone_api_key: str   = Field(default="", description="Pinecone API key")

    # ------------------------------------------------------------------
    # OpenRouter / LLM
    # ------------------------------------------------------------------
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter-compatible base URL",
    )
    model_name: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model identifier (any OpenRouter-supported model)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0 = deterministic)",
    )

    # ------------------------------------------------------------------
    # Pinecone
    # ------------------------------------------------------------------
    pinecone_index_name: str = Field(
        default="agentic-ai-index",
        description="Name of the Pinecone index",
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model name",
    )
    embedding_dimension: int = Field(
        default=384,
        gt=0,
        description="Vector dimension — must match the Pinecone index",
    )

    # ------------------------------------------------------------------
    # Chunking (used by ingestion.py)
    # ------------------------------------------------------------------
    chunk_size: int    = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50, description="Default chunks to retrieve")

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="info")
    # Allowed CORS origins — comma-separated in env, e.g. "http://localhost:3000,https://myapp.com"
    # Set to "*" only during local development; never in production with credentials.
    cors_origins: str = Field(default="*")

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------
    max_retries: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Max agentic retry iterations before accepting any answer",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided in .env or environment")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY must be provided in .env or environment")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience properties (use these instead of raw attribute names)
    # ------------------------------------------------------------------

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list (split on comma, strip whitespace)."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    @lru_cache ensures we parse .env exactly once per process.
    Call get_settings() everywhere instead of using the module-level
    `settings` object — it makes testing easy (just clear the cache).
    """
    return Settings()


# Module-level singleton — imported as `from src.config import settings`
# Tests can override via: get_settings.cache_clear(); monkeypatch env vars.
settings: Settings = get_settings()
