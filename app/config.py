"""Application settings, typed and validated once at startup.

Every value the service reads from its environment is declared here, grouped by the part of
the system it controls. Nothing else in the codebase touches ``os.environ``. Secrets are
``SecretStr`` so they cannot leak into logs, tracebacks or error responses.

Values come from environment variables, or from a ``.env`` file in the working directory for
local development. ``.env.example`` documents every variable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]
ToolsBackend = Literal["inprocess", "mcp"]


class Settings(BaseSettings):
    """All runtime configuration for the service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Runtime ------------------------------------------------------------------------
    environment: Environment = Field(
        default="dev",
        description="prod refuses to start with missing secrets; dev warns; test is silent.",
    )
    log_level: str = "INFO"
    log_json: bool | None = Field(
        default=None,
        description="Force JSON log lines. Unset means JSON in prod, coloured console elsewhere.",
    )

    # ---- HTTP API -----------------------------------------------------------------------
    api_key: SecretStr | None = Field(
        default=None, description="When set, POST /ask requires the X-API-Key header."
    )
    rate_limit: str = Field(default="30/minute", description="Per-client limit on POST /ask.")
    max_question_chars: int = 1000
    cors_origins: list[str] = Field(default_factory=list)

    # ---- Model providers ----------------------------------------------------------------
    gemini_api_key: SecretStr | None = Field(default=None, description="Primary provider.")
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    groq_api_key: SecretStr | None = Field(default=None, description="Failover provider.")
    groq_model: str = "openai/gpt-oss-120b"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # ---- Superhero API ------------------------------------------------------------------
    superhero_api_token: SecretStr | None = None
    superhero_base_url: str = "https://superheroapi.com/api"
    superhero_timeout_s: float = 10.0
    superhero_cache_ttl_s: int = 600
    superhero_cache_size: int = 512
    superhero_max_retries: int = 2
    superhero_breaker_failures: int = Field(
        default=5, description="Consecutive failures before the circuit breaker opens."
    )
    superhero_breaker_recovery_s: float = Field(
        default=30.0, description="Seconds the breaker stays open before probing again."
    )

    # ---- Retrieval ----------------------------------------------------------------------
    data_dir: Path = Path("data")
    bm25_top_k: int = Field(default=100, description="Candidates from the lexical retriever.")
    dense_top_k: int = Field(default=100, description="Candidates from the dense retriever.")
    rrf_k: int = Field(default=60, description="Reciprocal rank fusion constant, Cormack 2009.")
    rerank_top_k: int = Field(default=20, description="Passages kept after cross-encoder rerank.")
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"

    # ---- Agent graph --------------------------------------------------------------------
    max_agent_steps: int = Field(default=4, description="Tool-call rounds in the superhero agent.")
    max_query_rewrites: int = 1
    max_regenerations: int = Field(default=1, description="Retries when grounding check fails.")

    # ---- Tools --------------------------------------------------------------------------
    tools_backend: ToolsBackend = Field(
        default="inprocess",
        description="inprocess calls tool functions directly; mcp loads them over MCP.",
    )
    mcp_server_url: str | None = Field(
        default=None,
        description="Streamable HTTP MCP server. Unset with tools_backend=mcp spawns stdio.",
    )

    # ---- Caching and sessions -----------------------------------------------------------
    response_cache_ttl_s: int = 300
    response_cache_size: int = 1024
    session_ttl_s: int = 1800

    # ---- Observability ------------------------------------------------------------------
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "ai-engineer-assessment"
    metrics_enabled: bool = True

    @field_validator(
        "api_key",
        "gemini_api_key",
        "groq_api_key",
        "superhero_api_token",
        "mcp_server_url",
        "otel_exporter_otlp_endpoint",
        mode="before",
    )
    @classmethod
    def _empty_string_is_unset(cls, value: object) -> object:
        """Treat ``KEY=`` in a .env file as not set rather than as an empty secret."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def log_as_json(self) -> bool:
        """JSON logs when forced, otherwise only in production."""
        return self.log_json if self.log_json is not None else self.environment == "prod"

    @property
    def has_llm_provider(self) -> bool:
        """True when at least one model provider key is configured."""
        return bool(self.gemini_api_key or self.groq_api_key)

    def missing_runtime_secrets(self) -> list[str]:
        """Names of the secrets the service needs to do useful work but does not have."""
        missing: list[str] = []
        if not self.has_llm_provider:
            missing.append("GEMINI_API_KEY or GROQ_API_KEY")
        if not self.superhero_api_token:
            missing.append("SUPERHERO_API_TOKEN")
        return missing


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings instance. Tests bypass this and pass Settings explicitly."""
    return Settings()
