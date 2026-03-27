from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Expected environment variable prefix: FAVORITA_
    Example:
      FAVORITA_OPENROUTER_API_KEY=...
      FAVORITA_OPENROUTER_MODEL=openrouter/free
      FAVORITA_ARTIFACTS_DIR=data/artifacts/active
    """

    model_config = SettingsConfigDict(
        env_prefix="FAVORITA_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # OpenRouter / LLM
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/free"
    openrouter_app_name: str = "Retail Demand Forecast MVP"
    openrouter_site_url: str | None = None

    # App / artifacts
    artifacts_dir: str = "data/artifacts/active"
    log_level: str = "INFO"
    default_store_nbr: int = 45

    # CORS
    cors_allow_origins: str = "*"

    def cors_origins_list(self) -> List[str]:
        val = (self.cors_allow_origins or "").strip()
        if val == "*" or val == "":
            return ["*"]
        return [x.strip() for x in val.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()