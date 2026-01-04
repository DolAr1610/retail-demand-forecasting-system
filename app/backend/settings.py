from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    """
    Supports BOTH env styles:
      - MODELS_DIR / FRONTEND_DIR
      - APP_MODELS_DIR / APP_FRONTEND_DIR
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_prefix: str = Field(default="/api", validation_alias=AliasChoices("API_PREFIX", "APP_API_PREFIX"))
    ui_prefix: str = Field(default="/ui", validation_alias=AliasChoices("UI_PREFIX", "APP_UI_PREFIX"))

    models_dir: str = Field(default="data/models", validation_alias=AliasChoices("MODELS_DIR", "APP_MODELS_DIR"))
    frontend_dir: str = Field(default="app/frontend", validation_alias=AliasChoices("FRONTEND_DIR", "APP_FRONTEND_DIR"))

    cors_origins: str | None = Field(default=None, validation_alias=AliasChoices("CORS_ORIGINS", "APP_CORS_ORIGINS"))

    @property
    def cors_origins_list(self) -> list[str]:
        v = getattr(self, "cors_origins", "")
        return [x.strip() for x in v.split(",") if x.strip()]


settings = Settings()
