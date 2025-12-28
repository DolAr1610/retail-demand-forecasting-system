from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    ENV (optional):
      FAVORITA_ARTIFACTS_DIR=data/artifacts/active
      FAVORITA_CORS_ALLOW_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
      FAVORITA_LOG_LEVEL=INFO
      FAVORITA_DEFAULT_STORE_NBR=45
    """
    model_config = SettingsConfigDict(env_prefix="FAVORITA_", case_sensitive=False)

    artifacts_dir: str = "data/artifacts/active"
    cors_allow_origins: str = "*"  
    log_level: str = "INFO"

    default_store_nbr: int = 45

    def cors_origins_list(self) -> List[str]:
        val = (self.cors_allow_origins or "").strip()
        if val == "*" or val == "":
            return ["*"]
        return [x.strip() for x in val.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
