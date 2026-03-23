from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class BotSettings(BaseSettings):
    bot_token: str
    admin_ids: str = ""
    api_base_url: str = "http://api:8000"

    # Model paths (used for local existence checks)
    model_classifier: str = "ml/image_model/models/classifier.h5"
    model_colorizer: str = "ml/image_model/models/colorizer_oxford_iiit_pet_best.h5"

    # PostgreSQL
    postgres_user: str = "mlsandbox"
    postgres_password: str = "mlsandbox"
    postgres_db: str = "mlsandbox"
    postgres_host: str = "db"
    postgres_port: int = 5432

    @property
    def admin_id_list(self) -> list[int]:
        return [int(x.strip()) for x in self.admin_ids.split(",") if x.strip().isdigit()]

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "protected_namespaces": ("settings_",),
    }


@lru_cache
def get_settings() -> BotSettings:
    return BotSettings()


settings = get_settings()
