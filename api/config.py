from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram
    bot_token: str = ""
    admin_ids: str = ""

    # Model paths
    model_classifier: str = "ml/image_model/models/classifier_cifar10.h5"
    model_colorizer: str = "ml/image_model/models/colorizer_oxford_iiit_pet_best.h5"

    # Internal API
    api_base_url: str = "http://localhost:8000"

    # PostgreSQL
    postgres_user: str = "mlsandbox"
    postgres_password: str = "mlsandbox"
    postgres_db: str = "mlsandbox"
    postgres_host: str = "db"
    postgres_port: int = 5432

    # Training runner
    training_runner: str = "native"
    training_wsl_distribution: str = ""
    training_wsl_project_dir: str = ""
    training_wsl_python: str = ".venv/bin/python"
    training_log_tail_lines: int = 24

    @property
    def database_url(self) -> str:
        """Async URL used by SQLAlchemy + asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Sync URL used by Alembic migrations."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "protected_namespaces": ("settings_",),
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
