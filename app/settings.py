"""
Configuration management using Pydantic v2 for environment variables.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    App configuration loaded from environment variables.

    Attributes:
        openrouter_api_key: API key for OpenRouter — grants access to all supported providers.
        max_retries: Number of retries for both pydantic-ai tool-call / output-validation
            attempts and OpenAI SDK HTTP retries (e.g. on 429 / 5xx responses).
    """

    openrouter_api_key: str = Field(min_length=1)

    max_retries: int = Field(default=10, ge=0, le=10)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False, "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
