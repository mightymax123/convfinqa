"""
Configuration management using Pydantic v2 for environment variables.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    App configuration loaded from environment variables.

    Attributes:
        log_level (str): Logging level for the application.
        openai_api_key (str): API key for OpenAI.
        data_path (str): Path to the ConvFinQa dataset.
        random_seed (int): Random seed for reproducibility.
        max_retries (int): Maximum number of retry attempts for API calls.
        base_delay (float): Initial delay in seconds for exponential backoff.
    """

    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"

    openai_api_key: str = Field(min_length=1)

    data_path: str = "/app/data/convfinqa_dataset.json"

    random_seed: int = Field(default=42, ge=0)

    max_retries: int = Field(default=3, ge=0, le=10)

    base_delay: float = Field(default=2.0, gt=0)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False, "extra": "ignore"}


config = Config()
