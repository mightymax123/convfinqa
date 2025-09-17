from abc import ABC, abstractmethod
from enum import Enum
import time

from openai import OpenAI, RateLimitError, APITimeoutError, APIError
from pydantic import BaseModel, Field

from src.logger import get_logger

logger = get_logger(__name__)


class RetryConfig(BaseModel):
    """
    Retry configuration for exponential backoff with validation.
    
    Attributes:
        max_retries (int): Maximum number of retry attempts (0-10).
        base_delay (float): Initial delay in seconds for exponential backoff (0.1-60.0).
    """
    
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retry attempts")
    base_delay: float = Field(default=2.0, ge=0.1, le=60.0, description="Initial delay in seconds for exponential backoff")


class ModelName(Enum):
    """Enum for OpenAI model names."""

    GPT_4_1 = "gpt-4.1"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O4_MINI = "o4-mini"


class GetLlmResponse(ABC):
    """
    Abstract base class for getting responses from an LLM.
    This class should be extended to implement specific LLM response retrieval methods.
    """

    system_prompt = (
        "You are a financial question-answering assistant.\n"
        "You will receive a sequence of related questions in a single string, separated by the token `{next_question}`.\n"
        "Your task is to answer each question in order.\n"
        "Return your answers as a Python list of strings, like: ['Answer1', 'Answer2', 'Answer3', ...]."
    )

    def __init__(self, model_name: str) -> None:
        """
        Initialize the GetLlmResponse class with a specified model name.

        Args:
            model_name (str): The model to use for generating responses.
        """
        self.model_name = model_name

    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """
        Abstract method to get a response from the LLM for a given prompt.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The output text from the LLM.
        """
        pass


class OpenAiLlmResponse(GetLlmResponse):
    """
    OpenAI implementation of GetLlmResponse using the OpenAI Python API.
    """

    def __init__(
        self, 
        model_name: str = "gpt-4.1", 
        retry_config: RetryConfig | None = None
    ) -> None:
        """
        Initialize the OpenAiLlmResponse class with a specified model name and retry configuration.
        
        Args:
            model_name (str): The model to use for generating responses, must be a value from the ModelName enums. Defaults to GPT-4.1.
            retry_config (RetryConfig | None): Configuration for retry behavior. If None, uses default configuration.
        """
        try:
            valid_model = ModelName(model_name)
            logger.info(f"Using OpenAI model: {valid_model.value}")
        except ValueError as err:
            available_models = [model.value for model in ModelName]
            logger.error(f"Invalid model name '{model_name}' provided.")
            raise ValueError(
                f"Invalid model name: {model_name}. Must be one of {available_models}."
            ) from err

        super().__init__(model_name=valid_model.value)
        self.client = OpenAI()
        self.retry_config = retry_config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff using base_delay * (2 ^ attempt).
        
        Args:
            attempt (int): The current attempt number (0-indexed).
            
        Returns:
            float: The delay in seconds before the next retry attempt.
        """

        delay_time = self.retry_config.base_delay * (2 ** attempt)

        return delay_time

    def get_response(self, prompt: str) -> str:
        """
        Get a response from the OpenAI LLM for a given prompt with exponential backoff retry logic.
        
        Args:
            prompt (str): The input prompt for the LLM.
            
        Returns:
            str: The output text from the LLM.
            
        Raises:
            ValueError: If the API returns an empty response.
            RateLimitError: If rate limit is exceeded after all retries.
            APITimeoutError: If API timeout occurs after all retries.
            APIError: If general API error occurs after all retries.
        """

        retryable_errors = (RateLimitError, APITimeoutError, APIError)

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.debug(f"Sending prompt to OpenAI model: {self.model_name} (attempt {attempt + 1})")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Received empty response from OpenAI API")
                
                if attempt > 0:
                    logger.info(f"Request succeeded after {attempt + 1} attempts")
                
                return content

            except retryable_errors as e:
                if attempt == self.retry_config.max_retries:
                    logger.error(f"Max retries ({self.retry_config.max_retries}) exceeded for model {self.model_name}")
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}). "
                    f"Retrying in {delay:.1f}s. Error: {type(e).__name__}: {str(e)}"
                )
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Non-retryable error occurred: {type(e).__name__}: {str(e)}")
                raise
        
        # This should never be reached due to the logic above, but satisfies mypy requirements for pipeline.
        raise RuntimeError("Unexpected: retry loop completed without return or exception")
