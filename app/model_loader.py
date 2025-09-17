from abc import ABC, abstractmethod
from enum import Enum

from openai import OpenAI

from .logger import get_logger

logger = get_logger(__name__)


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

    def __init__(self, model_name: str = "gpt-4.1") -> None:
        """
        Initialize the OpenAiLlmResponse class with a specified model name.

        Args:
            model_name (str): The model to use for generating responses, must be a value from the ModelName enums. Defaults to GPT-4.1.
        """
        try:
            valid_model = ModelName(model_name)
            logger.info(f"Using openai model model: {valid_model.value}")
        except ValueError as err:
            logger.error(f"Invalid model name '{model_name}' provided.")
            raise ValueError(
                f"Invalid model name: {model_name}. Must be one of {[model.value for model in ModelName]}."
            ) from err

        super().__init__(model_name=valid_model.value)
        self.client = OpenAI()

    def get_response(self, prompt: str) -> str:
        """
        Get a response from the OpenAI LLM for a given prompt.
        Args:
            prompt (str): The input prompt for the LLM.
        Returns:
            str: The output text from the LLM.
        Raises:
            ValueError: If the API returns an empty response.
        """
        logger.debug(f"sending prompt to openai model: {self.model_name}")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        logger.debug(f"Received response from openAI model: {self.model_name}")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Received empty response from OpenAI API")
        return content
