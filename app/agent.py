"""
Pydantic AI agent setup for the ConvFinQA financial question-answering pipeline.
"""

from enum import Enum

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.settings import get_settings
from app.tools import add, divide, exp, greater, multiply, percentage_change, subtract


class ModelName(str, Enum):
    """Enum for supported OpenAI model names."""

    GPT_4_1 = "gpt-4.1"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O4_MINI = "o4-mini"


_SYSTEM_PROMPT = (
    "You are a financial question-answering assistant.\n"
    "You will receive a sequence of related questions in a single string, "
    "separated by the token `{next_question}`.\n"
    "Your task is to answer each question in order.\n"
    "Return your answers as a list of strings, one per question.\n"
    "For any numerical computation (addition, subtraction, multiplication, division, "
    "percentage change, comparisons, or exponentiation), you must call the appropriate "
    "tool rather than computing the value yourself. "
    "Only produce your final structured answer once all required tool calls have been made."
)


class LlmAnswers(BaseModel):
    """Structured output returned by the financial QA agent."""

    answers: list[str]


def build_agent(model_name: ModelName, max_retries: int) -> Agent[None, LlmAnswers]:
    """Construct a pydantic-ai Agent for the given model and retry settings.

    Args:
        model_name: The OpenAI model to use.
        max_retries: Maximum number of retry attempts on transient failures.

    Returns:
        A configured Agent that returns validated LlmAnswers output.
    """
    settings = get_settings()
    model = OpenAIModel(
        model_name.value,
        provider=OpenAIProvider(api_key=settings.openai_api_key),
    )
    return Agent(
        model,
        output_type=LlmAnswers,
        instructions=_SYSTEM_PROMPT,
        retries=max_retries,
        tools=[add, subtract, multiply, divide, percentage_change, greater, exp],
    )


def get_response(agent: Agent[None, LlmAnswers], prompt: str) -> LlmAnswers:
    """Run the agent with a prompt and return structured answers.

    Args:
        agent: The configured pydantic-ai Agent.
        prompt: The user prompt containing the financial document and questions.

    Returns:
        Validated LlmAnswers containing the list of answers.
    """
    result = agent.run_sync(prompt)
    return result.output
