"""
Pydantic AI agent setup for the ConvFinQA financial question-answering pipeline.
"""

from enum import Enum

import openai
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

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
    "You will receive a financial document followed by a sequence of related questions "
    "separated by the token `{next_question}`.\n"
    "Answer each question in order, one answer per question.\n\n"
    "Tool use:\n"
    "For any numerical computation — addition, subtraction, multiplication, division, "
    "percentage change, comparisons, or exponentiation — you must call the appropriate "
    "tool rather than computing the value yourself. "
    "Only produce your final answer once all required tool calls have been made.\n"
    "If a tool returns None (e.g. division by zero), record the answer for that question "
    "as 'N/A' and continue to the next question.\n\n"
    "Answer format:\n"
    "Answers should be concise numeric values. "
    "Include a '%' suffix when the answer is a percentage (e.g. '12.5%'). "
    "The percentage_change tool returns a raw number — e.g. 50.0 means 50%, so express "
    "it as '50.0%' in your answer. "
    "Do not include currency symbols or units unless the question explicitly asks for them."
)


class LlmAnswers(BaseModel):
    """Structured output returned by the financial QA agent."""

    answers: list[str]


def build_agent(
    model_name: ModelName,
    max_retries: int,
) -> Agent[None, LlmAnswers]:
    """Construct a pydantic-ai Agent for the given model and retry settings.

    Args:
        model_name: The OpenAI model to use.
        max_retries: Number of retries for both tool-call / output-validation
            attempts and OpenAI SDK HTTP retries (e.g. on 429 / 5xx responses).

    Returns:
        A configured Agent that returns validated LlmAnswers output.
    """
    settings = get_settings()
    model = OpenAIModel(
        model_name.value,
        provider=OpenAIProvider(
            openai_client=openai.AsyncOpenAI(
                api_key=settings.openai_api_key,
                max_retries=max_retries,
            )
        ),
    )
    return Agent(
        model,
        output_type=LlmAnswers,
        instructions=_SYSTEM_PROMPT,
        retries=max_retries,
        tools=[add, subtract, multiply, divide, percentage_change, greater, exp],
    )


async def get_response(agent: Agent[None, LlmAnswers], prompt: str) -> LlmAnswers | None:
    """Run the agent with a prompt and return structured answers.

    Returns None if the request limit is exceeded, allowing the caller to
    handle the skipped conversation gracefully. All other exceptions bubble up.

    Args:
        agent: The configured pydantic-ai Agent.
        prompt: The user prompt containing the financial document and questions.

    Returns:
        Validated LlmAnswers containing the list of answers, or None if the
        request limit was exceeded.
    """
    try:
        result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=25))
    except UsageLimitExceeded:
        return None
    return result.output
