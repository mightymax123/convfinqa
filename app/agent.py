"""
Pydantic AI agent setup for the ConvFinQA financial question-answering pipeline.
"""

import asyncio
from enum import Enum

import openai
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings, OpenRouterProviderConfig
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import UsageLimits

from app.settings import get_settings
from app.tools import add, divide, exp, greater, multiply, percentage_change, subtract


class ModelName(str, Enum):
    """Enum for supported model names, using OpenRouter provider-prefixed identifiers."""

    GPT_4_1 = "openai/gpt-4.1"
    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    O4_MINI = "openai/o4-mini"
    GPT_5_4 = "openai/gpt-5.4"
    GPT_5_4_MINI = "openai/gpt-5.4-mini"
    GPT_5_5 = "openai/gpt-5.5"
    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4.5"
    CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4.6"
    GEMINI_3_1_PRO = "google/gemini-3.1-pro-preview"
    GEMINI_3_1_FLASH_LITE = "google/gemini-3.1-flash-lite-preview"


_SYSTEM_PROMPT = (
    "You are a financial question-answering assistant.\n"
    "You will receive a financial document followed by a sequence of related questions "
    "separated by the token `{next_question}`.\n"
    "Answer each question in order, one answer per question.\n\n"
    "Document reading:\n"
    "The document contains a table and surrounding text (pre_text and post_text). "
    "You must search the entire document — including all prose text — to find the values "
    "needed to answer each question. Do not limit yourself to the table alone.\n\n"
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
    "Do not include currency symbols or units unless the question explicitly asks for them.\n\n"
    "Important:\n"
    "You must always provide a real answer for every question. "
    "Never return 'placeholder' or any other non-answer — if you are uncertain, give your "
    "best answer based on the document. "
    "Returning 'placeholder' is never acceptable."
)


class LlmAnswers(BaseModel):
    """Structured output returned by the financial QA agent."""

    answers: list[str]


def build_agent(
    model_name: ModelName,
    max_retries: int,
) -> Agent[None, LlmAnswers]:
    """Construct a pydantic-ai Agent for the given model and retry settings.

    Routes all requests through OpenRouter using OpenRouterModel, which handles
    provider-prefixed model names (e.g. anthropic/, google/, openai/) and exposes
    first-class OpenRouter routing settings.

    Amazon Bedrock is excluded from provider routing to avoid a known Bedrock bug
    where tool call responses omit the arguments field, causing placeholder answers.

    max_retries is applied at two levels:
    - HTTP level: passed to the underlying AsyncOpenAI client for 429/5xx retries
    - Agent level: passed to pydantic-ai for tool-call and output-validation retries

    Args:
        model_name: The model to use, identified by its OpenRouter provider-prefixed string.
        max_retries: Number of retries for both HTTP-level and agent-level failures.

    Returns:
        A configured Agent that returns validated LlmAnswers output.
    """
    settings = get_settings()
    openai_client = openai.AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=max_retries,
    )
    model_settings = OpenRouterModelSettings(
        openrouter_provider=OpenRouterProviderConfig(ignore=["amazon-bedrock"]),
    )
    model = OpenRouterModel(
        model_name.value,
        provider=OpenRouterProvider(openai_client=openai_client),
        settings=model_settings,
    )
    return Agent(
        model,
        output_type=LlmAnswers,
        instructions=_SYSTEM_PROMPT,
        retries=max_retries,
        tools=[add, subtract, multiply, divide, percentage_change, greater, exp],
    )


_INITIAL_RETRY_DELAY_SECONDS = 1.0


async def _run_agent(agent: Agent[None, LlmAnswers], prompt: str) -> LlmAnswers | None:
    try:
        result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=25))
    except UsageLimitExceeded:
        return None
    return result.output


async def get_response(
    agent: Agent[None, LlmAnswers],
    prompt: str,
    max_retries: int = 10,
) -> LlmAnswers | None:
    """Run the agent with a prompt and return structured answers.

    Retries on rate limit errors using exponential backoff, doubling the wait
    time on each attempt (1s, 2s, 4s, 8s, ...). This outer retry loop is
    separate from the openai SDK's internal retry mechanism, which handles
    short bursts; this loop handles sustained rate limits the SDK cannot outlast.

    Returns None if the request limit is exceeded, allowing the caller to
    handle the skipped conversation gracefully. All other exceptions bubble up.

    Args:
        agent: The configured pydantic-ai Agent.
        prompt: The user prompt containing the financial document and questions.
        max_retries: Maximum number of rate-limit retries before re-raising.

    Returns:
        Validated LlmAnswers containing the list of answers, or None if the
        request limit was exceeded.

    Raises:
        openai.RateLimitError: If the rate limit is still hit after all retries.
    """
    delay = _INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(max_retries + 1):
        try:
            result = await _run_agent(agent, prompt)
        except openai.RateLimitError:
            if attempt == max_retries:
                raise
            logger.warning(f"Rate limit hit — retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
            delay *= 2
            continue

        return result
