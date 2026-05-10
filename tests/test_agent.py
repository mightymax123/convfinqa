"""
Tests for the pydantic-ai agent setup in app/agent.py.
"""

# Prevent accidental real model requests in this test module.
from collections.abc import Generator

import openai
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.test import TestModel

from app.agent import LlmAnswers, ModelName, build_agent, get_response
from app.settings import Settings

_EXPECTED_TOOLS = {"add", "subtract", "multiply", "divide", "percentage_change", "greater", "exp"}


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """
    GIVEN tests that call build_agent(),
    WHEN any test in this module runs,
    THEN get_settings() is patched to return a dummy Settings instance so no real
         environment variable or .env file is required.
    """
    from unittest.mock import patch

    from app.settings import get_settings

    get_settings.cache_clear()
    dummy = Settings(openrouter_api_key="test-key")
    with patch("app.agent.get_settings", return_value=dummy):
        yield
    get_settings.cache_clear()


class TestLlmAnswers:
    def test_valid_answers_accepted(self) -> None:
        """
        GIVEN a list of string answers,
        WHEN LlmAnswers is initialised,
        THEN the answers field is set correctly.
        """
        result = LlmAnswers(answers=["10", "20"])

        assert result.answers == ["10", "20"]

    def test_empty_answers_accepted(self) -> None:
        """
        GIVEN an empty list,
        WHEN LlmAnswers is initialised,
        THEN the answers field is an empty list.
        """
        result = LlmAnswers(answers=[])

        assert result.answers == []


class TestBuildAgent:
    def test_build_agent_returns_agent_instance(self) -> None:
        """
        GIVEN a valid model name and max_retries value,
        WHEN build_agent is called,
        THEN an Agent instance is returned.
        """
        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=2)

        assert isinstance(agent, Agent)

    def test_build_agent_registers_all_tools(self) -> None:
        """
        GIVEN a valid model name and max_retries value,
        WHEN build_agent is called,
        THEN all expected arithmetic tools are registered on the agent.
        """
        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=1)
        registered = set(agent._function_toolset.tools.keys())

        assert registered == _EXPECTED_TOOLS

    def test_build_agent_configures_model(self) -> None:
        """
        GIVEN a valid model name,
        WHEN build_agent is called,
        THEN the agent's model is an OpenRouterModel with the correct model name.
        """
        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=5)
        assert isinstance(agent.model, OpenRouterModel)
        assert agent.model.model_name == ModelName.GPT_4_1.value


class TestGetResponse:
    async def test_get_response_returns_llm_answers(self) -> None:
        """
        GIVEN a TestModel configured to return structured answers,
        WHEN get_response is called with a prompt,
        THEN an LlmAnswers instance with the expected answers is returned.
        """
        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=1)
        test_model = TestModel(custom_output_args={"answers": ["42", "84"]})

        with agent.override(model=test_model, tools=[]):
            result = await get_response(agent, "What is revenue? {next_question} What is profit?", max_retries=1)

        assert isinstance(result, LlmAnswers)
        assert result.answers == ["42", "84"]

    async def test_get_response_handles_single_answer(self) -> None:
        """
        GIVEN a TestModel configured to return a single answer,
        WHEN get_response is called,
        THEN an LlmAnswers instance with one answer is returned.
        """
        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=1)
        test_model = TestModel(custom_output_args={"answers": ["100"]})

        with agent.override(model=test_model, tools=[]):
            result = await get_response(agent, "What is the total revenue?", max_retries=1)

        assert result is not None
        assert result.answers == ["100"]

    async def test_get_response_returns_none_on_usage_limit_exceeded(self) -> None:
        """
        GIVEN an agent run with a request limit of zero,
        WHEN get_response is called,
        THEN None is returned rather than raising UsageLimitExceeded.
        """
        from unittest.mock import patch

        from pydantic_ai.usage import UsageLimits

        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=1)
        test_model = TestModel(custom_output_args={"answers": ["42"]})

        with agent.override(model=test_model, tools=[]):
            with patch("app.agent.UsageLimits", return_value=UsageLimits(request_limit=0)):
                result = await get_response(agent, "What is revenue?", max_retries=1)

        assert result is None

    async def test_get_response_retries_on_rate_limit_then_succeeds(self) -> None:
        """
        GIVEN an agent that raises RateLimitError on the first call then succeeds,
        WHEN get_response is called with max_retries=2,
        THEN the result is returned after one retry and the backoff sleep is called once.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=1)
        success_output = MagicMock()
        success_output.output = LlmAnswers(answers=["42"])

        rate_limit_error = openai.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        call_count = 0

        async def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise rate_limit_error
            return success_output

        with patch.object(agent, "run", side_effect=fake_run):
            with patch("app.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await get_response(agent, "What is revenue?", max_retries=2)

        assert result is not None
        assert result.answers == ["42"]
        mock_sleep.assert_awaited_once_with(1.0)

    async def test_get_response_backoff_doubles_on_each_retry(self) -> None:
        """
        GIVEN an agent that raises RateLimitError on the first two calls then succeeds,
        WHEN get_response is called with max_retries=3,
        THEN asyncio.sleep is called with 1.0 then 2.0, confirming exponential doubling.
        """
        from unittest.mock import AsyncMock, MagicMock, call, patch

        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=1)
        success_output = MagicMock()
        success_output.output = LlmAnswers(answers=["42"])

        rate_limit_error = openai.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        call_count = 0

        async def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise rate_limit_error
            return success_output

        with patch.object(agent, "run", side_effect=fake_run):
            with patch("app.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await get_response(agent, "What is revenue?", max_retries=3)

        assert result is not None
        assert result.answers == ["42"]
        assert mock_sleep.await_args_list == [call(1.0), call(2.0)]

    async def test_get_response_raises_after_all_retries_exhausted(self) -> None:
        """
        GIVEN an agent that always raises RateLimitError,
        WHEN get_response is called with max_retries=2,
        THEN RateLimitError is re-raised after the retry budget is exhausted.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        agent = build_agent(model_name=ModelName.GPT_4_1, max_retries=1)
        rate_limit_error = openai.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        with patch.object(agent, "run", side_effect=rate_limit_error):
            with patch("app.agent.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(openai.RateLimitError):
                    await get_response(agent, "What is revenue?", max_retries=2)
