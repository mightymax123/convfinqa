"""
Tests for the LLM judge agent in app/judge.py.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, call, patch

import openai
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.test import TestModel

from app.judge import JudgeResult, build_judge_agent, get_judge_response
from app.settings import Settings


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """
    GIVEN tests that call build_judge_agent(),
    WHEN any test in this module runs,
    THEN get_settings() is patched to return a dummy Settings instance so no real
         environment variable or .env file is required.
    """
    from app.settings import get_settings

    get_settings.cache_clear()
    dummy = Settings(openrouter_api_key="test-key")
    with patch("app.judge.get_settings", return_value=dummy):
        yield
    get_settings.cache_clear()


class TestJudgeResult:
    def test_all_true_accepted(self) -> None:
        """
        GIVEN a list of True booleans,
        WHEN JudgeResult is initialised,
        THEN the results field is set correctly.
        """
        result = JudgeResult(results=[True, True, True])

        assert result.results == [True, True, True]

    def test_mixed_results_accepted(self) -> None:
        """
        GIVEN a mixed list of booleans,
        WHEN JudgeResult is initialised,
        THEN the results field preserves order and values.
        """
        result = JudgeResult(results=[True, False, True])

        assert result.results == [True, False, True]

    def test_empty_results_accepted(self) -> None:
        """
        GIVEN an empty list,
        WHEN JudgeResult is initialised,
        THEN the results field is an empty list.
        """
        result = JudgeResult(results=[])

        assert result.results == []


class TestBuildJudgeAgent:
    def test_build_judge_agent_returns_agent_instance(self) -> None:
        """
        GIVEN valid settings with a dummy API key,
        WHEN build_judge_agent is called,
        THEN an Agent instance is returned.
        """
        agent = build_judge_agent()

        assert isinstance(agent, Agent)

    def test_build_judge_agent_uses_flash_lite_model(self) -> None:
        """
        GIVEN valid settings,
        WHEN build_judge_agent is called,
        THEN the agent's model is OpenRouterModel configured with the Flash Lite model.
        """
        from app.agent import ModelName

        agent = build_judge_agent()

        assert isinstance(agent.model, OpenRouterModel)
        assert agent.model.model_name == ModelName.GEMINI_3_1_FLASH_LITE.value

    def test_build_judge_agent_registers_no_tools(self) -> None:
        """
        GIVEN valid settings,
        WHEN build_judge_agent is called,
        THEN no tools are registered on the agent.
        """
        agent = build_judge_agent()

        assert len(agent._function_toolset.tools) == 0


class TestGetJudgeResponse:
    async def test_get_judge_response_returns_all_correct(self) -> None:
        """
        GIVEN ground-truth and predicted lists that are all equivalent,
        WHEN get_judge_response is called with a TestModel returning all True,
        THEN a JudgeResult with all True results is returned.
        """
        agent = build_judge_agent()
        test_model = TestModel(custom_output_args={"results": [True, True, True]})

        with agent.override(model=test_model):
            result = await get_judge_response(
                agent,
                ground_truth=["10", "20", "30"],
                predicted=["10", "20.0", "30"],
                max_retries=1,
            )

        assert result is not None
        assert result.results == [True, True, True]

    async def test_get_judge_response_returns_all_incorrect(self) -> None:
        """
        GIVEN ground-truth and predicted lists that are all different,
        WHEN get_judge_response is called with a TestModel returning all False,
        THEN a JudgeResult with all False results is returned.
        """
        agent = build_judge_agent()
        test_model = TestModel(custom_output_args={"results": [False, False, False]})

        with agent.override(model=test_model):
            result = await get_judge_response(
                agent,
                ground_truth=["10", "20", "30"],
                predicted=["WRONG", "WRONG", "WRONG"],
                max_retries=1,
            )

        assert result is not None
        assert result.results == [False, False, False]

    async def test_get_judge_response_returns_mixed_results(self) -> None:
        """
        GIVEN ground-truth and predicted lists with partial matches,
        WHEN get_judge_response is called with a TestModel returning mixed booleans,
        THEN a JudgeResult with mixed results is returned.
        """
        agent = build_judge_agent()
        test_model = TestModel(custom_output_args={"results": [True, False, True]})

        with agent.override(model=test_model):
            result = await get_judge_response(
                agent,
                ground_truth=["10", "20", "30"],
                predicted=["10", "WRONG", "30"],
                max_retries=1,
            )

        assert result is not None
        assert result.results == [True, False, True]

    async def test_get_judge_response_retries_on_rate_limit_then_succeeds(self) -> None:
        """
        GIVEN an agent that raises RateLimitError on the first call then succeeds,
        WHEN get_judge_response is called with max_retries=2,
        THEN the result is returned after one retry and backoff sleep is called once.
        """
        agent = build_judge_agent()
        success_output = MagicMock()
        success_output.output = JudgeResult(results=[True, False])

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
            with patch("app.judge.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await get_judge_response(
                    agent,
                    ground_truth=["10", "20"],
                    predicted=["10", "WRONG"],
                    max_retries=2,
                )

        assert result is not None
        assert result.results == [True, False]
        mock_sleep.assert_awaited_once_with(1.0)

    async def test_get_judge_response_backoff_doubles_on_each_retry(self) -> None:
        """
        GIVEN an agent that raises RateLimitError on the first two calls then succeeds,
        WHEN get_judge_response is called with max_retries=3,
        THEN asyncio.sleep is called with 1.0 then 2.0, confirming exponential doubling.
        """
        agent = build_judge_agent()
        success_output = MagicMock()
        success_output.output = JudgeResult(results=[True])

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
            with patch("app.judge.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await get_judge_response(
                    agent,
                    ground_truth=["10"],
                    predicted=["10"],
                    max_retries=3,
                )

        assert result is not None
        assert mock_sleep.await_args_list == [call(1.0), call(2.0)]

    async def test_get_judge_response_raises_after_all_retries_exhausted(self) -> None:
        """
        GIVEN an agent that always raises RateLimitError,
        WHEN get_judge_response is called with max_retries=2,
        THEN RateLimitError is re-raised after the retry budget is exhausted.
        """
        agent = build_judge_agent()
        rate_limit_error = openai.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        with patch.object(agent, "run", side_effect=rate_limit_error):
            with patch("app.judge.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(openai.RateLimitError):
                    await get_judge_response(
                        agent,
                        ground_truth=["10"],
                        predicted=["10"],
                        max_retries=2,
                    )
